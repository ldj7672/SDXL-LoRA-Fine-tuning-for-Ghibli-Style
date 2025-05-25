#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import csv
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import Dataset
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import numpy as np

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from utils.datasets import ProductImageDataset, GhibliDataset

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")

logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")



def collate_fn(examples, dataset_type="product"):
    logger.debug(f"collate_fn called with {len(examples)} examples")
    try:
        if dataset_type == "product":
            # 제품 데이터셋용 collate
            images = torch.stack([example["image"] for example in examples])
            product_images = torch.stack([example["product_image"] for example in examples])
            texts = [example["text"] for example in examples]
            
            result = {
                "image": images,
                "product_image": product_images,
                "text": texts
            }
        else:  # ghibli
            # 지브리 데이터셋용 collate
            images = torch.stack([example["image"] for example in examples])
            texts = [example["text"] for example in examples]
            
            result = {
                "image": images,
                "text": texts
            }
        
        logger.debug(f"Successfully collated batch. Shapes: image={images.shape}, texts={len(texts)}")
        return result
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        logger.error(f"Example types: {[type(ex) for ex in examples]}")
        logger.error(f"Example keys: {[ex.keys() if isinstance(ex, dict) else 'not a dict' for ex in examples]}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL LoRA fine-tuning script for product styling.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["product", "ghibli"],
        default="ghibli",
        help="Type of dataset to use for training.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset_ghibli",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_ghibli",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=8.0,
        help="Alpha parameter for LoRA scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_k", "to_q", "to_v", "to_out.0"],
        help="List of module names to apply LoRA to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision training.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        default=True,
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2,
        help="Save checkpoint every X epochs.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory to save training logs.",
    )
    
    args = parser.parse_args()
    return args

def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts=0, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def main():
    args = parse_args()
    
    # 실험 이름 생성 (타임스탬프 + 데이터셋 타입 + 주요 설정)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{args.dataset_type}_r{args.lora_rank}_a{args.lora_alpha}_lr{args.learning_rate}"
    
    # 실험 디렉토리 생성
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 로깅 디렉토리 생성
    logging_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    
    # 아규먼트 정보 저장
    args_dict = vars(args)
    args_dict["experiment_name"] = experiment_name
    args_dict["timestamp"] = timestamp
    
    with open(os.path.join(experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 로그 파일 초기화 (메인 프로세스에서만)
    if accelerator.is_main_process:
        log_file = os.path.join(logging_dir, f"training_log.csv")
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'loss', 'learning_rate', 'timestamp'])
        
        # 최고 성능 모델 추적을 위한 변수
        best_loss = float('inf')
        best_model_path = os.path.join(experiment_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # 이미지 전처리를 위한 transform 정의
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Create dataset
    try:
        if args.dataset_type == "product":
            train_dataset = ProductImageDataset(args.dataset_dir, transform=image_transforms)
        else:  # ghibli
            train_dataset = GhibliDataset(args.dataset_dir, transform=image_transforms)
        logger.info(f"Successfully created {args.dataset_type} dataset with {len(train_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise
    
    # Load tokenizers and text encoders
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )
    
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # Apply LoRA to UNet using diffusers built-in adapter support (avoid PeftModel wrapper)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
    )
    unet.add_adapter(lora_config)
    
    # Freeze parameters
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.train()
    
    # Enable xformers if available
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # Enable gradient checkpointing if needed
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Create dataloader with collate_fn
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: collate_fn(x, args.dataset_type)  # 데이터셋 타입 전달
    )
    
    logger.info(f"Created dataloader with batch_size={args.train_batch_size}, num_workers={args.dataloader_num_workers}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )
    
    # Initialize scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * args.num_train_epochs,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move vae and text encoders to device
    vae.to(accelerator.device)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    
    # Training loop
    total_steps = len(train_dataloader) * args.num_train_epochs
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        epoch_losses = []  # 에폭별 손실 추적
        
        # 에폭별 progress bar
        epoch_progress = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            leave=False
        )
        
        for step, batch in epoch_progress:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["image"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    batch["text"],
                    [text_encoder_one, text_encoder_two],
                    [tokenizer_one, tokenizer_two],
                )

                # ✅ pooled_prompt_embeds 차원이 3D일 경우 2D로 변환
                if pooled_prompt_embeds.dim() == 3:
                    pooled_prompt_embeds = pooled_prompt_embeds.squeeze(1)  # [B, 1280]

                
                add_time_ids = torch.tensor([
                    args.resolution, args.resolution,  # original_size
                    0, 0,  # crops_coords_top_left
                    args.resolution, args.resolution,  # target_size
                ]).repeat(bsz, 1).to(accelerator.device)

                unet_added_conditions = {
                                            "text_embeds": pooled_prompt_embeds,   # 반드시 [B, 1280]
                                            "time_ids": add_time_ids               # 반드시 [B, 6]
                                        }

                
                # Predict noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                
                # Get target for loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                epoch_losses.append(loss.item())
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bars
            progress_bar.update(1)
            global_step += 1
            
            # Log loss
            if accelerator.is_local_main_process:
                current_lr = lr_scheduler.get_last_lr()[0]
                epoch_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Global Step': global_step
                })
                progress_bar.set_description(f"Epoch {epoch}/{args.num_train_epochs-1}, Step {step}, Loss: {loss.item():.4f}")
        
        # 에폭 종료 후 로깅 및 저장
        if accelerator.is_main_process:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            current_lr = lr_scheduler.get_last_lr()[0]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # CSV 로그 파일에 기록
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, global_step, avg_epoch_loss, current_lr, timestamp])
            
            logger.info(f"Epoch {epoch} completed - Avg Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.2e}")
            
            # 최고 성능 모델 저장
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                logger.info(f"New best model found! Loss: {best_loss:.4f} - Saving to {best_model_path}")
                
                # LoRA 가중치 추출 및 저장
                unwrapped_unet = accelerator.unwrap_model(unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
                
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=best_model_path,
                    unet_lora_layers=unet_lora_state_dict,
                    safe_serialization=True
                )
                
                # 최고 성능 정보 저장
                best_info = {
                    "epoch": epoch,
                    "loss": best_loss,
                    "learning_rate": current_lr,
                    "timestamp": timestamp
                }
                with open(os.path.join(best_model_path, "best_model_info.json"), 'w') as f:
                    json.dump(best_info, f, indent=2)
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % args.save_steps == 0:
                checkpoint_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # LoRA 가중치 추출 및 저장
                unwrapped_unet = accelerator.unwrap_model(unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
                
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=checkpoint_path,
                    unet_lora_layers=unet_lora_state_dict,
                    safe_serialization=True
                )
                
                # 체크포인트 정보 저장
                checkpoint_info = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": avg_epoch_loss,
                    "learning_rate": current_lr,
                    "timestamp": timestamp
                }
                with open(os.path.join(checkpoint_path, "checkpoint_info.json"), 'w') as f:
                    json.dump(checkpoint_info, f, indent=2)
                
                logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")
    
    # Save the final model
    if accelerator.is_main_process:
        final_model_path = os.path.join(experiment_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        # LoRA 가중치 추출 및 저장
        unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=final_model_path,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True
        )
        
        logger.info(f"Training completed! Final model saved to {final_model_path}")
        logger.info(f"Best model (Loss: {best_loss:.4f}) saved to {best_model_path}")
        logger.info(f"Training logs saved to {log_file}")
        logger.info(f"Experiment configuration saved to {os.path.join(experiment_dir, 'experiment_config.json')}")

if __name__ == "__main__":
    main()
