#!/usr/bin/env python
# coding=utf-8

"""
SDXL LoRA vs Base Model Comparison Script for Ghibli Style

사용법:
python inference_sdxl_ghibli.py --prompt "Draw a beautiful couple and their cat in a park in Ghibli style."
"""

import argparse
import os
import torch
from PIL import Image
from datetime import datetime
import json

from diffusers import StableDiffusionXLPipeline
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL LoRA inference comparison script for Ghibli style")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="output_ghibli/20250525_113655_ghibli_r8_a8.0_lr0.0001/final_model",
        help="Path to the trained LoRA model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Draw a beautiful couple and their cat in a park in Ghibli style.",
        help="Single text prompt for image generation.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs='+',
        default=['Draw a beautiful couple and their cat in a park in Ghibli style.',
                    'A young girl flying on a broomstick over a seaside town, in Ghibli animation style.',
                    'A quiet forest with a small cottage and a glowing spirit peeking from behind a tree, Ghibli-style.',
                    'A group of children chasing fireflies at dusk in a peaceful countryside setting, Ghibli-inspired.',
                    'An old train running through a field of golden wheat under a cloudy sky, in Ghibli art style.',
                    'A curious fox wearing a small backpack exploring a mossy ancient ruin, drawn in Ghibli style.',
                    'A calm lake reflecting the stars, with a girl sitting on a dock watching the sky, Ghibli-style.',
                    'A magical marketplace filled with unusual creatures and lanterns, illustrated in Ghibli animation style.',
                    'A flying island in the clouds with waterfalls cascading from its edges, in dreamy Ghibli aesthetic.',
                    'A boy riding a deer through a misty bamboo forest, in Studio Ghibli style.',
                    'An abandoned greenhouse slowly being taken over by nature, illustrated in warm Ghibli tones.',
                    'A cozy kitchen where a grandmother is baking bread with help from tiny forest spirits, Ghibli-inspired.',
                    'A mysterious black cat sitting on a rooftop watching over a rainy town, drawn like a Ghibli scene.',
                    'A child and a large fluffy creature sitting together under a giant mushroom during the rain, Ghibli-style.',
                    'A peaceful mountain village during a lantern festival at night, full of warm lights and soft details, Ghibli aesthetic.',
                    'An elderly couple walking their dog in a snowy forest, in Ghibli style.',
                    ],
        help="List of text prompts for image generation. If provided, will use these instead of single prompt.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, distorted, deformed, ugly, bad anatomy",
        help="Negative prompt for image generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=3,
        help="Number of images to generate per model.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of generated images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_visualization",
        help="Directory to save generated images. If not specified, will be created in the parent directory of lora_model_path.",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA scale factor (0.0 = no LoRA effect, 1.0 = full LoRA effect).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--run_base_inference",
        action='store_true',
        default=False,
        help="Whether to run base model inference or use saved results.",
    )
    parser.add_argument(
        "--base_results_dir",
        type=str,
        default="output_visualization/base_model_results",
        help="Directory containing saved base model results.",
    )
    parser.add_argument(
        "--create_comparison_grid",
        action='store_true',
        default=False,
        help="Whether to create comparison grid images.",
    )
    
    args = parser.parse_args()
    return args

def load_base_pipeline(args):
    """기본 SDXL 파이프라인을 로드합니다."""
    print(f"Loading base model: {args.pretrained_model_name_or_path}")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    pipeline = pipeline.to(args.device)
    return pipeline

def load_lora_pipeline(args):
    """LoRA 모델이 적용된 SDXL 파이프라인을 로드합니다."""
    print(f"Loading LoRA model: {args.lora_model_path}")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    
    if os.path.exists(args.lora_model_path):
        print(f"Loading LoRA weights from: {args.lora_model_path}")
        pipeline.load_lora_weights(args.lora_model_path)
        
        if args.lora_scale != 1.0:
            pipeline.fuse_lora(lora_scale=args.lora_scale)
        
        print(f"LoRA loaded successfully with scale: {args.lora_scale}")
        
        info_file = os.path.join(args.lora_model_path, "best_model_info.json")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                model_info = json.load(f)
            print(f"Model info: Epoch {model_info['epoch']}, Loss: {model_info['loss']:.4f}")
    else:
        print(f"Warning: LoRA model path not found: {args.lora_model_path}")
        raise FileNotFoundError(f"LoRA model not found at {args.lora_model_path}")
    
    pipeline = pipeline.to(args.device)
    return pipeline

def create_comparison_grid(base_images, lora_images, args, timestamp, prompt_idx):
    """기본 모델과 LoRA 모델의 결과를 비교하는 그리드를 생성합니다."""
    num_images = len(base_images)
    
    fig, axes = plt.subplots(2, num_images, figsize=(5*num_images, 10))
    
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    for i, image in enumerate(base_images):
        axes[0, i].imshow(image)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Base Model - Image {i+1}", fontsize=12, fontweight='bold')
    
    for i, image in enumerate(lora_images):
        axes[1, i].imshow(image)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"LoRA Model - Image {i+1}", fontsize=12, fontweight='bold')
    
    title = f"Ghibli Style Comparison - Prompt {prompt_idx + 1}\nPrompt: {args.prompts[prompt_idx][:60]}..."
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    filename = f"comparison_grid_prompt{prompt_idx+1}_{timestamp}.png"
    comparison_path = os.path.join(args.output_dir, filename)
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Comparison grid saved: {comparison_path}")
    
    plt.close()
    return comparison_path

def save_metadata(args, timestamp, base_paths, lora_paths, comparison_path, prompt_idx):
    """메타데이터를 저장합니다."""
    metadata = {
        "prompt": args.prompts[prompt_idx],
        "prompt_idx": prompt_idx + 1,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "lora_model_path": args.lora_model_path,
        "lora_scale": args.lora_scale,
        "timestamp": timestamp,
        "base_model_images": [os.path.basename(path) for path in base_paths],
        "lora_model_images": [os.path.basename(path) for path in lora_paths],
    }
    
    if comparison_path is not None:
        metadata["comparison_grid"] = os.path.basename(comparison_path)
    
    metadata_file = os.path.join(args.output_dir, f"metadata_prompt{prompt_idx+1}_{timestamp}.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved: {metadata_file}")
    return metadata_file

def main():
    args = parse_args()
    
    # output_dir이 지정되지 않은 경우 lora_model_path의 상위 디렉토리에 생성
    if args.output_dir is None:
        lora_parent_dir = os.path.dirname(os.path.dirname(args.lora_model_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(lora_parent_dir, f"inference_outputs_{timestamp}")
    
    # 프롬프트 리스트 설정
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = [args.prompt]
    
    print("=" * 80)
    print("SDXL LoRA vs Base Model Comparison Script for Ghibli Style")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nGeneration settings:")
    print(f"Number of prompts: {len(prompts)}")
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Seed: {args.seed}")
    print(f"Number of images per model per prompt: {args.num_images}")
    print(f"Run base inference: {args.run_base_inference}")
    
    try:
        print(f"\n{'='*80}")
        print("STEP 1: LOADING PIPELINES")
        print(f"{'='*80}")
        
        if args.run_base_inference:
            base_pipeline = load_base_pipeline(args)
        lora_pipeline = load_lora_pipeline(args)
        
        all_results = []
        
        for prompt_idx, current_prompt in enumerate(prompts):
            print(f"\n{'='*80}")
            print(f"PROCESSING PROMPT {prompt_idx + 1}/{len(prompts)}")
            print(f"{'='*80}")
            print(f"Current prompt: {current_prompt}")
            
            base_paths = []
            lora_paths = []
            
            # Base 모델 이미지 생성 또는 로드
            if args.run_base_inference:
                print(f"\n--- Generating {args.num_images} Base Model Images ---")
                for i in range(args.num_images):
                    print(f"Generating base model image {i+1}/{args.num_images}...")
                    seed_for_image = args.seed + (prompt_idx * args.num_images) + i
                    
                    generator = torch.Generator(device=args.device).manual_seed(seed_for_image)
                    
                    base_image = base_pipeline(
                        prompt=current_prompt,
                        negative_prompt=args.negative_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        num_images_per_prompt=1,
                        height=args.height,
                        width=args.width,
                        generator=generator,
                    ).images[0]
                    
                    base_filename = f"base_prompt{prompt_idx+1}_{timestamp}_{i+1}.png"
                    base_filepath = os.path.join(args.output_dir, base_filename)
                    base_image.save(base_filepath)
                    base_paths.append(base_filepath)
                    print(f"  ✓ Saved base image: {base_filepath}")
                    
                    del base_image
                    torch.cuda.empty_cache()
            else:
                print(f"\n--- Loading {args.num_images} Base Model Images ---")
                # 베이스 모델 결과 디렉토리에서 해당 프롬프트의 이미지 파일들을 찾습니다
                base_pattern = f"base_prompt{prompt_idx+1}_*.png"
                base_files = sorted([f for f in os.listdir(args.base_results_dir) if f.startswith(f"base_prompt{prompt_idx+1}_")])
                
                if len(base_files) < args.num_images:
                    raise FileNotFoundError(f"Not enough base model images found for prompt {prompt_idx+1}. Found {len(base_files)}, need {args.num_images}")
                
                # 필요한 수만큼의 이미지를 선택합니다
                for i in range(args.num_images):
                    base_filepath = os.path.join(args.base_results_dir, base_files[i])
                    base_paths.append(base_filepath)
                    print(f"  ✓ Loaded base image: {base_filepath}")
            
            # LoRA 모델 이미지 생성
            print(f"\n--- Generating {args.num_images} LoRA Model Images ---")
            for i in range(args.num_images):
                print(f"Generating LoRA model image {i+1}/{args.num_images}...")
                seed_for_image = args.seed + (prompt_idx * args.num_images) + i
                
                generator = torch.Generator(device=args.device).manual_seed(seed_for_image)
                
                lora_image = lora_pipeline(
                    prompt=current_prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                ).images[0]
                
                lora_filename = f"lora_prompt{prompt_idx+1}_{timestamp}_{i+1}.png"
                lora_filepath = os.path.join(args.output_dir, lora_filename)
                lora_image.save(lora_filepath)
                lora_paths.append(lora_filepath)
                print(f"  ✓ Saved LoRA image: {lora_filepath}")
                
                del lora_image
                torch.cuda.empty_cache()
            
            print(f"\n--- Creating Comparison Grid for Prompt {prompt_idx + 1} ---")
            base_images = [Image.open(path) for path in base_paths]
            lora_images = [Image.open(path) for path in lora_paths]
            
            if args.create_comparison_grid:
                comparison_path = create_comparison_grid(base_images, lora_images, args, timestamp, prompt_idx)
            else:
                comparison_path = None
            
            del base_images, lora_images
            torch.cuda.empty_cache()
            
            # 결과 저장
            result = {
                'prompt': current_prompt,
                'prompt_idx': prompt_idx + 1,
                'base_paths': base_paths,
                'lora_paths': lora_paths,
                'comparison_path': comparison_path
            }
            all_results.append(result)
            
            # 각 프롬프트별 메타데이터 저장
            save_metadata(args, timestamp, base_paths, lora_paths, comparison_path, prompt_idx)
            
            print(f"\n✓ COMPLETED PROMPT {prompt_idx + 1}/{len(prompts)}")
            print(f"  - Base images: {len(base_paths)}")
            print(f"  - LoRA images: {len(lora_paths)}")
            if args.create_comparison_grid:
                print(f"  - Comparison grid: 1")
        
        print("\n" + "="*80)
        print("GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Generated {args.num_images} images for each model for {len(prompts)} prompts")
        print(f"Total images generated: {args.num_images * 2 * len(prompts)}")
        print(f"All files saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
