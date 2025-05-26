import os
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login

def generate_image_with_hf_api(
    prompt: str,
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_model_id: str = None,
    token: str = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: str = None
):
    """
    Hugging Face API를 사용하여 이미지를 생성합니다.
    
    Args:
        prompt: 이미지 생성을 위한 프롬프트
        base_model_id: 베이스 모델 ID
        lora_model_id: LoRA 모델 ID
        token: Hugging Face API 토큰
        num_inference_steps: 추론 스텝 수
        guidance_scale: 가이던스 스케일
        negative_prompt: 네거티브 프롬프트
    """
    if token:
        login(token=token)
    
    # 베이스 모델 로드
    pipe = DiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # LoRA 모델이 있다면 로드
    if lora_model_id:
        pipe.load_lora_weights(lora_model_id)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # 이미지 생성
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    return image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="이미지 생성을 위한 프롬프트")
    parser.add_argument("--base_model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="베이스 모델 ID")
    parser.add_argument("--lora_model_id", type=str, default=None, help="LoRA 모델 ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API 토큰")
    parser.add_argument("--output", type=str, default="output.png", help="출력 이미지 경로")
    parser.add_argument("--steps", type=int, default=50, help="추론 스텝 수")
    parser.add_argument("--guidance", type=float, default=7.5, help="가이던스 스케일")
    parser.add_argument("--negative_prompt", type=str, default=None, help="네거티브 프롬프트")
    
    args = parser.parse_args()
    
    image = generate_image_with_hf_api(
        prompt=args.prompt,
        base_model_id=args.base_model_id,
        lora_model_id=args.lora_model_id,
        token=args.token,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        negative_prompt=args.negative_prompt
    )
    
    image.save(args.output)
    print(f"이미지가 저장되었습니다: {args.output}") 