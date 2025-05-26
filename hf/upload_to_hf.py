 import os
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def upload_model_to_hf(
    model_path: str,
    repo_name: str,
    token: str,
    private: bool = False
):
    """
    LoRA 모델을 Hugging Face에 업로드합니다.
    
    Args:
        model_path: 로컬 모델 경로
        repo_name: Hugging Face 저장소 이름
        token: Hugging Face API 토큰
        private: 비공개 저장소 여부
    """
    api = HfApi()
    
    # 모델 파일 업로드
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        token=token
    )
    
    print(f"모델이 성공적으로 업로드되었습니다: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="로컬 모델 경로")
    parser.add_argument("--repo_name", type=str, required=True, help="Hugging Face 저장소 이름")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API 토큰")
    parser.add_argument("--private", action="store_true", help="비공개 저장소로 생성")
    
    args = parser.parse_args()
    upload_model_to_hf(args.model_path, args.repo_name, args.token, args.private)