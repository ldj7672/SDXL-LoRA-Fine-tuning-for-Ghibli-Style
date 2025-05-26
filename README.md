# SDXL LoRA Fine-tuning for Ghibli Style

이 프로젝트는 Stable Diffusion XL (SDXL) 모델을 LoRA로 fine-tuning하여 지브리 스타일의 이미지를 생성하는 실험을 진행합니다. 제한된 리소스와 작은 데이터셋으로 진행된 실험이지만, 다양한 설정에 따른 결과를 공유하여 연구 개발자들에게 도움이 되고자 합니다.

## 실험 결과

### 1. 베이스 모델과 LoRA 모델 비교

베이스 모델과 LoRA fine-tuning 모델의 비교 결과입니다.

![Comparison Result](results/SDXL_ghibli_result_0.png)
![Comparison Result](results/SDXL_ghibli_result_1.png)

위 이미지에서 상단은 베이스 모델의 결과, 하단은 LoRA fine-tuning 모델의 결과를 보여줍니다.

LoRA의 한계가 분명히 존재하지만, 100장이라는 적은 데이터셋으로도 지브리 스타일의 특성을 어느 정도 반영할 수 있음을 확인했습니다. 특히 색감, 선의 흐름, 전체적인 분위기 등에서 지브리 애니메이션 특유의 감성이 잘 표현되었습니다. 다만 LoRA 방식 특성상, 인물이나 객체의 형태가 무너지는 경우도 일부 발생합니다.

### 2. LoRA 파라미터 실험 결과

다음은 다양한 LoRA rank와 alpha 값 조합에 따른 실험 결과입니다.

![Experiment Results](results/exp_result.png)

## 실험 환경

- **모델**: Stable Diffusion XL Base 1.0
- [**데이터셋**](https://huggingface.co/datasets/moving-j/ghibli-style-100) 
  - 웹에서 수집한 지브리 스타일 이미지 100장
  - Google의 Gemini 모델을 사용하여 `utils/creat_text_caption.py` 스크립트로 각 이미지에 대한 텍스트 캡션 생성
- **Fine-tuning 방법**: LoRA (Low-Rank Adaptation)
- **실험 변수**:
  - LoRA Rank: 4, 8, 16
  - LoRA Alpha: 4, 8, 16, 32
  - Learning Rate: 1e-4

## 프로젝트 구조

```
.
├── train_sdxl_lora.py           # LoRA fine-tuning 학습 스크립트
├── inference_sdxl_ghibli.py     # 추론 및 비교 스크립트
├── run_experiments.sh           # 실험 실행 스크립트
├── run_checkpoint_inference.sh  # 체크포인트 추론 스크립트
├── hf/                          # Hugging Face 관련 스크립트
│   ├── hf_inference.py         # Hugging Face API 추론 스크립트
│   └── upload_to_hf.py         # Hugging Face 모델 업로드 스크립트
├── utils/                       # 유틸리티 스크립트
└── results/                     # 실험 결과 이미지
```

## 사용 방법

1. 학습 실행:
```bash
python train_sdxl_lora.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_type="ghibli" \
    --dataset_dir="dataset_ghibli" \
    --output_dir="output_ghibli" \
    --resolution=512 \
    --train_batch_size=1 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --lora_rank=4 \
    --lora_alpha=8.0
```

2. 추론 실행:
```bash
python inference_sdxl_ghibli.py --prompt "원하는 프롬프트"
```

3. Hugging Face API를 통한 추론:
```bash
python hf/hf_inference.py \
    --prompt "a beautiful landscape in Ghibli style" \
    --base_model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --lora_model_id "moving-j/sdxl-base-1.0-ghibli-lora-r4" \
    --token "your_hf_token" \
    --output "generated_image.png"
```

파인튜닝된 LoRA 모델은 [Hugging Face](https://huggingface.co/moving-j/sdxl-base-1.0-ghibli-lora-r4)에서 확인하실 수 있습니다.

## ⚠️ 제한사항

- 제한된 컴퓨팅 리소스로 인해 학습이 충분하지 않을 수 있습니다.
- 작은 데이터셋(100장)으로 인해 모델의 일반화 성능이 제한적일 수 있습니다.
- 본 레포지토리의 내용과 실험 결과는 학습 및 연구 목적의 참고용으로만 활용해 주세요.

---

# SDXL LoRA Fine-tuning for Ghibli Style

This project explores fine-tuning Stable Diffusion XL (SDXL) using LoRA to generate images in Ghibli style. While conducted with limited resources and a small dataset, we share our experimental results to help researchers and developers.

## Experimental Results

### 1. Base Model vs LoRA Model Comparison

Comparison between base model and LoRA fine-tuned model.

![Comparison Result](results/SDXL_ghibli_result_0.png)
![Comparison Result](results/SDXL_ghibli_result_1.png)

The image above shows base model results (top) and LoRA fine-tuned model results (bottom).

LoRA's limitations are evident, but our experiments demonstrate that even with a small dataset of 100 images, the model can capture certain characteristics of the Ghibli style. The generated images show promising results in terms of color palette, line flow, and overall atmosphere, reflecting distinctive elements of Studio Ghibli's animation style. However, due to the nature of LoRA, there are some cases where the form of characters or objects may break down.

### 2. LoRA Parameter Experiment Results

Different experiment results based on different LoRA rank and alpha combinations.

![Experiment Results](results/exp_result.png)

## Project Structure

```
.
├── train_sdxl_lora.py           # LoRA fine-tuning training script
├── inference_sdxl_ghibli.py     # Inference and comparison script
├── run_experiments.sh           # Experiment execution script
├── run_checkpoint_inference.sh  # Checkpoint inference script
├── hf/                          # Hugging Face related scripts
│   ├── hf_inference.py         # Hugging Face API inference script
│   └── upload_to_hf.py         # Hugging Face model upload script
├── utils/                       # Utility scripts
└── results/                     # Experimental result images
```

## Usage

1. Run training:
```bash
python train_sdxl_lora.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_type="ghibli" \
    --dataset_dir="dataset_ghibli" \
    --output_dir="output_ghibli" \
    --resolution=512 \
    --train_batch_size=1 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --lora_rank=4 \
    --lora_alpha=8.0
```

2. Run inference:
```bash
python inference_sdxl_ghibli.py --prompt "your prompt"
```

3. Inference using Hugging Face API:
```bash
python hf/hf_inference.py \
    --prompt "a beautiful landscape in Ghibli style" \
    --base_model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --lora_model_id "moving-j/sdxl-base-1.0-ghibli-lora-r4" \
    --token "your_hf_token" \
    --output "generated_image.png"
```

The fine-tuned LoRA model can be found on [Hugging Face](https://huggingface.co/moving-j/sdxl-base-1.0-ghibli-lora-r4).

## ⚠️ Limitations

- Training may be insufficient due to limited computing resources.
- Model generalization may be limited due to the small dataset (100 images).
- The experiments and results in this repository may have limitations and are intended for educational or research reference only. 