# SDXL LoRA Fine-tuning for Ghibli Style

이 프로젝트는 Stable Diffusion XL (SDXL) 모델을 LoRA로 fine-tuning하여 지브리 스타일의 이미지를 생성하는 실험을 진행합니다. 제한된 리소스와 작은 데이터셋으로 진행된 실험이지만, 다양한 설정에 따른 결과를 공유하여 연구 및 개발자들에게 도움이 되고자 합니다.

## 실험 결과

베이스 모델과 LoRA fine-tuning 모델의 비교 결과입니다:

![Comparison Result](results/comparison_grid_prompt1_20250525_121036.png)
![Comparison Result](results/comparison_grid_prompt2_20250525_121036.png)

위 이미지에서 상단은 베이스 모델의 결과, 하단은 LoRA fine-tuning 모델의 결과를 보여줍니다.

LoRA의 한계가 분명히 존재하지만, 100장이라는 적은 데이터셋으로도 지브리 스타일의 특성을 어느 정도 포착하여 생성할 수 있음을 확인했습니다. 특히 색감, 선의 표현, 그리고 전반적인 분위기에서 지브리 애니메이션의 특징적인 요소들이 드러나는 것을 볼 수 있습니다.

## 실험 환경

- **모델**: Stable Diffusion XL Base 1.0
- **데이터셋**: 
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
├── train_sdxl_lora.py      # LoRA fine-tuning 학습 스크립트
├── inference_sdxl_ghibli.py # 추론 및 비교 스크립트
├── run_experiments.sh      # 실험 실행 스크립트
├── dataset_ghibli/         # 데이터셋 디렉토리
│   ├── images/            # 이미지 파일들
│   └── captions.json      # 이미지 캡션
└── output_ghibli/         # 학습 결과 저장 디렉토리
```

## 사용 방법

1. 학습 실행:
```bash
./run_experiments.sh
```

2. 추론 실행:
```bash
python inference_sdxl_ghibli.py --prompt "원하는 프롬프트"
```

## 제한사항

- 제한된 컴퓨팅 리소스로 인해 학습이 충분하지 않을 수 있습니다.
- 작은 데이터셋(100장)으로 인해 모델의 일반화 성능이 제한적일 수 있습니다.
- 실험 결과는 참고용으로만 사용해 주시기 바랍니다.

## Coming Soon 🚀

- [ ] 데이터셋 공개 (100장의 지브리 스타일 이미지)
- [ ] 실험 설정별 상세 결과 및 분석
- [ ] 추가 실험 결과 및 모델 업데이트

---

# SDXL LoRA Fine-tuning for Ghibli Style

This project explores fine-tuning Stable Diffusion XL (SDXL) using LoRA to generate images in Ghibli style. While conducted with limited resources and a small dataset, we share our experimental results to help researchers and developers.

## Experimental Results

Comparison between base model and LoRA fine-tuned model:

![Comparison Result](results/comparison_grid_prompt2_20250525_121036.png)

The image above shows base model results (top) and LoRA fine-tuned model results (bottom).

While the limitations of LoRA are evident, our experiments demonstrate that even with a small dataset of 100 images, the model can capture certain characteristics of the Ghibli style. The generated images show promising results in terms of color palette, line work, and overall atmosphere, reflecting distinctive elements of Studio Ghibli's animation style.

## Experimental Setup

- **Model**: Stable Diffusion XL Base 1.0
- **Dataset**: 
  - 100 Ghibli-style images collected from the web
  - Text captions generated for each image using Google's Gemini model via `utils/creat_text_caption.py` script
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Experimental Variables**:
  - LoRA Rank: 4, 8, 16
  - LoRA Alpha: 4, 8, 16, 32
  - Learning Rate: 1e-4

## Project Structure

```
.
├── train_sdxl_lora.py      # LoRA fine-tuning training script
├── inference_sdxl_ghibli.py # Inference and comparison script
├── run_experiments.sh      # Experiment execution script
├── dataset_ghibli/         # Dataset directory
│   ├── images/            # Image files
│   └── captions.json      # Image captions
└── output_ghibli/         # Training output directory
```

## Usage

1. Run training:
```bash
./run_experiments.sh
```

2. Run inference:
```bash
python inference_sdxl_ghibli.py --prompt "your prompt"
```

## Limitations

- Training may be insufficient due to limited computing resources.
- Model generalization may be limited due to the small dataset (100 images).
- Experimental results should be used for reference only.

## Coming Soon 🚀

- [ ] Dataset release (100 Ghibli-style images)
- [ ] Detailed results and analysis for each experimental setting
- [ ] Additional experimental results and model updates 