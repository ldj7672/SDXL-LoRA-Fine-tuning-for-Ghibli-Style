#!/bin/bash

# 기본 설정
BASE_DIR="output_ghibli/20250525_152322_ghibli_r8_a8.0_lr0.0001"
OUTPUT_DIR="output_visualization/20250525_152322_ghibli_r8_a8.0_lr0.0001_checkpoints"
BASE_RESULTS_DIR="output_visualization/base_model_results"

# 체크포인트 에포크 목록
CHECKPOINTS=(1 3 5 7 9)

PROMPTS=(
    "Draw a beautiful couple and their cat in a park in Ghibli style."
    "A young girl flying on a broomstick over a seaside town, in Ghibli animation style."
    "A quiet forest with a small cottage and a glowing spirit peeking from behind a tree, Ghibli-style."
    "A group of children chasing fireflies at dusk in a peaceful countryside setting, Ghibli-inspired."
    "An old train running through a field of golden wheat under a cloudy sky, in Ghibli art style."
    "A curious fox wearing a small backpack exploring a mossy ancient ruin, drawn in Ghibli style."
    "A calm lake reflecting the stars, with a girl sitting on a dock watching the sky, Ghibli-style."
    "A magical marketplace filled with unusual creatures and lanterns, illustrated in Ghibli animation style."
    "A flying island in the clouds with waterfalls cascading from its edges, in dreamy Ghibli aesthetic."
    "A boy riding a deer through a misty bamboo forest, in Studio Ghibli style."
    "An abandoned greenhouse slowly being taken over by nature, illustrated in warm Ghibli tones."
    "A cozy kitchen where a grandmother is baking bread with help from tiny forest spirits, Ghibli-inspired."
    "A mysterious black cat sitting on a rooftop watching over a rainy town, drawn like a Ghibli scene."
    "A child and a large fluffy creature sitting together under a giant mushroom during the rain, Ghibli-style."
    "A peaceful mountain village during a lantern festival at night, full of warm lights and soft details, Ghibli aesthetic."
    "An elderly couple walking their dog in a snowy forest, in Ghibli style."
    "A couple studying in a Japanese cafe in Ghibli style."
    "A male protagonist who rides a dragon and ascends to the sky in Ghibli style."
    "Japanese girl lying in a field in Ghibli style."
    "A couple's selfie in Ghibli style. The man is wearing glasses and a checkered shirt, and the woman has her hair down and is wearing a one-piece dress."
    "The image is a full shot of Kiki from Kiki's Delivery Service, standing in a field wearing her signature red dress and bow with her messenger bag. The image is in the style of Studio Ghibli, with soft colors and a focus on nature."
    "The image is a Ghibli-style illustration of a young girl with a red bow in her hair, riding on a broomstick with a black cat in her satchel, accompanied by a large seagull and another bird in the background. The scene is set against a backdrop of a coastal town and blue water."
    "The image is a Ghibli-style illustration of a young woman with dark hair and a white hat with a lavender ribbon. She is wearing a light yellow dress with a blue collar, and the background features a blue sky with fluffy white clouds and green trees."
    "The image is a portrait of a young woman in Ghibli style, with black hair and a striped shirt, set against a backdrop of a room with various decorations. The overall aesthetic is reminiscent of classic anime films."
    "The image is an animation in Ghibli style, featuring two characters sitting at a desk. On the left is a smiling boy, and on the right is a man with a stern expression, wearing a watch."
)

# 각 체크포인트에 대해 인퍼런스 실행
for epoch in "${CHECKPOINTS[@]}"; do
    # 체크포인트 모델 경로
    checkpoint_path="${BASE_DIR}/checkpoint_epoch_${epoch}"
    
    # 출력 디렉토리 생성
    output_path="${OUTPUT_DIR}/epoch_${epoch}"
    mkdir -p "${output_path}"
    
    echo "=========================================="
    echo "Running inference for checkpoint: epoch_${epoch}"
    echo "Checkpoint path: ${checkpoint_path}"
    echo "Output directory: ${output_path}"
    echo "=========================================="
    
    # 인퍼런스 실행
    python inference_sdxl_ghibli.py \
        --lora_model_path "${checkpoint_path}" \
        --output_dir "${output_path}" \
        --base_results_dir "${BASE_RESULTS_DIR}" \
        --prompts "${PROMPTS[@]}" \
        --height 1024 \
        --width 1024 \
    
    echo "Completed inference for epoch_${epoch}"
    echo "=========================================="
done

echo "All checkpoint experiments completed!" 