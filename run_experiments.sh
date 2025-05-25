#!/bin/bash

# 실험 시작 시간 기록
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "실험 시작 시간: $START_TIME"

# 실험 결과를 저장할 디렉토리 생성
RESULTS_DIR="experiment_results"
mkdir -p $RESULTS_DIR

# 실험 설정
LORA_RANKS=(4 8 16)
LORA_ALPHAS=(4 8 16 32)
LEARNING_RATE=1e-4

# 전체 실험 수 계산
TOTAL_EXPERIMENTS=$((${#LORA_RANKS[@]} * ${#LORA_ALPHAS[@]}))
echo "총 $TOTAL_EXPERIMENTS 개의 실험을 실행합니다."

# 실험 카운터 초기화
CURRENT_EXPERIMENT=1

# 각 실험 실행
for rank in "${LORA_RANKS[@]}"; do
    for alpha in "${LORA_ALPHAS[@]}"; do
        echo "================================================"
        echo "실험 $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
        echo "Rank: $rank, Alpha: $alpha, Learning Rate: $LEARNING_RATE"
        echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================"

        # 실험 실행
        python train_sdxl_lora.py \
            --lora_rank $rank \
            --lora_alpha $alpha \
            --learning_rate $LEARNING_RATE

        # 실험 결과 저장
        if [ $? -eq 0 ]; then
            STATUS="성공"
        else
            STATUS="실패"
        fi

        # 실험 정보를 JSON 파일에 저장
        echo "{
            \"experiment_number\": $CURRENT_EXPERIMENT,
            \"lora_rank\": $rank,
            \"lora_alpha\": $alpha,
            \"learning_rate\": $LEARNING_RATE,
            \"start_time\": \"$(date '+%Y-%m-%d %H:%M:%S')\",
            \"status\": \"$STATUS\"
        }" >> "$RESULTS_DIR/experiment_${rank}_${alpha}.json"

        echo "실험 완료: $STATUS"
        echo "================================================"

        # 다음 실험 전 대기
        sleep 5

        # 실험 카운터 증가
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    done
done

# 전체 실험 종료 시간 기록
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "실험 종료 시간: $END_TIME"

# 결과 요약 생성
echo "{
    \"total_experiments\": $TOTAL_EXPERIMENTS,
    \"start_time\": \"$START_TIME\",
    \"end_time\": \"$END_TIME\"
}" > "$RESULTS_DIR/summary.json"

echo "모든 실험이 완료되었습니다."
echo "결과는 $RESULTS_DIR 디렉토리에 저장되었습니다." 