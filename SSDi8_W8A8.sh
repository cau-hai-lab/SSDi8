mkdir -p logs

# 배치 사이즈
BATCH_SIZES=(16)

# 평가할 모델 리스트
MODELS=(
    "state-spaces/mamba2-1.3B"
    "state-spaces/mamba2-2.7B"
    "pretrained_models/mamba2-8b-converted"
)

for model in "${MODELS[@]}"; do
    modelname=$(basename "$model")

    for BS in "${BATCH_SIZES[@]}"; do
        echo "==== Running ${model} with batch_size=${BS} ===="

        python -W ignore main.py "$model" \
            --batch_size "$BS" \
            --eval_ppl \
            --quantize \
            --eval_zero_shot \
            --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande \
            --log_dir logs \
            --w_bits 8 \
            --a_bits 8 \
            --apply_hadamard \
            --compensation \
            --comp_out_decay 0.5 \
            2>&1 | tee "logs/88SSDim${modelname}${BS}.log"
    done
done
