export PYTHONNOUSERSITE=1
mkdir -p logs

# 배치 사이즈 리스트
BATCH_SIZES=(16)

# 모델 리스트
MODELS=(
    "state-spaces/mamba2-130m"
)

for model in "${MODELS[@]}"; do
    modelname=$(basename "$model")

    for BS in "${BATCH_SIZES[@]}"; do
        echo "==== Running ${model} with batch_size=${BS} ===="

        python -W ignore main.py "$model" \
            --batch_size "$BS" \
            --eval_ppl \
            --quantize \
            --log_dir logs \
            --w_bits 4 \
            --a_bits 8 \
            --apply_hadamard \
            --apply_gptq \
            --compensation \
            --comp_out_decay 0.5 \
            2>&1 | tee "logs/48SSDim${modelname}${BS}.log"
    done
done
