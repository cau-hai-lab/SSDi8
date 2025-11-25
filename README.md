"""
This file is a modified version of the original file from the Quamba repo.
https://github.com/enyac-group/Quamba
"""

# SSDi8
- 🔧 Supports W4A8 / W8A8 for Mamba2



## Setup

### Hardware Requirements
- NVIDIA GPU Ampere architecture 

### Software Requirements
- CUDA 12.1 or above
- CMAKE version 3.22.1 or above

### Clone SSDi8
- Clone the repository with all submodules:
```bash
cd SSDi8
###Need SSH key
git submodule update --init --recursive
```

- Create conda environment
```bash
cd SSDi8
conda create -n ssdi8 python=3.10
conda activate ssdi8
pip install -r requirements.txt
```

### Build 3rd-party Libraries

- Install `fast-hadamard-transform`:
```bash
# set force build to include 12N, 40N from the newer commit
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
pip install 3rdparty/fast-hadamard-transform
```

- Install `lm-evaluation-harness`:
```bash
# lm_eval-0.4.2 word2number-1.1
pip install 3rdparty/lm-evaluation-harness
``````

- Install mamba
```bash
export MAMBA_FORCE_BUILD=TRUE
pip install 3rdparty/mamba
```

- Install CUTLASS
```bash
# cmake version >= 3.22.1
install -c conda-forge cmake=3.22.1 ninja
bash build_cutlass.sh
```

- Install Megatron-LM
```bash
pip install -e 3rdparty/Megatron-LM
# Not sure why Megatron-LM will force to install pytorch 2.6.0+cu124,
# run `pip install -r requirements.txt` again if necessary
```

```bash
pip install .
pip uninstall -y causal-conv1d causal_conv1d || true
## Need causal-conv1d-1.5.0.post8
pip install --no-build-isolation "causal-conv1d==1.5.0.post8"
```
- Triton update
```bash
pip uninstall -y triton
pip install triton==3.4.0
```




### Convert Nvidia Mamba2-8B to HuggingFace

Download the checkpoint using `huggingface-cli`
```bash
huggingface-cli download nvidia/mamba2-8b-3t-4k --local-dir ./pretrained_models/mamba2-8b-3t-4k
```
After downloading, you will have the directory `./pretrained_models/mamba2-8b-3t-4k` having a structure like this
```bash
├── latest_checkpointed_iteration.txt
├── mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model (This is tokenizer)
├── README.md
└── release
    └── mp_rank_00
        └── model_optim_rng.pt (This is weights)
```
+ Run the conversion scripts to get the model directory
```bash
python convert_mamba2_8b_to_hf.py \
./pretrained_models/mamba2-8b-3t-4k/release/mp_rank_00/model_optim_rng.pt \
./pretrained_models/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
--model_save_path ./pretrained_models/mamba2-8b-converted
```

### Quantize and Evaluate Mamba2-8B

After running, you will see a directory called `mamba2-8b-converted` has been created. Then you can run it with evaluation, profiling as the instructions above. However, it requires at least *24GB* memory on the GPU to quantize the Mamba2-8b model.

For example:
```bash
python main.py pretrained_models/mamba2-8b-converted \
--batch_size 16 \
--eval_zero_shot \
--task_list lambada_openai \
--quantize \
--apply_gptq \
--w_bits 4 \
--a_bits 8
--log_dir logs
``` 



