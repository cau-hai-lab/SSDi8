"""
This file is a modified version of the original file from the Quamba repo.
https://github.com/enyac-group/Quamba
"""
import os
import json
import random
import argparse
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from ssdi8.megatron_utils import _GPTSentencePieceTokenizer


def build_mamba_and_tokenizer(args, model_type="mamba"):
    is_ssdi8 = False
    device = "cuda"
    dtype = torch.float16 # use half, otherwise real quant won't run
    if model_type == "mamba" or model_type == "mamba2":
        if "mamba2-8b" not in args.model:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        else:
            # NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
            # FIXME: hardcode the tokenizer file name for now
            tokenizer_ckpt = os.path.join(args.model, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            tokenizer = _GPTSentencePieceTokenizer(tokenizer_ckpt)
        model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2''")
    return model, tokenizer, is_ssdi8

def build_mamba_and_tokenizer_hybrid(args, model_type="mamba"):
    is_ssdi8 = False
    device = "cuda"
    dtype = torch.float16 # use half, otherwise real quant won't run
    if model_type == "mamba" or model_type == "mamba2":
        if "mamba2-8b" not in args.model:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        else:
            # NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
            # FIXME: hardcode the tokenizer file name for now
            tokenizer_ckpt = os.path.join(args.model, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            tokenizer = _GPTSentencePieceTokenizer(tokenizer_ckpt)
        model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2'")
    return model, tokenizer, is_ssdi8

def set_deterministic(seed):
    # Fix all possible random seef for reproduce
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_quantize_options(parser):
    # quantization parameters
    parser.add_argument(
        '--quantize', action='store_true', default=False,
    )
    # calibration parameters
    parser.add_argument(
        '--calib_data_num', type=int, default=512,
        help='Number of calibration data (default: 512)'
    )
    parser.add_argument(
        '--calib_seqlen', type=int, default=512,
        help='Maximum sequence length for calibration data (default: 512)'
    )
    # load/store model
    parser.add_argument(
        '--pretrained_dir', type=str, default=None,
        help='The path to store both the quantized model and its act_scales_cache.'
        'Not storing if not provided. (default: None)'
    )
    # quantization parameters
    parser.add_argument(
        "--quantize_embedding", action='store_true', default=False,
        help="Whether to quantize the embedding layer (default: False)"
    )
    parser.add_argument(
        "--quantize_lm_head", action='store_true', default=False,
        help="Whether to quantize the lm_head layer (default: False)"
    )
    parser.add_argument(
        '--apply_gptq',  action='store_true', default=False,
        help='Whether to apply the GPTQ quantizer (default: False)'
    )
    parser.add_argument(
        '--w_bits', type=int, default=8,
        help='The bit-width for weights applied in the real quantization (default: 8)'
    )
    parser.add_argument(
        '--a_bits', type=int, default=8,
        help='The bit-width for activations applied in the real quantization (default: 8)'
    )

    parser.add_argument(
        '--apply_hadamard', action='store_true', default=False,
        help='Whether to apply the Hadamard transform for quantization (default: False)'
    )
    
    ##COMP##
    parser.add_argument(
        '--compensation', action='store_true', default=False,
    )
    parser.add_argument(
        '--comp_out_decay', type=float, default=0.1
    )
    parser.add_argument(
        '--comp_sam_num', type=int, default=512
    )
    parser.add_argument(
        '--comp_layers',
        type=int,          # 요소 타입
        nargs='+',         # 공백으로 구분된 여러 개 받기
        default=None,
        help="보상 적용할 레이어 인덱스 (예: --comp_layer 0 1 2 3)"
    )
    ##COMP##
    

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='Mamba to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to print the debug level information'
    )
    ##### General Evaluation Settings #####
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--task_list', type=lambda s: [item for item in s.split(',')], default=["lambada_openai"],
        help='Task to be evaled, e.g., --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande'
    )
    parser.add_argument(
        '--eval_zero_shot', action='store_true', default=False,
        help='Whether to evaluate the zero-shot performance Task(s) specified by `--tasks_list`, e.g, --tasks_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande'
    )
    parser.add_argument(
        '--eval_few_shot', action='store_true', default=False,
        help='Whether to evaluate the few-shot performance. Task(s) specified by `--tasks_list` and `--fewshot`'
    )
    parser.add_argument(
        '--fewshot', type=int, default=0,
        help='Number of shots for few-shot evaluation (0 for zero-shot)'
    )
    parser.add_argument(
        '--eval_generation', action='store_true', default=False,
        help='Whether to evaluate the performance of the generation tasks. Task(s) specified by `--tasks_list`, e.g, --tasks_list nq_open,squadv2'
    )
    parser.add_argument(
        '--testing', action='store_true',
        help='testing with decreased sample count'
    )
    parser.add_argument(
        '--eval_ppl', action='store_true', default=False,
        help='Whether to evaluate the wikitext2 ppl'
    )
    parser.add_argument(
        '--ppl_dataset', type=str, default='wikitext2',
        help='Dataset for ppl evaluation'
    )
    parser.add_argument(
        '--log_dir', type=str,
        help='path to the json log file storing the result of lm_evan and quantization settingarg'
    )
    parser.add_argument(
        '--ppl_seq_len', type=int, default=2048,
        help='Fixed block length for perplexity eval (default: 2048)'
    )
    parser.add_argument(
        '--plot', action='store_true', default=False,
    )
    get_quantize_options(parser)
    args = parser.parse_args()
    return args
    