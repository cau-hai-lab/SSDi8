"""
This file is a modified version of the original file from the Quamba repo.
https://github.com/enyac-group/Quamba
"""

import os
import gc
import logging
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from datasets import load_dataset

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm

from .qLinearLayer import HadLinear
from .qActLayer import ActIdentity
from .qMamba2 import Mamba2Simple, W4A8QMamba2,W8A8QMamba2
from .qNorm import QRMSNorm
from .observer import PerTensorMinmaxObserver, PerTensorPercentileObserver
from .observer import _ChannelMeanCollector
from .observer import ChunkCollector
from .gptq_utils import GPTQ
from .hadamard_utils import had_transform
from .data_loaders import get_loaders


logger = logging.getLogger(__name__)

@torch.no_grad()
def fuse_ln_linear(norm, linear) -> None:
    """
    fuse the layernorm weight to the adjacent linear layer.
    """
    linear_dtype = linear.weight.dtype

    # Calculating new weight and bias
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ * norm.weight.double()).to(linear_dtype)  
    if hasattr(norm, 'bias') and norm.bias is not None:
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float32))
        linear.bias.data = linear.bias.data.double() + torch.matmul(W_, norm.bias.to(torch.float32))
        linear.bias.data = linear.bias.data.to(linear_dtype)
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1
@torch.no_grad()
def configure_model(model, model_type, use_had_transform=True):
    device = next(model.parameters()).device
    if model_type == "mamba":
        raise ValueError(f"Unsupported - mamba1 ")
    elif model_type == "mamba2":
        # process embedding and lm_head
        if use_had_transform:            
            # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_clone = model.lm_head.weight.data.clone()
            # transform embedding first
            model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data) #입력 임베딩 회전
            # do layernorm fusion to lm_head and then transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
            ##LM-Head 와 RMSNorm 합치기
            model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
            ##RMSNorm 합쳤으니까 1로만들기
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            ##다시한번 회전
            torch.cuda.empty_cache()
        # process layers
        
        layers = model.backbone.layers
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj) 
                m = Mamba2Simple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
    model.config.use_cache = False
    model.eval()

    print("mdoel=,",model)
    return model

@torch.no_grad()
def run_calibration(
        model, model_type, tokenizer, num_samples=512, seq_len=2048,
        calibration_dataset=None, preprocess_fn=None
    ):

    if model_type == "mamba":
        raise ValueError(f"Unsupported - mamba1 ")
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        is_x = lambda op: op == "x_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995  # for smaller model like 130m, use 0.99999
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")

    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]
    
    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    fp_proj_mean_hooks = {}
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        mixer = get_mamba(layers[i])
        x_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        ori_x_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        B_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        C_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        state_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        cb_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        
        ##DO
        _ssd_col = _ChannelMeanCollector()
        _out_col = _ChannelMeanCollector()

        h_ssd = mixer.ssd_out_act.register_forward_hook(
            lambda _m, _i, out, col=_ssd_col: col.update(out)
        )
        h_out = mixer.out_proj.register_forward_hook(
            lambda _m, _i, out, col=_out_col: col.update(out)
        )
        fp_proj_mean_hooks[f"L{i}/ssd_out"] = (_ssd_col, h_ssd)
        fp_proj_mean_hooks[f"L{i}/out_proj_out"] = (_out_col, h_out)
        
        ### COMP ###


        # mixer에 달아두기 → forward에서 collect_obs로 전달됨
        mixer._collect_obs = {"ori_x_chp": ori_x_col,"x_chp": x_col, "B_cgn": B_col,"C_cali":C_cali,"state_cali":state_cali,"cb_cali":cb_cali}
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op) or is_ssm_state(op):
                    observers[i][op + ":input"] = PerTensorPercentileObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        percentile_alpha=percentile_alpha
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )

    device = next(model.parameters()).device
    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        # prepare inference cache for getting ssm_state scales
        prompt_len = input_ids.size(1)
        # inf_cache = model.allocate_inference_cache(1, prompt_len)
        lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
        inference_params = InferenceParams(
            max_seqlen=prompt_len,
            max_batch_size=1,
            seqlen_offset=0,
            # key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        # do not set num_last_tokens because we want all activations to lm_head
        model(input_ids, inference_params=inference_params)
        # clean up the cache
        # del inf_cache
    
    for h in hooks:
        h.remove()
        
    
    fp_means = {}
    for k,(collector,handle) in fp_proj_mean_hooks.items():
        handle.remove()
        fp_means[k] = collector.mean
    
    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale.to(torch.float32)
        if i < len(layers):
            col = getattr(get_mamba(layers[i]), "_collect_obs", None)
            if col is not None:
                act_scales[i]["chunk_state:x_scale_chp"] = col["x_chp"].get_scale()  # (C,H,P)
                act_scales[i]["chunk_state:ori_x_scale_chp"] = col["ori_x_chp"].get_scale()  # (C,H,P)
                act_scales[i]["chunk_state:B_scale_cgn"] = col["B_cgn"].get_scale()  # (C,G,N)
                act_scales[i]["chunk_scan:C_scale"] = col["C_cali"].get_scale()  # (C,H,P)
                act_scales[i]["chunk_scan:cb_scale"] = col["cb_cali"].get_scale()  # (C,H,P)
                act_scales[i]["ssd_combined:state_scale"] = col["state_cali"].get_scale()  # (C,H,P)
                # 메모리/부작용 방지
                delattr(get_mamba(layers[i]), "_collect_obs")
    del observers
    return act_scales, fp_means


@torch.no_grad()
def fuse_had_matrices(model):
    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gptq
    layers = model.backbone.layers
    for i in range(len(layers)):
        # in_proj: fuse had matrices with weight matrices
        if isinstance(layers[i].mixer.in_proj, HadLinear):
            layers[i].mixer.in_proj.fuse_hadamard()
        # out_proj: fuse had matrices with weight matrices
        if isinstance(layers[i].mixer.out_proj, HadLinear):
            layers[i].mixer.out_proj.fuse_hadamard()
    return model

@torch.no_grad()
def apply_gptq(model, tokenizer, device,args,w_bits=4,):
    # Hardcode gptq hyper-parameters for now
    nsamples = 128
    seqlen = 1024
    bits = w_bits
    assert bits in [4, 8], "Only support 4 or 8 bits weights for now"
    logging.info("Start Quantized Linear Layers with GPTQ")
    logging.info("* Number of samples: %d" % nsamples)
    logging.info("* Sequence length: %d" % seqlen)
    logging.info("* Target bit-width for weights: %d" % bits)
    logging.info("Build calibration loader for GPTQ")
    #build dataloader
    dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen)
    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )    
    residual = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )    

    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module  
        def forward(self, inp, res = None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    # the hook to collect inputs for in_proj, out_proj, and lm_head
    def add_batch(module, inp, out, gptq):
        gptq.add_batch(inp[0].data, out.data)

    layers[0] = layers[0].module # remove Catcher
    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(device)
        gptq = {
            "in_proj": GPTQ(layer.mixer.in_proj),
            "out_proj": GPTQ(layer.mixer.out_proj),
        }
        handles = [
            layer.mixer.in_proj.register_forward_hook(partial(add_batch, gptq=gptq["in_proj"])),
            layer.mixer.out_proj.register_forward_hook(partial(add_batch, gptq=gptq["out_proj"]))
        ]


        for j in range(nsamples):
            layer(
                inps[j].unsqueeze(0), 
                residual=residual[j].unsqueeze(0)
            )
        for h in handles:
            h.remove()   
                                          
        
        # start running GPTQ
        for name in gptq.keys():
            logging.debug(f"Performing GPTQ on layer.{i}.mixer.{name} with {bits} bits")
            gptq[name].fasterquant(
                percdamp=0.01, group_size=128, w_bits=bits
            )
            gptq[name].free()
        del gptq
        
        # collect the outputs for the next layer
        for j in range(nsamples):
            inps[j], residual[j] = layer(inps[j].unsqueeze(0), residual=residual[j].unsqueeze(0))
        
        # garbage collection and clean cache
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        gc.collect()

    model = model.to("cpu") # move model to cpu to save memory
    model.lm_head = model.lm_head.to(device)
    model.backbone.norm_f = model.backbone.norm_f.to(device)
    logging.info("Quantizing lm_head with GPTQ")
    gptq_lm_head = GPTQ(model.lm_head)
    handle = model.lm_head.register_forward_hook(partial(add_batch, gptq=gptq_lm_head))
    
    assert model.backbone.fused_add_norm, "Only support fused_add_norm=True for now"
    #Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L202
    final_hidden_states = layer_norm_fn(
        x=inps,
        weight=model.backbone.norm_f.weight,
        bias=model.backbone.norm_f.bias,
        eps=model.backbone.norm_f.eps,
        residual=residual,
        prenorm=False,
        residual_in_fp32=model.backbone.residual_in_fp32,
        is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm),
    )
    
    for j in range(nsamples):
        model.lm_head(final_hidden_states[j].unsqueeze(0))

    handle.remove()
    # compute with fp16 to save memory
    gptq_lm_head.fasterquant(
        percdamp=0.01, group_size=128, dtype=torch.float16
    )
    gptq_lm_head.free()
    del gptq_lm_head
    
    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    return model


def quantize_norm_a8(block_type, norm, layer_idx, act_scales, device):
    norm = QRMSNorm.from_fp16(
        originalLayer=norm,
        output_scale=act_scales[layer_idx]["in_proj:input"].item())
    return norm.to(device)


def quantize_mixer_w8a8(block_type, mixer, layer_idx, act_scales,use_had_transform, device):
    W8A8Mixers = {
        "Mamba2": W8A8QMamba2,
    }
    if block_type not in W8A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W8A8 Mixer")
    if W8A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W8A8")
    mixer = W8A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                use_had_transform=use_had_transform,  )
    return mixer.to(device)


def quantize_mixer_w4a8(block_type, mixer, layer_idx, act_scales,use_had_transform, device):
    W4A8Mixers = {
        "Mamba2": W4A8QMamba2,
    }
    if block_type not in W4A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W4A8 Mixer")
    if W4A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W4A8")
    mixer = W4A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                use_had_transform=use_had_transform,)
    return mixer.to(device)

def get_quantize_block_fn(act_scales, w_bits, a_bits, device, use_had_transform):
    if w_bits == 4 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w4a8, act_scales=act_scales, device=device, use_had_transform=use_had_transform)
    elif w_bits == 8 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w8a8, act_scales=act_scales, device=device, use_had_transform=use_had_transform)
    else:
        raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
    return quantize_norm_fn, quantize_mixer_fn
    
@torch.no_grad()
def quantize_fp16_model(model, model_type, act_scales, device, w_bits=4, a_bits=8, use_had_transform=True):
    assert w_bits in [4, 8], "Only support 4 or 8 bits weights for now"
    assert a_bits in [8], "Only support 8 or 16 bits activations for now"
    quantize_norm_fn, quantize_mixer_fn = get_quantize_block_fn(act_scales, w_bits, a_bits, device, use_had_transform)
    model.config.use_cache = False
    if model_type == "mamba2":
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                layers[i].fused_add_norm = True
  
                layers[i].norm = quantize_norm_fn(
                        block_type="Mamba2",
                        norm=layers[i].norm,
                        layer_idx=i)
                layers[i].mixer = quantize_mixer_fn(
                    block_type="Mamba2", 
                    mixer=layers[i].mixer,
                    layer_idx=i,)
                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
    
    gc.collect()
    torch.cuda.empty_cache()
    return model


def get_model_size(model, model_name, w_bits, a_bits):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
    logging.info(f'W{w_bits}A{a_bits} {model_name} size: {model_mb:.3f} MB')


def quantize_model_mamba(model, model_type, tokenizer, device, args, calibration_dataset=None, calib_preprocess_fn=None,):
    model = configure_model(model, model_type, use_had_transform=args.apply_hadamard,)  
    logging.info(f"Target bit-width W{args.w_bits}A{args.a_bits}")
    if args.a_bits == 8:
            # collect 8-bit activation scales
        act_scales, fp_means = run_calibration(model, model_type, tokenizer,
                                            num_samples=args.calib_data_num,
                                            seq_len=args.calib_seqlen,
                                            calibration_dataset=calibration_dataset,
                                            preprocess_fn=calib_preprocess_fn)
    else:
        raise ValueError(f"Unsupported activation bit-width: {args.a_bits}, try --a_bits 8 or --a_bits 16?")
    

    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gpt
    model = fuse_had_matrices(model) 
    if args.apply_gptq:
        model = apply_gptq(model, tokenizer, device, args , w_bits=args.w_bits)

    model = quantize_fp16_model(
        model, model_type, act_scales, device,
        w_bits=args.w_bits, a_bits=args.a_bits,
        use_had_transform=args.apply_hadamard
    )
    
    if args.compensation :
        model._fp_means = fp_means
        model = mamba_sequential_compensation_ssdout(
            model=model,
            device=device,
            nsamples=args.comp_sam_num,
            tokenizer=tokenizer,
            seq_len=args.calib_seqlen,
            dataloader=None,                     
            dataset=calibration_dataset,          
            preprocess_fn=calib_preprocess_fn,     
            comp_out_decay=args.comp_out_decay,
            layers=args.comp_layers
        )
        model = model.to(device).eval()
        torch.cuda.empty_cache()
        
        
        

    model_name = args.model.lower().split('/')[-1]
    model_name = model_name.replace("mamba", "ssdi8")
    model_name = model_name + f"-w{args.w_bits}aX"
        # store tokenizer for mamba2-8b
    if "mamba2-8b" in args.model:
        # model.save_pretrained should already create the saved dir
        saved_dir = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
        tokenizer.save(saved_dir)
        logging.info(f"Tokenizer is stored at {saved_dir}")
    # quantized model
    get_model_size(model, args.model, args.w_bits, args.a_bits)
    return model.to(device)





@torch.no_grad()
def mamba_sequential_compensation_ssdout(
    model,
    device,
    nsamples,
    tokenizer,
    seq_len=2048,
    dataloader=None,
    dataset=None,
    preprocess_fn=None,
    comp_out_decay=0.1,
    layers=None,           
): 

    def _batch_sum_and_count(y: torch.Tensor):
        if y.dim() == 3: return y.sum(dim=(0,1)).float(), y.shape[0]*y.shape[1]
        if y.dim() == 2: return y.sum(dim=0).float(), y.shape[0]
        raise RuntimeError(f"Unexpected shape {tuple(y.shape)}")

    model.eval()
    blocks = model.backbone.layers
    fp_means = getattr(model, "_fp_means", None)
    assert isinstance(fp_means, dict)

    model.backbone.embedding = model.backbone.embedding.to(device)
    blocks[0] = blocks[0].to(device)

    if dataloader is None and (dataset is None or preprocess_fn is None):
        from datasets import load_dataset
        dataset = load_dataset("monology/pile-uncopyrighted",
                               data_files="val.jsonl.zst", split="train")
        def _preprocess(data, tokenizer, max_tokens, device):
            return tokenizer(data["text"], return_tensors="pt",
                             truncation=True, max_length=max_tokens
                            ).input_ids.to(device)
        from functools import partial
        preprocess_fn = partial(_preprocess, tokenizer=tokenizer,
                                max_tokens=seq_len, device=device)

    inps_list = []
    class Catcher(nn.Module):
        def __init__(self, mod): super().__init__(); self.mod = mod
        def forward(self, x, *args, **kwargs):
            inps_list.append(x.squeeze(0).detach() if (x.dim()==3 and x.size(0)==1) else x.detach())
            raise ValueError
    blocks[0] = Catcher(blocks[0])

    if dataloader is not None:
        it = iter(dataloader)
        for _ in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
            batch = next(it)
            try:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(xb.to(device))
            except ValueError:
                pass
    else:
        for i in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
            x = preprocess_fn(dataset[i])
            try: model(x)
            except ValueError: pass

    blocks[0] = blocks[0].mod
    blocks[0] = blocks[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()
    assert len(inps_list) == nsamples

    res_in_list = [None] * nsamples

    L = len(model.backbone.layers)
    def _norm_idxs(idxs):
        s=set()
        for k in idxs:
            if k<0: k=L+k
            if 0<=k<L: s.add(int(k))
        return s

    if layers is None:
        target_layers = set(range(L))
    else:
        target_layers = _norm_idxs(list(layers) if not isinstance(layers, (list, tuple)) else layers)

    for li in tqdm(range(L), desc="[Layer] Sequential compensation", leave=True):
        layer = blocks[li].to(device)
        mix = layer.mixer
        mix.compensation = True

        if li not in target_layers:
            next_inps, next_res = [], []
            for j in range(nsamples):
                xj = inps_list[j];  xj = xj.unsqueeze(0) if xj.dim()==2 else xj
                yj, resj = layer(xj, residual=res_in_list[j])
                next_inps.append(yj.squeeze(0).detach())
                next_res.append(resj.detach() if torch.is_tensor(resj) else resj)
            blocks[li] = layer.cpu(); torch.cuda.empty_cache()
            inps_list, res_in_list = next_inps, next_res
            continue


        if getattr(mix, "comp_out", None) is None:
            mix.register_buffer("comp_out", torch.zeros(
                mix.out_proj.out_features, dtype=torch.float16, device=device))

        m_fp_out  = fp_means.get(f"L{li}/out_proj_out", None)
        if m_fp_out is not None: m_fp_out = m_fp_out.to(device=device, dtype=torch.float32)
        mix.comp_out.zero_()

        out_sum=None; out_cnt=0
        def out_hook(_m,_i,out):
            nonlocal out_sum,out_cnt
            s, n = _batch_sum_and_count(out)
            out_sum = s if out_sum is None else (out_sum+s)
            out_cnt += int(n)
        hC = mix.out_proj.register_forward_hook(out_hook)
        for j in tqdm(range(nsamples), desc=f"[L{li}] Phase C: out_proj", leave=False):
            xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
            _ = layer(xj, residual=res_in_list[j], comp_calib=True)
        hC.remove()
        if (m_fp_out is not None) and (out_cnt > 0):
            q_mean_out = torch.where(torch.isfinite(out_sum/out_cnt), out_sum/out_cnt, torch.zeros_like(out_sum))
            mix.comp_out.copy_((m_fp_out - q_mean_out).to(mix.comp_out.dtype) * comp_out_decay)

        next_inps, next_res = [], []
        for j in tqdm(range(nsamples), desc=f"[L{li}] Phase D: propagate", leave=False):
            xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
            yj, resj = layer(xj, residual=res_in_list[j], comp_calib=True)
            next_inps.append(yj.squeeze(0).detach())
            next_res.append(resj.detach() if torch.is_tensor(resj) else resj)

        blocks[li] = layer.cpu(); torch.cuda.empty_cache()
        inps_list, res_in_list = next_inps, next_res

    model._fp_means = None
    return model




