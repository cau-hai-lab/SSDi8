"""
This file is a modified version of the original file from the Quamba repo.
https://github.com/enyac-group/Quamba
"""
import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _quant_state_passing_fwd_kernel(
    # Pointers (모두 int8 버퍼로 가정: states/out/final/init)
    states_ptr, out_ptr, final_states_ptr,
    dA_cs_ptr, initstates_ptr, seq_idx_ptr,     # dA_cs_ptr: FP16/FP32 게이트 원천 (아래서 즉시 양자화)

    # Sizes
    dim, nchunks, seqlen, chunk_size,

    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,

    # Meta
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GATE_SHIFT: tl.constexpr,   # e.g. 7  (S = 1<<7)
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    # base ptrs
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr  += pid_b * stride_dA_cs_batch  + pid_h * stride_dA_cs_head
    out_ptr    += pid_b * stride_out_batch    + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs    = out_ptr    + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    # s32 누적 상태 초기화
    if not HAS_INITSTATES:
        s32 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        s8_init = tl.load(initstates_ptrs, mask=offs_m < dim, other=0).to(tl.int8)
        s32 = tl.cast(s8_init, tl.int32)

    # 첫 out 저장 (int8)
    s8_store = tl.minimum(tl.maximum(s32, -127), 127).to(tl.int8)
    tl.store(out_ptrs, s8_store, mask=offs_m < dim)
    out_ptrs += stride_out_chunk

    seq_idx = tl.zeros((), dtype=tl.int32)
    for c in range(nchunks):
        # 새로운 chunk state (int8)
        new_s8 = tl.load(states_ptrs, mask=offs_m < dim, other=0).to(tl.int8)
        new_s32 = tl.cast(new_s8, tl.int32)

        # 게이트 = exp(dA_last)  (스칼라 per (b,h,c))
        dA_cs   = tl.load(dA_cs_ptr).to(tl.float32)   # FP
        gate_fp = tl.exp(dA_cs)

        if HAS_SEQ_IDX:
            # 마지막 토큰의 seq id 비교
            idx = (tl.minimum((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen
            seq_idx_new = tl.load(seq_idx_ptr + idx)
            gate_fp = tl.where(seq_idx_new == seq_idx, gate_fp, 0.0)
            seq_idx = seq_idx_new

        # ---- 게이트 int8 양자화 (Q0.SHIFT)
        S = 1 << GATE_SHIFT
        gate_scaled = gate_fp * S
        gate_r = tl.where(gate_scaled >= 0, tl.floor(gate_scaled + 0.5), tl.ceil(gate_scaled - 0.5))
        gate_r = tl.minimum(tl.maximum(gate_r, -127.0), 127.0)
        gate_q = tl.cast(gate_r, tl.int32)  # int32로 연산

        # ---- 정수 경로: (s32 * gate_q) >> SHIFT  + new_s32
        mul32 = s32 * gate_q
        # round-to-nearest for right-shift
        mul32 = (mul32 + (S >> 1)) >> GATE_SHIFT
        s32 = mul32 + new_s32

        # 중간 out 저장(마지막 chunk만 final_states에 저장)
        if c < (nchunks - 1):
            s8_mid = tl.minimum(tl.maximum(s32, -127), 127).to(tl.int8)
            tl.store(out_ptrs, s8_mid, mask=offs_m < dim)
        else:
            s8_fin = tl.minimum(tl.maximum(s32, -127), 127).to(tl.int8)
            tl.store(final_states_ptrs, s8_fin, mask=offs_m < dim)

        # advance
        states_ptrs += stride_states_chunk
        dA_cs_ptr   += stride_dA_cs_chunk
        out_ptrs    += stride_out_chunk

def _quant_state_passing_fwd(states, dA_chunk_cumsum, initial_states=None, seq_idx=None, chunk_size=None,
                             out_dtype=None):
    # shapes 동일
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)

    # ★ int8 입출력 버퍼
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=torch.int8)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.int8)

    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)

    with torch.cuda.device(states.device.index):
        _quant_state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, initial_states, seq_idx,
            dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
              if initial_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            GATE_SHIFT=7,    # ← 게이트 고정소수점 비트(실험값). 6~8 사이에서 골라봐도 됨.
        )
    return out, final_states