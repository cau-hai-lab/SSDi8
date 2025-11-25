"""
This file is a modified version of the original file from the Quamba repo.
https://github.com/enyac-group/Quamba
"""

import torch
from einops import rearrange, repeat

from ssdi8.triton.quant_chunk_cumsum import _quant_chunk_cumsum_fwd
from ssdi8.triton.quant_state_passing import _quant_state_passing_fwd
from ssdi8.triton.quant_chunk_state import _quant_chunk_state_fwd
from ssdi8.triton.quant_chunk_scan import _quant_chunk_scan_fwd
from ssdi8.triton.quant_bmm_chunk import _quant_bmm_chunk_fwd

def _quant_mamba_chunk_scan_combined_fwd(
        q_x, x_scale, q_dt, dt_scale, q_A_log, A_log_scale,
        q_B, B_scale, q_C, C_scale, ssm_state_scale, chunk_size,
        q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, initial_states=None, seq_idx=None,
        cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), mm_dtype=torch.float16,    x_row_scale_chp=None,  ori_x_row_scale_chp=None ,
        C_chunkscan_scale=None,cb_chunkscan_scale=None,state_chunkscan_scale=None,
        B_row_scale_cgn=None, comp_calib=False
    ):
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape

    assert x_scale.is_cuda
    assert x_scale.numel() == 1
    assert B_scale.is_cuda
    assert B_scale.numel() == 1
    assert C_scale.is_cuda
    assert C_scale.numel() == 1

    assert nheads % ngroups == 0
    assert q_x.is_cuda
    #assert q_x.dtype == torch.int8
    assert q_x.shape == (batch, seqlen, nheads, headdim)
    assert q_B.is_cuda
    #assert q_B.dtype == torch.int8
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert q_dt.is_cuda
    assert q_dt.shape == (batch, seqlen, nheads)
    assert q_A_log.is_cuda
    assert q_A_log.dtype == torch.int8
    assert q_A_log.shape == (nheads,)
    assert q_C.is_cuda
    #assert q_C.dtype == torch.int8
    assert q_C.shape == q_B.shape
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if q_B.stride(-1) != 1:
        q_B = q_B.contiguous()
    if q_C.stride(-1) != 1:
        q_C = q_C.contiguous()
    if q_x.stride(-1) != 1 and q_x.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_x = q_x.contiguous()
    if q_z is not None and q_z.stride(-1) != 1 and q_z.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_z = q_z.contiguous()
    if q_D is not None and q_D.stride(-1) != 1:
        q_D = q_D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    dev = q_x.device if q_x.is_cuda else None
    if not comp_calib :
        q_B, B_scale_g = quantize_int8_per_group_triton(q_B, B_row_scale_cgn, group_dim=2)
    else :
        q_B, B_scale_g = quantize_int8_per_group_torch(q_B, B_row_scale_cgn, group_dim=2)
        
    if not comp_calib :
        q_C, C_scale_g = quantize_int8_per_group_triton(q_C, C_chunkscan_scale, group_dim=2)
    else :
        q_C, C_scale_g = quantize_int8_per_group_torch(q_C, C_chunkscan_scale, group_dim=2)



    dA_cumsum, dt = _quant_chunk_cumsum_fwd(q_dt, dt_scale, q_A_log, A_log_scale, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _quant_chunk_state_fwd(
            q_B, q_x,
            dt, dA_cumsum,
            mm_dtype=torch.float16, seq_idx=seq_idx, states_in_fp32=True,
            x_row_scale_chp=x_row_scale_chp, B_scale_g=B_scale_g,
            state_chunkscan_scale=state_chunkscan_scale,
        )
                

    states, final_states = _quant_state_passing_fwd(
                                    rearrange(states, "... p n -> ... (p n)"),
                                    dA_cumsum[:, :, :, -1],
                                    initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                        if initial_states is not None else None,
                                    seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                                )
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]

    CB = _quant_bmm_chunk_fwd(q_C,q_B,C_scale_g,B_scale_g, cb_chunkscan_scale,chunk_size, seq_idx=seq_idx, )

    out, out_x = _quant_chunk_scan_fwd(
            CB,
            q_x, x_scale,
            dt, dA_cumsum,
            q_C, C_scale,
            states,
            q_D=q_D, 
            D_scale=D_scale,
            q_z=q_z, 
            z_scale=z_scale,
            seq_idx=seq_idx,
            mm_dtype=torch.float16,
            C_chunkscan_scale=C_scale_g,
            cb_chunkscan_scale=cb_chunkscan_scale,
            state_chunkscan_scale=state_chunkscan_scale,
            ori_x_row_scale_chp=ori_x_row_scale_chp,
        )
    if cu_seqlens is None:
        return out, final_states
    else:
        raise NotImplementedError("Only supports `cu_seqlens=None`")
    
    
    
    
@torch.no_grad()
def quantize_B_int8_per_gn(B_fp: torch.Tensor, B_row_scale_cgn: torch.Tensor) -> torch.Tensor:
    """
    B_fp: (B, S, G, N)   FP 텐서
    B_row_scale_cgn: (G, N)  per-(G,N) 스케일 (커널에서 쓰던 것과 동일)
    return: q_B_int8: (B, S, G, N)  int8
    """
    assert B_fp.dim() == 4 and B_row_scale_cgn.dim() == 2
    B, S, G, N = B_fp.shape
    assert B_row_scale_cgn.shape == (G, N)
    assert B_fp.is_cuda and B_row_scale_cgn.is_cuda

    # (1,1,G,N)로 브로드캐스트 후 per-(G,N) 스케일로 나눔
    inv_scale = (1.0 / B_row_scale_cgn).to(dtype=torch.float32, device=B_fp.device).view(1, 1, G, N)
    x = B_fp.to(torch.float32) * inv_scale

    # round half away from zero
    q = torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
    q.clamp_(-127, 127)
    return q.to(torch.int8)


def quantize_int8_per_group(t: torch.Tensor, scales_g: torch.Tensor, group_dim: int):
    """
    per-G 양자화 전용:
      q = round(t / S_g), clamp to int8
    t: (..., G, ...)
    scales_g: (G,)  혹은 실수로 (G,N) 들어오면 max 축약해서 (G,)로 사용
    return: (q_int8, S_g_float32)  # S_g는 (G,)
    """
    assert t.is_cuda and scales_g.is_cuda
    # (G,N)로 들어오면 per-G로 축약
    if scales_g.dim() == 2:
        scales_g = scales_g.max(dim=-1).values.contiguous()
    assert scales_g.dim() == 1, "이 버전은 per-G 스케일만 지원합니다."
    G = t.shape[group_dim]
    assert scales_g.shape[0] == G, f"스케일 길이({scales_g.shape[0]}) != G({G})"

    view = [1] * t.dim()
    view[group_dim] = G
    S = scales_g.to(dtype=t.dtype, device=t.device).view(*view)  # (..., G, ...)
    q = torch.clamp(torch.round(t / S), -127, 127).to(torch.int8)
    return q, scales_g.to(torch.float32).contiguous()






import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64 }, num_stages=5, num_warps=2),
    ],
    key=['B', 'S', 'G', 'N'],
)
@triton.jit
def _quantize_int8_per_g_kernel(
    t_ptr, q_ptr, scales_g_ptr,
    # sizes
    B: tl.constexpr, S: tl.constexpr, G: tl.constexpr, N: tl.constexpr,
    # strides for t: (B,S,G,N)
    st_b, st_s, st_g, st_n,
    # strides for q: (B,S,G,N)
    sq_b, sq_s, sq_g, sq_n,
    # meta
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Tiles over M=(B*S) rows and N cols for a fixed group g.
    Grid:
      axis0: tiles over (M,N)
      axis1: group g in [0..G)
    """
    # program ids
    pid_g = tl.program_id(axis=1)  # group id
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) %  num_pid_n

    # bounds / offsets
    M = B * S
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask   = mask_m[:, None] & mask_n[None, :]

    # decompose m -> (b, s)
    b_idx = offs_m // S
    s_idx = offs_m - b_idx * S  # offs_m % S

    # base ptrs (broadcast into tile)
    t_ptrs = (t_ptr
              + b_idx[:, None] * st_b
              + s_idx[:, None] * st_s
              + pid_g          * st_g
              + offs_n[None, :] * st_n)
    q_ptrs = (q_ptr
              + b_idx[:, None] * sq_b
              + s_idx[:, None] * sq_s
              + pid_g          * sq_g
              + offs_n[None, :] * sq_n)

    # load per-G scale (float32), guard against zeros
    s_g = tl.load(scales_g_ptr + pid_g).to(tl.float32)
    eps = 1e-8
    s_g = tl.maximum(s_g, eps)

    # hints
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)

    # load, scale, quantize (half-away-from-zero), clamp, store
    t = tl.load(t_ptrs, mask=mask, other=0.0).to(tl.float32)  # robust quant: do math in fp32
    x = t / s_g
    qf = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
    qf = tl.minimum(tl.maximum(qf, -127.0), 127.0)
    q8 = tl.cast(qf, tl.int8)
    tl.store(q_ptrs, q8, mask=mask)
    
    
@torch.no_grad()
def quantize_int8_per_group_triton(t: torch.Tensor, scales_g: torch.Tensor, group_dim: int):
    """
    Triton-optimized per-G quantization for (B,S,G,N) with group_dim==2.
    q = round(t / S_g), clamp to int8.
    Returns: (q_int8, S_g_float32)
    """
    assert t.is_cuda and scales_g.is_cuda, "t, scales_g must be CUDA tensors"
    assert t.dim() == 4 and group_dim == 2, "This kernel supports shape (B,S,G,N) with group_dim=2"
    B, S, G, N = t.shape

    # Prepare per-G scale (G,)
    if scales_g.dim() == 2:
        # (G,N) -> (G,) by max over N   (your original policy)
        scales_g = scales_g.max(dim=-1).values.contiguous()
    assert scales_g.dim() == 1 and scales_g.shape[0] == G, f"scales_g must be (G,), got {tuple(scales_g.shape)}"

    # Output allocation
    q = torch.empty_like(t, dtype=torch.int8)

    # Launch
    grid = lambda META: (
        triton.cdiv(B * S, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        G
    )

    # Make sure we pass strides in element units (PyTorch gives strides in elements already)
    _quantize_int8_per_g_kernel[grid](
        t, q, scales_g.to(dtype=torch.float32),
        B, S, G, N,
        t.stride(0), t.stride(1), t.stride(2), t.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    )

    return q, scales_g.to(torch.float32).contiguous()




##### torch quantization for compensation calibration #####
import torch

@torch.no_grad()
def _round_half_away_from_zero(x: torch.Tensor):
    # Triton 커널과 동일한 half-away-from-zero
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

@torch.no_grad()
def quantize_int8_per_hp_torch(x: torch.Tensor, s_hp: torch.Tensor):
    """
    x: (B,S,H,P) CUDA
    s_hp: (H,P) CUDA  (float32 권장)
    return: q_x(int8, B,S,H,P), s_hp(float32, contiguous)
    """
    assert x.is_cuda and x.dim() == 4
    H, P = x.shape[2], x.shape[3]
    assert s_hp.is_cuda and tuple(s_hp.shape) == (H, P)

    s_hp = s_hp.to(torch.float32).clamp_min_(1e-8).contiguous()
    scale = s_hp.view(1, 1, H, P)                      # (1,1,H,P)
    qf = _round_half_away_from_zero(x.to(torch.float32) / scale)
    q  = torch.clamp(qf, -127, 127).to(torch.int8)
    return q, s_hp


@torch.no_grad()
def quantize_int8_per_group_torch(
    t: torch.Tensor,
    scales_g: torch.Tensor,
    group_dim: int = 2,
    eps: float = 1e-8,
):
    """
    Pure-PyTorch per-group int8 quantization for tensors shaped (B, S, G, N) with group_dim == 2.

    Args:
        t:          float tensor (B, S, G, N). CUDA or CPU 모두 가능.
        scales_g:   (G,) 또는 (G, N). (G, N)인 경우 dim=-1 max로 (G,)로 축약.
        group_dim:  반드시 2 (G 차원).
        eps:        0 스케일 가드.

    Returns:
        q_int8:     torch.int8 tensor, same shape/device as t.
        s_g:        torch.float32 tensor of shape (G,), contiguous.
    """
    assert t.dim() == 4 and group_dim == 2, "This function supports shape (B,S,G,N) with group_dim=2"
    B, S, G, N = t.shape

    # 스케일 형태 정규화
    if scales_g.dim() == 2:
        # (G, N) -> (G,) by max over N  (원본 정책과 동일)
        scales_g = scales_g.max(dim=-1).values
    assert scales_g.dim() == 1 and scales_g.numel() == G, f"scales_g must be (G,), got {tuple(scales_g.shape)}"

    # dtype/디바이스 정렬
    device = t.device
    s_g = scales_g.to(device=device, dtype=torch.float32).contiguous()
    s_g = torch.clamp(s_g, min=eps)  # 0 가드

    # 브로드캐스팅 모양: (1,1,G,1)
    s_g_bc = s_g.view(1, 1, G, 1)

    # 연산은 안정성을 위해 fp32로
    x = (t.to(torch.float32) / s_g_bc)

    # half-away-from-zero 반올림: x>=0 ? floor(x+0.5) : ceil(x-0.5)
    qf_pos = torch.floor(x + 0.5)
    qf_neg = torch.ceil(x - 0.5)
    qf = torch.where(x >= 0, qf_pos, qf_neg)

    # int8 클램프 [-127, 127]
    qf = torch.clamp(qf, -127.0, 127.0)

    # int8 캐스트
    q = qf.to(torch.int8)

    return q, s_g