"""
This file is a modified version of the original file from the Quamba repo.
https://github.com/enyac-group/Quamba
"""

from packaging import version
import torch
import triton
import triton.language as tl

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _quant_chunk_scan_fwd_kernel(
    # Pointers
    cb_ptr, x_ptr, x_cols_scale_ptr, z_ptr, z_scale, out_ptr, out_x_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, C_ptr, C_scale, prev_states_ptr, states_scale, D_ptr, D_scale,
    CB_scale_ptr,
    # Dims
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head, stride_ss_head, stride_ss_hdim,
    stride_xs_head, stride_xs_hdim,
    # Meta
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    USE_FLOA32_MM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    gid   = pid_h // nheads_ngroups_ratio
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize,
                      mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    seq_idx_prev = tl.zeros((), dtype=tl.int32)
    seq_idx_m = tl.zeros((BLOCK_SIZE_M,), dtype=tl.int32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m    = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
                               mask=offs_m < chunk_size_limit, other=-1)

    # per-G scalar & per-(H,P) column scales
    gscale  = tl.load(C_scale + gid).to(tl.float32)  # (G,)
    ss_cols = tl.load(states_scale + pid_h * stride_ss_head + offs_n * stride_ss_hdim,
                      mask=(offs_n < hdim), other=1.0).to(tl.float32)  # (BN,)
    Sx_cols = tl.load(x_cols_scale_ptr + pid_h * stride_xs_head + offs_n * stride_xs_hdim,
                      mask=(offs_n < hdim), other=1.0).to(tl.float32)  # (BN,)

    # row gate
    row_gate = tl.exp(dA_cs_m) if not HAS_SEQ_IDX else tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)

    # ===== states path: INT8xINT8 -> INT32 =====
    acc_i32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    k = 0
    while k < dstate:
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        C_ptrs  = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k[None, :] * stride_C_dstate)
        S_i8ptr = prev_states_ptr + (offs_k[:, None] * stride_states_dstate + offs_n[None, :] * stride_states_hdim)

        Ci8 = tl.load(C_ptrs,
                      mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate),
                      other=0).to(tl.int8)
        Si8 = tl.load(S_i8ptr,
                      mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim),
                      other=0).to(tl.int8)

        acc_i32 += tl.dot(Ci8, Si8, out_dtype=tl.int32)
        k += BLOCK_SIZE_K

    acc_prev = tl.cast(acc_i32, tl.float32)
    acc_prev *= gscale
    acc_prev *= ss_cols[None, :]
    acc = acc_prev * row_gate[:, None]

    # ===== conv path: cb(int8->fp) @ X(fp on-the-fly q&dq) =====
    # ===== conv path: cb(int8->fp) @ X(fp 그대로) =====
    mm_dtype = tl.float32 if USE_FLOA32_MM else tl.float16
    Scb = tl.load(CB_scale_ptr + (pid_h // nheads_ngroups_ratio)).to(tl.float32)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs  = x_ptr  + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)

    for kk in range(0, K_MAX, BLOCK_SIZE_K):
        # CB: int8 -> fp, exp gate & dt
        cb_q = tl.load(cb_ptrs,
                    mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - kk),
                    other=0).to(tl.int8)
        cb = tl.cast(cb_q, tl.float32) * Scb
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - kk, other=0.0).to(tl.float32)
        cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - kk, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= kk + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)

        # X: FP 그대로 로드 (FP16/BF16/FP32 → 계산은 float32/float16로)
        Xfp = tl.load(x_ptrs,
                    mask=(offs_k[:, None] < chunk_size_limit - kk) & (offs_n[None, :] < hdim),
                    other=0.0).to(tl.float32)

        acc += tl.dot(cb.to(mm_dtype), Xfp.to(mm_dtype))

        cb_ptrs        += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs         += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs        += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # ===== residual D (X도 FP 그대로) =====
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_scale) * tl.load(D_ptr + pid_h * stride_D_head + offs_out_n,
                                        mask=offs_out_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_scale) * tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)

        x_res = tl.load(x_ptr + (offs_out_m[:, None] * stride_x_seqlen + offs_out_n[None, :] * stride_x_hdim),
                        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
                        other=0.0).to(tl.float32)
        acc += x_res * D

    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_scale) * tl.load(z_ptrs,
                                       mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
                                       other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


def _quant_chunk_scan_fwd(
    cb, q_x, x_scale,                # x_fp: FP 텐서 (기존 q_x 자리에 FP를 넣음)
    dt, dA_cumsum, q_C, C_scale, states,
    q_D=None, D_scale=None, q_z=None, z_scale=None,
    seq_idx=None, mm_dtype=torch.float16,
    C_chunkscan_scale=None, cb_chunkscan_scale=None, state_chunkscan_scale=None, ori_x_row_scale_chp=None,
):
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_C.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert q_x.is_cuda and q_x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert nheads % ngroups == 0
    assert mm_dtype in (torch.float16, torch.float32)
    assert q_C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    device = q_x.device

    # scales
    states_scale_tensor = state_chunkscan_scale.to(device=states.device, dtype=torch.float32)
    C_scale_tensor      = C_chunkscan_scale.to(device=q_C.device, dtype=torch.float32).view(ngroups)
    CB_scale_tensor     = cb_chunkscan_scale.to(device=cb.device, dtype=torch.float32).view(ngroups)
    Sx_cols_tensor      = ori_x_row_scale_chp.to(device=device, dtype=torch.float32)  # (H,P)

    # 출력
    out = torch.empty(batch, seqlen, nheads, headdim, device=device, dtype=torch.float16)
    out_x = torch.empty_like(out) if q_z is not None else None

    grid = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
        batch * nchunks,
        nheads
    )

    _quant_chunk_scan_fwd_kernel[grid](
        # ---- Pointers ---------------------------------------------------------
        cb,                   # cb_ptr
        q_x,                  # x_ptr (FP16 원본)
        Sx_cols_tensor,       # x_cols_scale_ptr (FP 경로여도 시그니처 맞춤, 사용 안 해도 됨)
        q_z if q_z is not None else q_x,  # z_ptr (없으면 더미로 q_x 전달)
        z_scale if z_scale is not None else q_x.new_tensor(1.0, dtype=torch.float32),
        out,                  # out_ptr
        out_x if out_x is not None else out,  # out_x_ptr (없으면 out 전달)
        dt,                   # dt_ptr
        dA_cumsum,            # dA_cumsum_ptr
        seq_idx if seq_idx is not None else q_x,  # seq_idx_ptr (없으면 더미)
        q_C,                  # C_ptr
        C_scale_tensor,       # C_scale
        states,               # prev_states_ptr
        states_scale_tensor,  # states_scale
        q_D if q_D is not None else q_C,        # D_ptr (없으면 더미)
        D_scale if D_scale is not None else q_x.new_tensor(1.0, dtype=torch.float32),  # D_scale
        CB_scale_tensor,      # CB_scale_ptr

        # ---- Dims ------------------------------------------------------------
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,

        # ---- Strides (순서 매우 중요!) ---------------------------------------
        # cb
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),

        # x (q_x: FP16)
        q_x.stride(0), q_x.stride(1), q_x.stride(2), q_x.stride(3),

        # z (없으면 0,0,0,0)
        *( (q_z.stride(0), q_z.stride(1), q_z.stride(2), q_z.stride(3))
        if q_z is not None else (0, 0, 0, 0) ),

        # out
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),

        # dt  (커널 시그니처: batch, chunk, head, csize)
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),

        # dA_cumsum (동일한 순서)
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),

        # seq_idx (없으면 0,0)
        *( (seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0) ),

        # C
        q_C.stride(0), q_C.stride(1), q_C.stride(2), q_C.stride(3),

        # states (***** 여기 5개 모두 필수 *****)
        states.stride(0),  # stride_states_batch
        states.stride(1),  # stride_states_chunk
        states.stride(2),  # stride_states_head
        states.stride(3),  # stride_states_hdim
        states.stride(4),  # stride_states_dstate  ← 에러에 나온 첫 누락 항목

        # D (shape: (H,P) 또는 (H,))
        (q_D.stride(0) if q_D is not None else 0),  # stride_D_head

        # states_scale (shape: (H,P))
        states_scale_tensor.stride(0),  # stride_ss_head
        states_scale_tensor.stride(1),  # stride_ss_hdim

        # x_cols_scale (shape: (H,P)) - FP경로에서도 자리만 맞추면 됨
        Sx_cols_tensor.stride(0),  # stride_xs_head
        Sx_cols_tensor.stride(1),  # stride_xs_hdim

        # ---- Meta ------------------------------------------------------------
        IS_CAUSAL=True,
        HAS_D=(q_D is not None),
        D_HAS_HDIM=(q_D is not None and q_D.dim() == 2),
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=(q_z is not None),
        HAS_SEQ_IDX=(seq_idx is not None),
        USE_FLOA32_MM=(mm_dtype is torch.float32),
        IS_TRITON_22=TRITON_22,
    )
    return out, out_x


