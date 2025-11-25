"""
The code is modfied from
https://github.com/state-spaces/mamba
"""
import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ssdi8.quant_utils import quantize_tensor_per_tensor_absmax
from ssdi8.triton.quant_ssd_combined import _quant_mamba_chunk_scan_combined_fwd


class Quamba2ChunkScan(nn.Module):

    def __init__(self, d_ssm, headdim, d_state, ngroups, D_has_hdim, chunk_size,
                 nhead_groups=0, ndim_groups=0, delta_softplus=True,
                 quant_dt=True, quant_B=True, quant_C=True, quant_z=False,
                 dt_limit=(0.0, float("inf")), device=None, **kwargs):
        factory_kwargs = {"device": "cuda"} # we use partial function, so it is a hardcode for putting everything on cuda
        # factory_kwargs = {"device": device}
        super().__init__()

        self.d_ssm = d_ssm
        self.headdim = headdim
        self.d_state = d_state
        self.ngroups = ngroups
        self.D_has_hdim = D_has_hdim
        self.chunk_size = chunk_size
        self.nhead_groups = nhead_groups
        self.ndim_groups = ndim_groups
        self.delta_softplus = delta_softplus
        self.dt_limit = dt_limit

        nheads = d_ssm // headdim
        # create space for dt bias
        self.register_buffer('dt_bias', torch.empty(nheads, dtype=torch.float16, **factory_kwargs)) # we do not quant dt_bias
        # create space for A
        self.register_buffer('A_log', torch.empty(nheads, dtype=torch.int8, **factory_kwargs))
        self.register_buffer('A_log_scale', torch.empty([], dtype=torch.float32, **factory_kwargs)) # no-shape
        # create space for D
        self.register_buffer('D', torch.empty(
            self.d_ssm if self.D_has_hdim else nheads,
            dtype=torch.int8, **factory_kwargs))
        self.register_buffer('D_scale', torch.empty([], dtype=torch.float32, **factory_kwargs)) # no-shape
        self.register_buffer("x_chunk_state_scale_chp", None)
        self.register_buffer("ori_x_chunk_state_scale_chp", None)
        self.register_buffer("B_chunk_state_scale_cgn", None)
        self.register_buffer("C_chunkscan_scale", None)
        self.register_buffer("cb_chunkscan_scale", None)
        self.register_buffer("state_chunkscan_scale", None)
        # Create space for scales
        quant_params = {'dt': quant_dt, 'z': quant_z}
        for name, quant_flag in quant_params.items():
            if quant_flag:
                self.register_buffer(f'{name}_scale', torch.empty([], dtype=torch.float32, **factory_kwargs))
            else:
                setattr(self, f'{name}_scale', None)

        if nhead_groups > 0 and ndim_groups > 0:
            # create space for SSM state scale
            self.register_buffer('ssm_state_scale', torch.empty(
                [self.ngroups, self.nhead_groups, self.ndim_groups, self.d_state],
                dtype=torch.float32, **factory_kwargs))
            self.register_buffer('x_scales', torch.empty(
                [ngroups, nhead_groups, ndim_groups], dtype=torch.float32, **factory_kwargs))
            self.register_buffer('x_head_group_range', torch.empty(
                [ngroups, nhead_groups], dtype=torch.int32, **factory_kwargs))
            self.register_buffer('x_dim_group_range', torch.empty(
                [ngroups, nhead_groups, ndim_groups], dtype=torch.int32, **factory_kwargs))
            quant_params = {'B': quant_B, 'C': quant_C}
            for name, quant_flag in quant_params.items():
                if quant_flag:
                    self.register_buffer(f'{name}_scale', torch.empty([ngroups], dtype=torch.float32, **factory_kwargs))
                else:
                    setattr(self, f'{name}_scale', None)
        elif nhead_groups ==0 and ndim_groups == 0:
            # create space for SSM state scale
            self.register_buffer('ssm_state_scale', torch.empty([1], dtype=torch.float32, **factory_kwargs))
            # scales must be on cuda for triton kernels
            self.register_buffer('x_scales', torch.empty([1], dtype=torch.float32, **factory_kwargs))
            self.x_head_group_range = None
            self.x_dim_group_range = None
            quant_params = {'B': quant_B, 'C': quant_C}
            for name, quant_flag in quant_params.items():
                if quant_flag:
                    self.register_buffer(f'{name}_scale', torch.empty([1], dtype=torch.float32, **factory_kwargs))
                else:
                    setattr(self, f'{name}_scale', None)
        else:
            raise ValueError("nhead_groups and ndim_groups must be both 0 or both not 0")

        self.set_chunk_scan_fn()

    @classmethod
    def from_fp16(cls, d_ssm, headdim, d_state, ngroups,
                 x_scales, x_head_group_range, x_dim_group_range,
                 A_log, chunk_size, ssm_state_scale, D=None, D_has_hdim=False, dt_bias=None, delta_softplus=True,
                 dt_scale=None, B_scale=None, C_scale=None, z_scale=None, dt_limit=(0.0, float("inf")),
                 x_row_scale_chp=None,ori_x_row_scale_chp=None, B_row_scale_cgn=None,C_chunkscan_scale=None,cb_chunkscan_scale=None,state_chunkscan_scale=None,):

        nhead_groups = x_head_group_range.shape[1] if x_head_group_range is not None else 0
        ndim_groups = x_dim_group_range.shape[1] if x_dim_group_range is not None else 0
        qchunkscan = cls(d_ssm, headdim, d_state, ngroups, D_has_hdim,
                         chunk_size, nhead_groups, ndim_groups, delta_softplus, dt_limit)

        A_log_quant, A_log_scale = quantize_tensor_per_tensor_absmax(A_log, n_bits=8)
        qchunkscan.A_log = A_log_quant.to(torch.int8)
        qchunkscan.A_log_scale = A_log_scale.float().to(A_log.device)

        if x_head_group_range is not None and x_dim_group_range is not None:
            qchunkscan.ssm_state_scale = cls.get_ssm_state_scales(
                ssm_state_scale, ngroups,
                x_head_group_range.shape[1], x_dim_group_range.shape[2],
                d_state, A_log.device)
        else:
            qchunkscan.ssm_state_scale = ssm_state_scale.float().to(A_log.device)

        if D is not None:
            # rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            D_quant, D_scale = quantize_tensor_per_tensor_absmax(D, n_bits=8)
            qchunkscan.D = D_quant.to(torch.int8)
            qchunkscan.D_scale = D_scale.float().to(D.device)
        else:
            qchunkscan.D = None
        
        if dt_bias is not None:
            qchunkscan.dt_bias = dt_bias.to(torch.float16) # we do not quant dt_bias
        else:
            qchunkscan.dt_bias = None  

            
        if x_row_scale_chp is not None:
            qchunkscan.x_chunk_state_scale_chp = x_row_scale_chp.to(
                qchunkscan.A_log.device, dtype=torch.float32
            ).contiguous()
        if ori_x_row_scale_chp is not None:
            qchunkscan.ori_x_chunk_state_scale_chp = ori_x_row_scale_chp.to(
                qchunkscan.A_log.device, dtype=torch.float32
            ).contiguous()
        if B_row_scale_cgn is not None:
            qchunkscan.B_chunk_state_scale_cgn = B_row_scale_cgn.to(
                qchunkscan.A_log.device, dtype=torch.float32
            ).contiguous()
        if cb_chunkscan_scale is not None:
            qchunkscan.cb_chunkscan_scale = cb_chunkscan_scale.to(
                qchunkscan.A_log.device, dtype=torch.float32
            ).contiguous()
        if C_chunkscan_scale is not None:
            qchunkscan.C_chunkscan_scale = C_chunkscan_scale.to(
                qchunkscan.A_log.device, dtype=torch.float32
            ).contiguous()
        if state_chunkscan_scale is not None:
            qchunkscan.state_chunkscan_scale = state_chunkscan_scale.to(
                qchunkscan.A_log.device, dtype=torch.float32
            ).contiguous()
        # use scale.item() will cause Floating point exception (core dumped), since python use float64
        # to make sure we are using float32, we just pass in the scalar tensors with torch.float32 type
        scales = {'dt_scale': dt_scale, 'B_scale': B_scale, 'C_scale': C_scale, 'z_scale': z_scale}
        for name, scale in scales.items():
            setattr(qchunkscan, name, scale.float() if scale is not None else None)  # Move to CUDA and convert to float

        qchunkscan.x_scales = x_scales.to(A_log.device)
        qchunkscan.x_head_group_range = None if x_head_group_range is None else x_head_group_range.to(A_log.device)
        qchunkscan.x_dim_group_range = None if x_dim_group_range is None else x_dim_group_range.to(A_log.device)
        qchunkscan.set_chunk_scan_fn()

        return qchunkscan

    @staticmethod
    def get_ssm_state_scales(scales, ngroups, n_head_groups, n_dim_groups, dstates, device):
        # out scales shape: [ngroups, n_head_groups, n_dim_groups, dstates]
        if isinstance(scales, list):
            out_scales = torch.empty((ngroups, n_head_groups, n_dim_groups, dstates), device=device)
            for ssd_g in range(ngroups):
                for idx, (h_gsize, ch_gsize, ch_scales) in enumerate(scales[ssd_g]):
                    # h_gsize: int, ch_gsize: List[int], ch_scales: List[float]
                    # ch_scales is a tensor with shape: [4, 128]
                    out_scales[ssd_g, idx, :, :] = ch_scales.to(device)
            return out_scales
        else:
            return scales.to(device)

    def set_chunk_scan_fn(self):
        self.x_scales = torch.tensor(1.0, dtype=torch.float32, device=self.A_log.device)
        self.B_scale  = torch.tensor(1.0, dtype=torch.float32, device=self.A_log.device)
        self.C_scale  = torch.tensor(1.0, dtype=torch.float32, device=self.A_log.device)
        self.dt_scale = torch.tensor(1.0, dtype=torch.float32, device=self.A_log.device)
        self.z_scale  = torch.tensor(1.0, dtype=torch.float32, device=self.A_log.device) if self.z_scale is not None else None
        if self.x_head_group_range is None and self.x_dim_group_range is None:
            # scales must be on cuda for triton kernels
            self.x_scales = self.x_scales.float()
            self.x_head_group_range = None
            self.x_dim_group_range = None

            # get chunk_scan_combined_fwd
            # scales must be on cuda before using partial for triton kernels
            self.chunk_scan_combined_fwd = partial(
                _quant_mamba_chunk_scan_combined_fwd,
                    x_scale=self.x_scales, dt_scale=self.dt_scale,
                    q_A_log=self.A_log, A_log_scale=self.A_log_scale,
                    ssm_state_scale=self.ssm_state_scale,
                    B_scale=self.B_scale, C_scale=self.C_scale,
                    q_D=self.D, D_scale=self.D_scale,
                    chunk_size=self.chunk_size, dt_bias=self.dt_bias,
                    initial_states=None, seq_idx=None,
                    cu_seqlens=None, dt_softplus=self.delta_softplus,
                    dt_limit=self.dt_limit, mm_dtype=torch.float16,
                    x_row_scale_chp=self.x_chunk_state_scale_chp,
                    ori_x_row_scale_chp=self.ori_x_chunk_state_scale_chp,
                    B_row_scale_cgn=self.B_chunk_state_scale_cgn,
                    cb_chunkscan_scale = self.cb_chunkscan_scale,
                    C_chunkscan_scale = self.C_chunkscan_scale,
                    state_chunkscan_scale = self.state_chunkscan_scale,
                    )
        else:
            raise ValueError("x_head_group_range and x_dim_group_range must be both None or both not None")

    @torch.no_grad()
    def forward(self, x, dt, B, C, z=None, return_final_states=False, comp_calib=False):
        # output y is fp16
        y, final_states = self.chunk_scan_combined_fwd(
            q_x=x, q_dt=dt, q_B=B, q_C=C, q_z=z,
            z_scale=self.z_scale.cuda() if z is not None else None,
            comp_calib=comp_calib
        )
        return y if not return_final_states else (y, final_states)
    def __repr__(self):
        return f"QChunkScan()"