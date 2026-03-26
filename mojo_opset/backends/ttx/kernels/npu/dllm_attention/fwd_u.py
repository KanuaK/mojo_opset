import torch
import triton
import triton.language as tl

from .micro_kernel import micro_kernel_fwd


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
        )
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def kernel_da_fwd_u_single(
    q,
    k,
    v,
    o,
    lse,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ul,
    mask_ur,
    GROUP_SIZE: tl.constexpr,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    offset_r_local = tl.arange(0, BLOCK_R)[:, None]
    offset_c_local = tl.arange(0, BLOCK_R)[None, :]
    block_mask_ul = tl.load(mask_ul + offset_r_local * STRIDE_MASK + offset_c_local)
    block_mask_ur = tl.load(mask_ur + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N,
            pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            offs_h = tl.arange(0, H)

            ptr_q = (
                q
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_o = (
                o
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_lse = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_o = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)
            block_l = tl.full([BLOCK_R], 0.0, dtype=tl.float32)
            block_m = tl.full([BLOCK_R], -1e6, dtype=tl.float32)

            boundary_mask = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed) &
                (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q, k, v, block_o, block_m, block_l, scale,
                seq_st + idx_r * BLOCK_R, seq_ed,
                block_mask_ul, idx_n, offs_h,
                STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                GROUP_SIZE, BLOCK_R, boundary_mask=boundary_mask,
            )

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q, k, v, block_o, block_m, block_l, scale,
                S + seq_st + idx_r * BLOCK_R, S + seq_ed,
                block_mask_ur, idx_n, offs_h,
                STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                GROUP_SIZE, BLOCK_R, boundary_mask=boundary_mask,
            )

            for idx_tile_r in range(idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R, idx_r):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q, k, v, block_o, block_m, block_l, scale,
                    S + seq_st + idx_tile_r * BLOCK_R, S + seq_ed,
                    None, idx_n, offs_h,
                    STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                    STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                    GROUP_SIZE, BLOCK_R,
                )

            for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q, k, v, block_o, block_m, block_l, scale,
                    S + seq_st + idx_c * BLOCK_C, S + seq_ed,
                    None, idx_n, offs_h,
                    STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                    STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                    GROUP_SIZE, BLOCK_C,
                )

            block_o = block_o / block_l[:, None]
            block_lse = tl.log(block_l) + block_m
            tl.store(ptr_o, block_o, mask=mask_q)
            tl.store(ptr_lse, block_lse, mask=mask_lse)

        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[triton.Config({"BLOCK_R": 64})],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_fwd_u_ul(
    q, k, v, o_out, m_out, l_out,
    cu_seqlens, num_seqs, scale, mask_ul,
    GROUP_SIZE: tl.constexpr, S,
    N: tl.constexpr, H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr, STRIDE_Q_N: tl.constexpr, STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr, STRIDE_K_N: tl.constexpr, STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr, STRIDE_V_N: tl.constexpr, STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr, STRIDE_D_N,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    offset_r_local = tl.arange(0, BLOCK_R)[:, None]
    offset_c_local = tl.arange(0, BLOCK_R)[None, :]
    block_mask_ul = tl.load(mask_ul + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N, pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            offs_h = tl.arange(0, H)

            ptr_q = q + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_o = o_out + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_m = m_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_l = l_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_o = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)
            block_l = tl.full([BLOCK_R], 0.0, dtype=tl.float32)
            block_m = tl.full([BLOCK_R], -1e6, dtype=tl.float32)

            boundary_mask = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed)
                & (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q, k, v, block_o, block_m, block_l, scale,
                seq_st + idx_r * BLOCK_R, seq_ed,
                block_mask_ul, idx_n, offs_h,
                STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                GROUP_SIZE, BLOCK_R, boundary_mask=boundary_mask,
            )

            tl.store(ptr_o, block_o, mask=mask_q)
            tl.store(ptr_m, block_m, mask=mask_d)
            tl.store(ptr_l, block_l, mask=mask_d)

        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[triton.Config({"BLOCK_R": 64})],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_fwd_u_ur(
    q, k, v, o_out, m_out, l_out,
    cu_seqlens, num_seqs, scale, mask_ur,
    GROUP_SIZE: tl.constexpr, S,
    N: tl.constexpr, H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr, STRIDE_Q_N: tl.constexpr, STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr, STRIDE_K_N: tl.constexpr, STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr, STRIDE_V_N: tl.constexpr, STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr, STRIDE_D_N,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    offset_r_local = tl.arange(0, BLOCK_R)[:, None]
    offset_c_local = tl.arange(0, BLOCK_R)[None, :]
    block_mask_ur = tl.load(mask_ur + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N, pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            offs_h = tl.arange(0, H)

            ptr_q = q + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_o = o_out + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_m = m_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_l = l_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_o = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)
            block_l = tl.full([BLOCK_R], 0.0, dtype=tl.float32)
            block_m = tl.full([BLOCK_R], -1e6, dtype=tl.float32)

            boundary_mask = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed)
                & (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q, k, v, block_o, block_m, block_l, scale,
                S + seq_st + idx_r * BLOCK_R, S + seq_ed,
                block_mask_ur, idx_n, offs_h,
                STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                GROUP_SIZE, BLOCK_R, boundary_mask=boundary_mask,
            )

            tl.store(ptr_o, block_o, mask=mask_q)
            tl.store(ptr_m, block_m, mask=mask_d)
            tl.store(ptr_l, block_l, mask=mask_d)

        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[triton.Config({"BLOCK_R": 64, "BLOCK_C": 256})],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_fwd_u_residual(
    q, k, v, o_out, m_out, l_out,
    cu_seqlens, num_seqs, scale,
    GROUP_SIZE: tl.constexpr, S,
    N: tl.constexpr, H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr, STRIDE_Q_N: tl.constexpr, STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr, STRIDE_K_N: tl.constexpr, STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr, STRIDE_V_N: tl.constexpr, STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr, STRIDE_D_N,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N, pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            offs_h = tl.arange(0, H)

            ptr_q = q + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_o = o_out + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_m = m_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_l = l_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_o = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)
            block_l = tl.full([BLOCK_R], 0.0, dtype=tl.float32)
            block_m = tl.full([BLOCK_R], -1e6, dtype=tl.float32)

            for idx_tile_r in range(idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R, idx_r):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q, k, v, block_o, block_m, block_l, scale,
                    S + seq_st + idx_tile_r * BLOCK_R, S + seq_ed,
                    None, idx_n, offs_h,
                    STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                    STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                    GROUP_SIZE, BLOCK_R,
                )

            tl.store(ptr_o, block_o, mask=mask_q)
            tl.store(ptr_m, block_m, mask=mask_d)
            tl.store(ptr_l, block_l, mask=mask_d)

        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[triton.Config({"BLOCK_R": 64, "BLOCK_C": 256})],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_fwd_u_aligned(
    q, k, v, o_out, m_out, l_out,
    cu_seqlens, num_seqs, scale,
    GROUP_SIZE: tl.constexpr, S,
    N: tl.constexpr, H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr, STRIDE_Q_N: tl.constexpr, STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr, STRIDE_K_N: tl.constexpr, STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr, STRIDE_V_N: tl.constexpr, STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr, STRIDE_D_N,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N, pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            offs_h = tl.arange(0, H)

            ptr_q = q + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_o = o_out + idx_n * STRIDE_Q_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
            ptr_m = m_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_l = l_out + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_o = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)
            block_l = tl.full([BLOCK_R], 0.0, dtype=tl.float32)
            block_m = tl.full([BLOCK_R], -1e6, dtype=tl.float32)

            for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q, k, v, block_o, block_m, block_l, scale,
                    S + seq_st + idx_c * BLOCK_C, S + seq_ed,
                    None, idx_n, offs_h,
                    STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
                    STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
                    GROUP_SIZE, BLOCK_C,
                )

            tl.store(ptr_o, block_o, mask=mask_q)
            tl.store(ptr_m, block_m, mask=mask_d)
            tl.store(ptr_l, block_l, mask=mask_d)

        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


def kernel_da_fwd_u(
    q, k, v, o, lse,
    cu_seqlens, num_seqs, scale, mask_ul, mask_ur,
    GROUP_SIZE, S, N, H,
    STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
    STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
    STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
    STRIDE_D_S, STRIDE_D_N,
    STRIDE_MASK,
    num_cores,
):
    def _make_o_ws():
        return torch.zeros_like(o)

    def _make_m_ws():
        ws = torch.empty(lse.shape[1], lse.shape[0], dtype=lse.dtype, device=lse.device).T
        ws.fill_(-1e6)
        return ws

    def _make_l_ws():
        return torch.zeros(lse.shape[1], lse.shape[0], dtype=lse.dtype, device=lse.device).T

    o_ul, m_ul, l_ul = _make_o_ws(), _make_m_ws(), _make_l_ws()
    o_ur, m_ur, l_ur = _make_o_ws(), _make_m_ws(), _make_l_ws()
    o_residual, m_residual, l_residual = _make_o_ws(), _make_m_ws(), _make_l_ws()
    o_aligned, m_aligned, l_aligned = _make_o_ws(), _make_m_ws(), _make_l_ws()

    _kernel_fwd_u_ul[(num_cores,)](
        q, k, v, o_ul, m_ul, l_ul,
        cu_seqlens, num_seqs, scale, mask_ul,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N, STRIDE_MASK,
    )

    _kernel_fwd_u_ur[(num_cores,)](
        q, k, v, o_ur, m_ur, l_ur,
        cu_seqlens, num_seqs, scale, mask_ur,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N, STRIDE_MASK,
    )

    _kernel_fwd_u_residual[(num_cores,)](
        q, k, v, o_residual, m_residual, l_residual,
        cu_seqlens, num_seqs, scale,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N,
    )

    _kernel_fwd_u_aligned[(num_cores,)](
        q, k, v, o_aligned, m_aligned, l_aligned,
        cu_seqlens, num_seqs, scale,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N,
    )

    # Pairwise online-softmax merge
    def _merge(o_a, m_a, l_a, o_b, m_b, l_b):
        m_new = torch.maximum(m_a, m_b)
        s_a = torch.exp(m_a - m_new)
        s_b = torch.exp(m_b - m_new)
        l_new = s_a * l_a + s_b * l_b
        o_new = s_a.unsqueeze(-1) * o_a + s_b.unsqueeze(-1) * o_b
        return o_new, m_new, l_new

    o_acc, m_acc, l_acc = _merge(o_ul, m_ul, l_ul, o_ur, m_ur, l_ur)
    o_acc, m_acc, l_acc = _merge(o_acc, m_acc, l_acc, o_residual, m_residual, l_residual)
    o_acc, m_acc, l_acc = _merge(o_acc, m_acc, l_acc, o_aligned, m_aligned, l_aligned)

    # Normalize and write to output
    o[:] = o_acc / l_acc.unsqueeze(-1)
    lse[:] = torch.log(l_acc) + m_acc
