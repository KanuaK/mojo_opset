import torch
import triton
import triton.language as tl

from .micro_kernel import micro_kernel_bwd_q


@triton.autotune(
    configs=[
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=2,
        #     tile_mix_cube_loop=2,
        # ),
        triton.Config(
            {"BLOCK_R": 64},
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=4,
        ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=2,
        #     tile_mix_cube_loop=8,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=2,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=4,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=8,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=2,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=4,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=8,
        # ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_bwd_q_u_masked_ul(
    q,
    k,
    v,
    do,
    d,
    lse,
    dq,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ul,
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
            ptr_do = (
                do
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_dq = (
                dq
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_d = d + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
            block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
            block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
            block_dq = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)

            boundary_mask = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed) &
                (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )

            block_dq = micro_kernel_bwd_q(
                block_q,
                k,
                v,
                block_do,
                block_d,
                block_dq,
                block_lse,
                scale,
                seq_st + idx_r * BLOCK_R,
                seq_ed,
                block_mask_ul,
                idx_n,
                offs_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                boundary_mask=boundary_mask,
            )

            tl.store(ptr_dq, block_dq, mask=mask_q)
        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=2,
        #     tile_mix_cube_loop=2,
        # ),
        triton.Config(
            {"BLOCK_R": 64},
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=4,
        ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=2,
        #     tile_mix_cube_loop=8,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=2,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=4,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=8,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=2,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=4,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=8,
        # ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_bwd_q_u_masked_ur(
    q,
    k,
    v,
    do,
    d,
    lse,
    dq,
    cu_seqlens,
    num_seqs,
    scale,
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
            ptr_do = (
                do
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_dq = (
                dq
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_d = d + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
            block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
            block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
            block_dq = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)

            boundary_mask = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed) &
                (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )

            block_dq = micro_kernel_bwd_q(
                block_q,
                k,
                v,
                block_do,
                block_d,
                block_dq,
                block_lse,
                scale,
                S + seq_st + idx_r * BLOCK_R,
                S + seq_ed,
                block_mask_ur,
                idx_n,
                offs_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                boundary_mask=boundary_mask,
            )

            tl.store(ptr_dq, block_dq, mask=mask_q)
        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=2,
        #     tile_mix_cube_loop=2,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=2,
        #     tile_mix_cube_loop=4,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=2,
        #     tile_mix_cube_loop=8,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=2,
        # ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=4,
            tile_mix_cube_loop=4,
        ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=4,
        #     tile_mix_cube_loop=8,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=2,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=4,
        # ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        #     tile_mix_vector_loop=8,
        #     tile_mix_cube_loop=8,
        # ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_bwd_q_u_residual(
    q,
    k,
    v,
    do,
    d,
    lse,
    dq,
    cu_seqlens,
    num_seqs,
    scale,
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
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
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
            ptr_do = (
                do
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_dq = (
                dq
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_d = d + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
            block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
            block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
            block_dq = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)

            for idx_tile_r in range(idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R, idx_r):
                block_dq = micro_kernel_bwd_q(
                    block_q,
                    k,
                    v,
                    block_do,
                    block_d,
                    block_dq,
                    block_lse,
                    scale,
                    S + seq_st + idx_tile_r * BLOCK_R,
                    S + seq_ed,
                    None,
                    idx_n,
                    offs_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_R,
                )

            tl.store(ptr_dq, block_dq, mask=mask_q)
        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=4,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=8,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=4,
            tile_mix_cube_loop=2,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=4,
            tile_mix_cube_loop=4,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=4,
            tile_mix_cube_loop=8,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=8,
            tile_mix_cube_loop=2,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=8,
            tile_mix_cube_loop=4,
        ),
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            tile_mix_vector_loop=8,
            tile_mix_cube_loop=8,
        ),
        # triton.Config(
        #     {"BLOCK_R": 64, "BLOCK_C": 256},
        # ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_bwd_q_u_aligned(
    q,
    k,
    v,
    do,
    d,
    lse,
    dq,
    cu_seqlens,
    num_seqs,
    scale,
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
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
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
            ptr_do = (
                do
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_dq = (
                dq
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_d = d + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
            block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
            block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
            block_dq = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)

            for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
                block_dq = micro_kernel_bwd_q(
                    block_q,
                    k,
                    v,
                    block_do,
                    block_d,
                    block_dq,
                    block_lse,
                    scale,
                    S + seq_st + idx_c * BLOCK_C,
                    S + seq_ed,
                    None,
                    idx_n,
                    offs_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_C,
                )

            tl.store(ptr_dq, block_dq, mask=mask_q)
        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
        )
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def kernel_da_bwd_q_u_single(
    q,
    k,
    v,
    do,
    d,
    lse,
    dq,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ur,
    mask_ul,
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
            ptr_do = (
                do
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_dq = (
                dq
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_d = d + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
            block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
            block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
            block_dq = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)

            boundary_mask = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed) &
                (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )

            block_dq = micro_kernel_bwd_q(
                block_q,
                k,
                v,
                block_do,
                block_d,
                block_dq,
                block_lse,
                scale,
                seq_st + idx_r * BLOCK_R,
                seq_ed,
                block_mask_ul,
                idx_n,
                offs_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                boundary_mask=boundary_mask,
            )

            block_dq = micro_kernel_bwd_q(
                block_q,
                k,
                v,
                block_do,
                block_d,
                block_dq,
                block_lse,
                scale,
                S + seq_st + idx_r * BLOCK_R,
                S + seq_ed,
                block_mask_ur,
                idx_n,
                offs_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                boundary_mask=boundary_mask,
            )

            for idx_tile_r in range(idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R, idx_r):
                block_dq = micro_kernel_bwd_q(
                    block_q,
                    k,
                    v,
                    block_do,
                    block_d,
                    block_dq,
                    block_lse,
                    scale,
                    S + seq_st + idx_tile_r * BLOCK_R,
                    S + seq_ed,
                    None,
                    idx_n,
                    offs_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_R,
                )

            for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
                block_dq = micro_kernel_bwd_q(
                    block_q,
                    k,
                    v,
                    block_do,
                    block_d,
                    block_dq,
                    block_lse,
                    scale,
                    S + seq_st + idx_c * BLOCK_C,
                    S + seq_ed,
                    None,
                    idx_n,
                    offs_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_C,
                )

            tl.store(ptr_dq, block_dq.to(tl.bfloat16), mask=mask_q)
        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


def kernel_da_bwd_q_u(
    q,
    k,
    v,
    do,
    d,
    lse,
    dq,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ur,
    mask_ul,
    GROUP_SIZE,
    S,
    N,
    H,
    STRIDE_Q_S,
    STRIDE_Q_N,
    STRIDE_Q_H,
    STRIDE_K_S,
    STRIDE_K_N,
    STRIDE_K_H,
    STRIDE_V_S,
    STRIDE_V_N,
    STRIDE_V_H,
    STRIDE_D_S,
    STRIDE_D_N,
    STRIDE_MASK,
    num_cores,
):
    dq_masked_ul = torch.zeros_like(dq, dtype=torch.float32).npu()
    dq_masked_ur = torch.zeros_like(dq, dtype=torch.float32).npu()
    dq_residual = torch.zeros_like(dq, dtype=torch.float32).npu()
    dq_aligned = torch.zeros_like(dq, dtype=torch.float32).npu()

    _kernel_bwd_q_u_masked_ul[(num_cores,)](
        q, k, v, do, d, lse, dq_masked_ul,
        cu_seqlens, num_seqs, scale, mask_ul,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N, STRIDE_MASK,
        enable_ubuf_saving=True,
        multibuffer=True,
        unit_flag=True,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4,
    )
    _kernel_bwd_q_u_masked_ur[(num_cores,)](
        q, k, v, do, d, lse, dq_masked_ur,
        cu_seqlens, num_seqs, scale, mask_ur,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N, STRIDE_MASK,
        enable_ubuf_saving=True,
        multibuffer=True,
        unit_flag=True,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4,
    )
    _kernel_bwd_q_u_residual[(num_cores,)](
        q, k, v, do, d, lse, dq_residual,
        cu_seqlens, num_seqs, scale,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N,
        enable_ubuf_saving=True,
        multibuffer=True,
        unit_flag=True,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4,
    )
    _kernel_bwd_q_u_aligned[(num_cores,)](
        q, k, v, do, d, lse, dq_aligned,
        cu_seqlens, num_seqs, scale,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N,
        enable_ubuf_saving=True,
        multibuffer=True,
        unit_flag=True,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4,
    )

    dq[:] = (dq_masked_ul + dq_masked_ur + dq_residual + dq_aligned)#.to(torch.bfloat16)
