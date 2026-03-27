import torch
import triton
import triton.language as tl

from .micro_kernel import micro_kernel_bwd_kv, micro_kernel_bwd_kv_test, micro_kernel_bwd_kv_test2


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_C": 64},
        ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_bwd_kv_ur_masked(
    q,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
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
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_c_st = 0
    NUM_GROUP = N // GROUP_SIZE

    offset_r_local = tl.arange(0, BLOCK_C)[:, None]
    offset_c_local = tl.arange(0, BLOCK_C)[None, :]
    block_mask_ur = tl.load(mask_ur + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_c_ed = offset_block_c_st + tl.cdiv(seq_ed - seq_st, BLOCK_C)
        for task_id in range(
            offset_block_c_st * NUM_GROUP + ((pid % pnum - offset_block_c_st * NUM_GROUP % pnum + pnum) % pnum),
            offset_block_c_ed * NUM_GROUP,
            pnum,
        ):
            idx_c = task_id // NUM_GROUP - offset_block_c_st
            idx_group = task_id % NUM_GROUP
            offs_h = tl.arange(0, H)

            ptr_k = (
                k
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            ptr_dk = (
                dk
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_dv = (
                dv
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            mask_kv = (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < seq_ed

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_dk = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)
            block_dv = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)

            block_k = tl.trans(block_k)
            block_v = tl.trans(block_v)

            for idx_ingroup in range(GROUP_SIZE):
                idx_n = idx_group * GROUP_SIZE + idx_ingroup

                boundary_mask = (
                    (seq_st + idx_c * BLOCK_C + offset_r_local < seq_ed)
                    & (seq_st + idx_c * BLOCK_C + offset_c_local < seq_ed)
                )

                block_dk, block_dv = micro_kernel_bwd_kv(
                    q,
                    block_k,
                    block_v,
                    do,
                    d,
                    block_dk,
                    block_dv,
                    lse,
                    scale,
                    seq_st + idx_c * BLOCK_C,
                    seq_ed,
                    block_mask_ur,
                    idx_n,
                    offs_h,
                    STRIDE_Q_S,
                    STRIDE_Q_N,
                    STRIDE_Q_H,
                    STRIDE_D_S,
                    STRIDE_D_N,
                    BLOCK_C,
                    boundary_mask=boundary_mask,
                )

            tl.store(ptr_dk, block_dk, mask=mask_kv)
            tl.store(ptr_dv, block_dv, mask=mask_kv)
        seq_st = seq_ed
        offset_block_c_st = offset_block_c_ed


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 256, "BLOCK_C": 64},
        ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_bwd_kv_ur_residual(
    q,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
    cu_seqlens,
    num_seqs,
    scale,
    scale_dv,
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
    offset_block_c_st = 0
    NUM_GROUP = N // GROUP_SIZE
    

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_c_ed = offset_block_c_st + tl.cdiv(seq_ed - seq_st, BLOCK_C)
        for task_id in range(
            offset_block_c_st * NUM_GROUP + ((pid % pnum - offset_block_c_st * NUM_GROUP % pnum + pnum) % pnum),
            offset_block_c_ed * NUM_GROUP,
            pnum,
        ):
            idx_c = task_id // NUM_GROUP - offset_block_c_st
            idx_group = task_id % NUM_GROUP
            offs_h = tl.arange(0, H)

            ptr_k = (
                k
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            ptr_dk = (
                dk
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_dv = (
                dv
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            mask_kv = (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < seq_ed

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_dk = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)
            block_dv = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)

            block_k = tl.trans(block_k)
            block_v = tl.trans(block_v)

            for idx_ingroup in range(GROUP_SIZE):
                idx_n = idx_group * GROUP_SIZE + idx_ingroup

                for idx_tile_r in range(idx_c + 1, (idx_c * BLOCK_C // BLOCK_R + 1) * BLOCK_R // BLOCK_C):
                    block_dk, block_dv = micro_kernel_bwd_kv_test(
                        q, q,
                        block_k,
                        block_v,
                        do,
                        d,
                        block_dk,
                        block_dv,
                        lse,
                        scale,
                        scale_dv,
                        seq_st + idx_tile_r * BLOCK_C,
                        seq_ed,
                        None,
                        idx_n,
                        offs_h,
                        STRIDE_Q_S,
                        STRIDE_Q_N,
                        STRIDE_Q_H,
                        STRIDE_D_S,
                        STRIDE_D_N,
                        BLOCK_C,
                    )
            tl.store(ptr_dk, block_dk, mask=mask_kv)
            tl.store(ptr_dv, block_dv, mask=mask_kv)
        seq_st = seq_ed
        offset_block_c_st = offset_block_c_ed


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 256, "BLOCK_C": 64},
        ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def _kernel_bwd_kv_ur_aligned(
    q, q2,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
    cu_seqlens,
    num_seqs,
    scale,
    scale_dv,
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
    offset_block_c_st = 0
    NUM_GROUP = N // GROUP_SIZE

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_c_ed = offset_block_c_st + tl.cdiv(seq_ed - seq_st, BLOCK_C)
        for task_id in range(
            offset_block_c_st * NUM_GROUP + ((pid % pnum - offset_block_c_st * NUM_GROUP % pnum + pnum) % pnum),
            offset_block_c_ed * NUM_GROUP,
            pnum,
        ):
            idx_c = task_id // NUM_GROUP - offset_block_c_st
            idx_group = task_id % NUM_GROUP
            offs_h = tl.arange(0, H)

            ptr_k = (
                k
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            ptr_dk = (
                dk
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_dv = (
                dv
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            mask_kv = (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < seq_ed

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_dk = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)
            block_dv = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)

            block_k = tl.trans(block_k)
            block_v = tl.trans(block_v)

            for idx_ingroup in range(GROUP_SIZE):
                idx_n = idx_group * GROUP_SIZE + idx_ingroup

                for idx_r in range(idx_c * BLOCK_C // BLOCK_R + 1, (seq_ed - seq_st + BLOCK_R - 1) // BLOCK_R):
                    # block_dk, block_dv = micro_kernel_bwd_kv_test(
                    #     q, q2,
                    #     block_k,
                    #     block_v,
                    #     do,
                    #     d,
                    #     block_dk,
                    #     block_dv,
                    #     lse,
                    #     scale,
                    #     scale_dv,
                    #     seq_st + idx_r * BLOCK_R,
                    #     seq_ed,
                    #     None,
                    #     idx_n,
                    #     offs_h,
                    #     STRIDE_Q_S,
                    #     STRIDE_Q_N,
                    #     STRIDE_Q_H,
                    #     STRIDE_D_S,
                    #     STRIDE_D_N,
                    #     BLOCK_R,
                    # )
                    block_dk, block_dv = micro_kernel_bwd_kv_test2(
                        q,
                        block_k,
                        block_v,
                        do,
                        d,
                        block_dk,
                        block_dv,
                        lse,
                        scale,
                        scale_dv,
                        seq_st + idx_r * BLOCK_R,
                        seq_ed,
                        None,
                        idx_n,
                        offs_h,
                        STRIDE_Q_S,
                        STRIDE_Q_N,
                        STRIDE_Q_H,
                        STRIDE_D_S,
                        STRIDE_D_N,
                        BLOCK_R,
                    )

            tl.store(ptr_dk, block_dk, mask=mask_kv)
            tl.store(ptr_dv, block_dv, mask=mask_kv)
        seq_st = seq_ed
        offset_block_c_st = offset_block_c_ed

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 256, "BLOCK_C": 64},
        )
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def kernel_da_bwd_kv_ur_single(
    q,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
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
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_c_st = 0
    NUM_GROUP = N // GROUP_SIZE

    offset_r_local = tl.arange(0, BLOCK_C)[:, None]
    offset_c_local = tl.arange(0, BLOCK_C)[None, :]
    block_mask_ur = tl.load(mask_ur + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_c_ed = offset_block_c_st + tl.cdiv(seq_ed - seq_st, BLOCK_C)
        for task_id in range(
            offset_block_c_st * NUM_GROUP + ((pid % pnum - offset_block_c_st * NUM_GROUP % pnum + pnum) % pnum),
            offset_block_c_ed * NUM_GROUP,
            pnum,
        ):
            idx_c = task_id // NUM_GROUP - offset_block_c_st
            idx_group = task_id % NUM_GROUP
            offs_h = tl.arange(0, H)

            ptr_k = (
                k
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            ptr_dk = (
                dk
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_dv = (
                dv
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            mask_kv = (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < seq_ed

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_dk = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)
            block_dv = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)

            block_k = tl.trans(block_k)
            block_v = tl.trans(block_v)

            for idx_ingroup in range(GROUP_SIZE):
                idx_n = idx_group * GROUP_SIZE + idx_ingroup

                boundary_mask = (
                    (seq_st + idx_c * BLOCK_C + offset_r_local < seq_ed)
                    & (seq_st + idx_c * BLOCK_C + offset_c_local < seq_ed)
                )

                block_dk, block_dv = micro_kernel_bwd_kv(
                    q,
                    block_k,
                    block_v,
                    do,
                    d,
                    block_dk,
                    block_dv,
                    lse,
                    scale,
                    seq_st + idx_c * BLOCK_C,
                    seq_ed,
                    block_mask_ur,
                    idx_n,
                    offs_h,
                    STRIDE_Q_S,
                    STRIDE_Q_N,
                    STRIDE_Q_H,
                    STRIDE_D_S,
                    STRIDE_D_N,
                    BLOCK_C,
                    boundary_mask=boundary_mask,
                )

                for idx_tile_r in range(idx_c + 1, (idx_c * BLOCK_C // BLOCK_R + 1) * BLOCK_R // BLOCK_C):
                    block_dk, block_dv = micro_kernel_bwd_kv(
                        q,
                        block_k,
                        block_v,
                        do,
                        d,
                        block_dk,
                        block_dv,
                        lse,
                        scale,
                        seq_st + idx_tile_r * BLOCK_C,
                        seq_ed,
                        None,
                        idx_n,
                        offs_h,
                        STRIDE_Q_S,
                        STRIDE_Q_N,
                        STRIDE_Q_H,
                        STRIDE_D_S,
                        STRIDE_D_N,
                        BLOCK_C,
                    )

                for idx_r in range(idx_c * BLOCK_C // BLOCK_R + 1, (seq_ed - seq_st + BLOCK_R - 1) // BLOCK_R):
                    block_dk, block_dv = micro_kernel_bwd_kv(
                        q,
                        block_k,
                        block_v,
                        do,
                        d,
                        block_dk,
                        block_dv,
                        lse,
                        scale,
                        seq_st + idx_r * BLOCK_R,
                        seq_ed,
                        None,
                        idx_n,
                        offs_h,
                        STRIDE_Q_S,
                        STRIDE_Q_N,
                        STRIDE_Q_H,
                        STRIDE_D_S,
                        STRIDE_D_N,
                        BLOCK_R,
                    )

            tl.store(ptr_dk, block_dk.to(tl.bfloat16), mask=mask_kv)
            tl.store(ptr_dv, block_dv.to(tl.bfloat16), mask=mask_kv)
        seq_st = seq_ed
        offset_block_c_st = offset_block_c_ed



def kernel_da_bwd_kv_ur(
    q,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ur,
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
    
    # dk_workspace = torch.zeros_like(dk, dtype=torch.float32).npu()
    # dv_workspace = torch.zeros_like(dv, dtype=torch.float32).npu()
    
    dk_masked = torch.zeros_like(dk, dtype=torch.float32).npu()
    dv_masked = torch.zeros_like(dv, dtype=torch.float32).npu()
    dk_residual = torch.zeros_like(dk, dtype=torch.float32).npu()
    dv_residual = torch.zeros_like(dv, dtype=torch.float32).npu()
    dk_aligned = torch.zeros_like(dk, dtype=torch.float32).npu()
    dv_aligned = torch.zeros_like(dv, dtype=torch.float32).npu()

    _kernel_bwd_kv_ur_masked[(num_cores,)](
        q, k, v, do, d, lse, dk_masked, dv_masked,
        cu_seqlens, num_seqs, scale, mask_ur,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N, STRIDE_MASK,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4, 
        tile_mix_vector_loop=2,
        tile_mix_cube_loop=4,
    )
    scale_dv = 1.0
    _kernel_bwd_kv_ur_residual[(num_cores,)](
        q, k, v, do, d, lse, dk_residual, dv_residual,
        cu_seqlens, num_seqs, scale, scale_dv,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4, 
        tile_mix_vector_loop=2,
        tile_mix_cube_loop=4,
    )
    _kernel_bwd_kv_ur_aligned[(num_cores,)](
        q, q, k, v, do, d, lse, dk_aligned, dv_aligned,
        cu_seqlens, num_seqs, scale, scale_dv,
        GROUP_SIZE, S, N, H,
        STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
        STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
        STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
        STRIDE_D_S, STRIDE_D_N,
        # limit_auto_multi_buffer_only_for_local_buffer=False,
        # set_workspace_multibuffer=4, 
        # tile_mix_vector_loop=2,
        # tile_mix_cube_loop=4,
    )

    # dk[S:] = dk_workspace[S:].to(torch.bfloat16)
    # dv[S:] = dv_workspace[S:].to(torch.bfloat16)
    
    dk[S:] = (dk_masked[S:] + dk_residual[S:] + dk_aligned[S:]).to(torch.bfloat16)
    dv[S:] = (dv_masked[S:] + dv_residual[S:] + dv_aligned[S:]).to(torch.bfloat16)
    
    #####  debug test part

    # _kernel_bwd_kv_ur_masked_2[(num_cores,)](
    #     q, k, v, do, d, lse, dk_masked, dv_masked,
    #     cu_seqlens, num_seqs, scale, mask_ur,
    #     GROUP_SIZE, S, N, H,
    #     STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
    #     STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
    #     STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
    #     STRIDE_D_S, STRIDE_D_N, STRIDE_MASK,
    #     # enable_ubuf_saving=True,
    #     # multibuffer=True, # 现在默认是True，可不写，控制开double_buffer
    #     # unit_flag=True, #cube搬出的一个优化项，比较容易卡死
    #     # limit_auto_multi_buffer_only_for_local_buffer=False,
    #     # set_workspace_multibuffer=4, 
    #     # tile_mix_vector_loop=2,   # 可以2，4，8，主要修改此值
    #     # tile_mix_cube_loop=4,    #可以2，4，8 去配，主要修改此值
    # )

    # dk[S:] = (dk_masked[S:] + dk_aligned[S:]).to(torch.bfloat16)
    # dv[S:] = (dv_masked[S:] + dv_aligned[S:]).to(torch.bfloat16)


    # dk_single = torch.zeros_like(dk, dtype=torch.float32).npu()
    # dv_single = torch.zeros_like(dv, dtype=torch.float32).npu()

    # kernel_da_bwd_kv_ur_single[(num_cores,)](
    #     q, k, v, do, d, lse, dk_single, dv_single,
    #     cu_seqlens, num_seqs, scale, mask_ur,
    #     GROUP_SIZE, S, N, H,
    #     STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
    #     STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
    #     STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
    #     STRIDE_D_S, STRIDE_D_N, STRIDE_MASK,
    # )
    # _kernel_bwd_kv_ur_aligned[(num_cores,)](
    #     q, k, v, do, d, lse, dk_single, dv_single,
    #     cu_seqlens, num_seqs, scale, scale_dv,
    #     GROUP_SIZE, S, N, H,
    #     STRIDE_Q_S, STRIDE_Q_N, STRIDE_Q_H,
    #     STRIDE_K_S, STRIDE_K_N, STRIDE_K_H,
    #     STRIDE_V_S, STRIDE_V_N, STRIDE_V_H,
    #     STRIDE_D_S, STRIDE_D_N,
    # )
    

    # diff = (dk_masked[S:] - dk_single[S:]).abs()
    # diff = (dk_residual[S:] - dk_single[S:]).abs()
    # diff = (dk_aligned[S:] - dk_single[S:]).abs()
    # diff = (dk[S:] - dk_single[S:]).abs()
    # print(f"dk max diff: {diff.max()}, nonzero positions: {(diff > 1e-6).sum()}")
    # diff = (dv_masked[S:] - dv_single[S:]).abs()
    # diff = (dv_residual[S:] - dv_single[S:]).abs()
    # diff = (dv_aligned[S:] - dv_single[S:]).abs()
    # diff = (dv[S:] - dv_single[S:]).abs()
    # print(f"dv max diff: {diff.max()}, nonzero positions: {(diff > 1e-6).sum()}")