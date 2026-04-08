import math
import random

import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import auto_switch_platform

from mojo_opset.experimental import MojoDllmAttentionUpFunction


def generate_test_data(
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seqs: list,
    block_size: int,
):
    max_seq_length = sum(seqs)
    
    query = torch.randn(max_seq_length * 2, q_head_num, head_dim, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(max_seq_length * 2, kv_head_num, head_dim, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(max_seq_length * 2, kv_head_num, head_dim, dtype=torch.bfloat16, requires_grad=True)

    cu = [0]
    for seqlen in seqs:
        cu.append(cu[-1] + seqlen)
    cu_seqlen = torch.tensor(cu[1:], dtype=torch.int32)
    
    scale = 1.0 / math.sqrt(head_dim)

    return query, key, value, cu_seqlen, scale, block_size

FIXED_SEQ_LENGTHS = [
    #[2895,354, 9, 12,4]  # for msprof op simulator
    [28951,  3542, 99,   128,    48],
    #[32466,    69,    90,    41,    37,    59],
    # [  931,   745,  1608,  2149,   433, 16814,   268,   207,  2193,  4278,
    #       606,   254,   128,   192,   254,  1255,   182,   177,    61,    26],
    # [ 843,   50,  118,   71,  805,  325,  578,  199,  151,   74,  478,  275,
    #      101,  193,   89,   82,  340,  110,  118, 1010, 1463, 1226,  491,  638,
    #      603,  157, 3754,  530, 1233,  608,  797, 2788,  265,  486,  472, 1482,
    #      573,   84,  489, 1381,  148,  207,  777,  258, 1094,  133,  676,  137,
    #       88,  508,  455,  365,  234,  155, 1024,   97,  792,   58]
    
    #[24951,  3542, 99,   128,    48], # 28951 to be to 24951 compare precision, due to torch OOM
]
test_params = []
for seqs in FIXED_SEQ_LENGTHS:
    test_params.append(
        pytest.param(
            *generate_test_data(
                q_head_num=5, #1 2 5 8 20 40
                kv_head_num=1,
                head_dim=128,
                seqs=seqs,
                block_size=8,
            ),
            id=f"seq_set_{len(test_params)+1}"
        )
    )


@pytest.mark.parametrize(
    "query, key, value, cu_seqlen, scale, block_size",
    test_params,
)
# @pytest.mark.skip
@auto_switch_platform(set_perf=True)
def test_diffusion_attention_up_func(query, key, value, cu_seqlen, scale, block_size):
    print(f"cu_seqlen {cu_seqlen}")
    ctx = MockFunctionCtx()
    o = MojoDllmAttentionUpFunction.forward(ctx, query, key, value, cu_seqlen, scale, block_size)
    perf(lambda:MojoDllmAttentionUpFunction.forward(ctx, query, key, value, cu_seqlen, scale, block_size))

    #ctx_ref = MockFunctionCtx()
    # o_ref = MojoDllmAttentionUpFunction._registry.get("torch").forward(
    #     ctx_ref, query, key, value, cu_seqlen, scale, block_size
    # )

    #assert_close(o, o_ref)

    do = torch.rand_like(o)
    grads = MojoDllmAttentionUpFunction.backward(ctx, do)

    #grads_ref = MojoDllmAttentionUpFunction._registry.get("torch").backward(ctx_ref, do)
    perf(lambda: MojoDllmAttentionUpFunction.backward(ctx, do))
    # assert_close(grads, grads_ref)
