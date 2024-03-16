#!/usr/bin/env python3

import torch
import torch.nn.functional as F

# (silu(x @ w1) * x @ w3) @ w2
def ref_ff(x, w1, w2, w3):
    # [q_len, hidden_dim] @ [hidden_dim, dim] -> [q_len, dim]
    w1_x = torch.matmul(x, w1)
    silu_w1_x = F.silu(w1_x)
    # [q_len, hidden_dim] @ [hidden_dim, dim] -> [q_len, dim]
    w3_x = torch.matmul(x, w3)
    # [q_len, dim] @ [dim, hidden_dim] -> [q_len, hidden_dim]
    return torch.matmul(silu_w1_x * w3_x, w2)
    

# how to simulate the fused_ff function in one pass?
def fused_ff(Q, K, V):
    pass

if __name__ == "__main__":
    q_len = 2
    dim = 4
    hidden_dim = 4 * dim
    
    torch.manual_seed(1)
    
    x = torch.randn(q_len, dim)
    w1 = torch.randn(dim, hidden_dim)
    w2 = torch.randn(hidden_dim, dim)
    w3 = torch.randn(dim, hidden_dim)

    ref_out = ref_ff(x, w1, w2, w3)
    print(ref_out)
    
    # out_1 = fused_ff(Q, K, V)
    # print(torch.allclose(out_1, ref_out, atol=1e-5))