#!/usr/bin/env python3

import torch
from utils import ref_attention

BLOCK_SIZE = 2
NEG_INF = -1e10 # -infinity
EPSILON = 1e-10

# python implementation of flash_attention_1
# https://arxiv.org/abs/2205.14135
def flash_attention_1(Q, K, V):
    # decide the block size for Q and KV
    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[0])
    KV_BLOCK_SIZE = min(BLOCK_SIZE, K.shape[0])

    # divide Q into Tr blocks Q1, ..., QTr of size [Q_BLOCK_SIZE, dim]
    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=0)
    Tr = len(Q_BLOCKS)
    
    # divide KV into Tc blocks K1, ..., KTc of size [KV_BLOCK_SIZE, dim]
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=0)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=0)
    Tc = len(K_BLOCKS)

    # divide O, l, m into Tr blocks of size [Q_BLOCK_SIZE, 1]
    # [q_len, dim]
    O = torch.zeros_like(Q)
    # [q_len, 1]
    l = torch.zeros(Q.shape[:-1])[...,None]
    # [q_len, 1]
    m = torch.ones(Q.shape[:-1])[...,None] * NEG_INF
    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=0))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=0))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=0))

    for j in range(Tc):
        # load kj, vj from global memory to shared memory
        # using async vectorized operations
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]

        for i in range(Tr):
            # load qi, oi, li, mi from global memory to shared memory
            # using async vectorized operations
            Qi = Q_BLOCKS[i]
            Oi = O_BLOCKS[i]
            li = l_BLOCKS[i]
            mi = m_BLOCKS[i]

            # On chip, compute Sij = Qi @ KjT, using tensor cores
            S_ij = torch.einsum('q d, k d -> q k', Qi, Kj)

            # on chip, compute mij = rowmax(Sij) [q_len, 1]
            m_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            # on chip, compute Pij = exp(Sij - mij) [q_len, kv_len]
            P_ij = torch.exp(S_ij - m_ij)
            # on chip, compute lij = rowsum(Pij) [q_len, 1]
            l_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

            # on chip, compute mi_new = max(mij, mi) [q_len, 1]
            mi_new = torch.maximum(m_ij, mi)
            # on chip, compute li_new = exp(mi - mi_new) * li + exp(m_ij - mi_new) * l_ij [q_len, 1]
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_ij - mi_new) * l_ij
            # on chip, compute PijVj = Pij @ Vj [q_len, dim]
            P_ij_Vj = torch.einsum('q k, k d -> q d', P_ij, Vj)
            # on chip, compute Oi = (li/li_new) * exp(mi - mi_new) * Oi + exp(m_ij - mi_new) * PijVj / li_new [q_len, dim]
            Oi_new = (li/li_new) * torch.exp(mi - mi_new) * Oi + (torch.exp(m_ij - mi_new) / li_new) * P_ij_Vj
            
            # write Oi, li_new, mi_new back to global memory
            O_BLOCKS[i].copy_(Oi_new)
            l_BLOCKS[i].copy_(li_new)
            m_BLOCKS[i].copy_(mi_new)
    return O

if __name__ == "__main__":
    q_len = 4
    kv_len = 16
    dim = 8
    
    Q = torch.randn(q_len, dim)
    K = torch.randn(kv_len, dim)
    V = torch.randn(kv_len, dim)

    out_1 = flash_attention_1(Q, K, V)
    ref_out = ref_attention(Q, K, V)
    print(torch.allclose(out_1, ref_out, atol=1e-5))