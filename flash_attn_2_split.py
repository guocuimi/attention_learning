import torch

BLOCK_SIZE = 2
NUM_SPLITS = 4
NEG_INF = -1e10 # -infinity
EPSILON = 1e-10

def normal_attention(Q, K, V):
    # [q_len, dim] @ [kv_len, dim] -> [q_len, kv_len]  
    QKt = torch.einsum('q d, k d -> q k', Q, K)
    attn = torch.nn.functional.softmax(QKt, dim=-1)
    # [q_len, kv_len] @ [kv_len, dim] -> [q_len, dim]
    return attn @ V

# python implementation of flash_attention_2
def flash_attention_2(Q, K, V):
    # decide the block size for Q and KV
    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
    KV_BLOCK_SIZE = BLOCK_SIZE

    # divide Q into Tr blocks Q1, ..., QTr of size [Q_BLOCK_SIZE, dim]
    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=0)
    Tr = len(Q_BLOCKS)
    
    # divide KV into Tc blocks K1, ..., KTc of size [KV_BLOCK_SIZE, dim]
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=0)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=0)
    Tc = len(K_BLOCKS)

    # divide O into Tr blocks of size [Q_BLOCK_SIZE, 1]
    # [q_len, dim]    
    O = torch.zeros_like(Q)
    # [q_len, 1]
    l = torch.zeros(Q.shape[:-1])[...,None]
    # [q_len, 1]
    m = torch.ones(Q.shape[:-1])[...,None] * NEG_INF
    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=0))
    l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=0))
    m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=0))

    for i in range(Tr):
        # load qi from global memory to shared memory
        # using async vectorized operations
        Qi = Q_BLOCKS[i]
        # on chip, initialize Oi, li, mi
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BLOCKS[i]
        
        for j in range(Tc):
            # load kj, vj from global memory to shared memory
            # using async vectorized operations
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]

            # On chip, compute Sij = Qi @ KjT, using tensor cores
            S_ij = torch.einsum('q d, k d -> q k', Qi, Kj)

            # on chip, compute mij = rowmax(Sij) [q_len, 1]
            m_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            # on chip, compute mi_new = max(mij, mi) [q_len, 1]
            mi_new = torch.maximum(m_ij, mi)
            
            # on chip, compute Pij = exp(Sij - mij) [q_len, kv_len]
            P_ij = torch.exp(S_ij - mi_new)
            
            # on chip, compute lij = rowsum(Pij) [q_len, 1]
            l_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
            # on chip, compute li_new = exp(mi - mi_new) * li + rowsum(Pij) [q_len, 1]
            li_new = torch.exp(mi - mi_new) * li + l_ij

            # on chip, compute PijVj = Pij @ Vj [q_len, dim]
            P_ij_Vj = torch.einsum('q k, k d -> q d', P_ij, Vj)
            # on chip, compute Oi = exp(mi - mi_new) * Oi + PijVj [q_len, dim]
            Oi_new = torch.exp(mi - mi_new) * Oi + P_ij_Vj
            
            # update 
            Oi = Oi_new
            li = li_new
            mi = mi_new

        # write Oi back to global memory
        O_BLOCKS[i].copy_(Oi)
        l_BLOCKS[i].copy_(li)
        m_BLOCKS[i].copy_(mi)
    return O, l, m

# simulation of flash_attention_2 split-k implementation
# https://pytorch.org/blog/flash-decoding/
def flash_attention_split(Q, K, V):
    # split kv int NUM_SPLITS
    SPLIT_SIZE = K.shape[0] // NUM_SPLITS
    K_BLOCKS = torch.split(K, SPLIT_SIZE, dim=0)
    V_BLOCKS = torch.split(V, SPLIT_SIZE, dim=0)
    
    Ois, lis, mis = [], [], []
    # get O, li, mi for each split
    for i in range(NUM_SPLITS):
        Oi, lii, mii = flash_attention_2(Q, K_BLOCKS[i], V_BLOCKS[i])
        Ois.append(Oi)
        lis.append(lii)
        mis.append(mii)
        
    # merge O, li, mi
    Oi = Ois[0]
    li = lis[0]
    mi = mis[0]
    for i in range(1, NUM_SPLITS):
        mi_new = torch.maximum(mi, mis[i])
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mis[i] - mi_new) * lis[i]
        Oi_new = torch.exp(mi - mi_new) * Oi + torch.exp(mis[i] - mi_new) * Ois[i]
        
        Oi = Oi_new
        li = li_new
        mi = mi_new
    
    # normalize O
    O = Oi / li
    return O
    

if __name__ == "__main__":
    q_len = 4
    kv_len = 16
    dim = 8
    
    Q = torch.randn(q_len, dim)
    K = torch.randn(kv_len * NUM_SPLITS, dim)
    V = torch.randn(kv_len * NUM_SPLITS, dim)

    out_splits = flash_attention_split(Q, K, V)
    ref_out = normal_attention(Q, K, V)
    print(torch.allclose(out_splits, ref_out, atol=1e-5))