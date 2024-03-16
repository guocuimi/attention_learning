#!/usr/bin/env python3

import torch
import torch.distributed as dist
import time
from utils import State, ref_attention
from flash_attn_2_split import flash_attention_2

# ring communication class
# all devices are connected in a ring topology
class Ring:
    def __init__(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # send kv to the next rank and receive kv from the previous rank
        self.send_rank = (rank + 1) % world_size
        self.recv_rank = (rank - 1) % world_size
        
    # send kv to the next rank and receive kv from the previous rank
    def send_recv_kv(self, k_send, v_send, k_recv, v_recv):
        send_k_op = dist.P2POp(dist.isend, k_send, self.send_rank)
        send_v_op = dist.P2POp(dist.isend, v_send, self.send_rank)
        recv_k_op = dist.P2POp(dist.irecv, k_recv, self.recv_rank)
        recv_v_op = dist.P2POp(dist.irecv, v_recv, self.recv_rank)
        self.reqs = dist.batch_isend_irecv([send_k_op, send_v_op, recv_k_op, recv_v_op])
    
    # wait for all the ops to complete
    def wait(self):
        for req in self.reqs:
            req.wait()
        self.reqs = None
    

# pytorch implementation of ring attention
# https://arxiv.org/abs/2310.01889
# https://arxiv.org/abs/2305.19370
def ring_attention(Q, K, V):
    # initialize the ring communication instance
    ring = Ring()
    
    # steps equal to the number of devices/processes
    steps = dist.get_world_size()
    
    k, v = K, V
    for step in range(steps):
        # allocate memory for the next k and v
        next_k = torch.empty_like(k)
        next_v = torch.empty_like(v)
        
        # send and receive kv asynchronously
        # to overlap communication and computation
        if step != steps - 1:
            ring.send_recv_kv(k, v, next_k, next_v)
        
        # local attention computation
        O, lse = flash_attention_2(Q, k, v)
        
        # merge the local states
        if step == 0:
            state = State(O, lse, 1)
        else:
            state.merge(O, lse, 1)
        
        # wait for the send and receive operations to complete
        if step != steps - 1:
            ring.wait()

        # update k and v for the next step
        k, v = next_k, next_v
        
    # normalize the attention scores
    return state.normalize()

# torchrun --nproc_per_node 4 ./ring_attn.py
if __name__ == "__main__":    
    # initialize the process group for ring communication
    dist.init_process_group(backend="gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # set up the same seed for all processes
    # in order to generate same input across all processes
    torch.manual_seed(1)
    
    q_len = 4
    kv_len = 16
    dim = 8
    
    # global input
    Q = torch.randn(q_len * world_size, dim)
    K = torch.randn(kv_len * world_size, dim)
    V = torch.randn(kv_len * world_size, dim)
    
    # get the local input for the current process
    # both Q, K, V are shared across all processes 
    local_q = Q.chunk(world_size, dim=0)[rank]
    local_k = K.chunk(world_size, dim=0)[rank]
    local_v = V.chunk(world_size, dim=0)[rank]
    
    # compute the attention scores using ring attention
    local_out = ring_attention(local_q, local_k, local_v)

    # compute the global attention scores using reference attention
    ref_out = ref_attention(Q, K, V)
    # get the local ref attention out
    ref_local_out = ref_out.chunk(world_size, dim=0)[rank]
    
    # compare the local attention scores
    equal = torch.allclose(local_out, ref_local_out, atol=1e-5)
    
    # sleep to avoid overlapping print statements
    time.sleep(rank * 0.1)
    print(f"rank={rank}: {equal}")