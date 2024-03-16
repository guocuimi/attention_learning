import torch


class State:
    def __init__(self, O, m, l):
        self.O = O
        self.m = m
        self.l = l
    
    def merge(self, O, m, l):
        m_new = torch.maximum(self.m, m)
        self.l = torch.exp(self.m - m_new) * self.l + torch.exp(m - m_new) * l
        self.O = torch.exp(self.m - m_new) * self.O + torch.exp(m - m_new) * O
        self.m = m_new
    
    def normalize(self):
        self.O.div_(self.l)
        return self.O
    
    def get_lse(self):
        return self.m + self.l.log()
    
    
def ref_attention(Q, K, V):
    # [q_len, dim] @ [kv_len, dim] -> [q_len, kv_len]  
    QKt = torch.einsum('q d, k d -> q k', Q, K)
    attn = torch.nn.functional.softmax(QKt, dim=-1)
    # [q_len, kv_len] @ [kv_len, dim] -> [q_len, dim]
    return attn @ V