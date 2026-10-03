import torch
from torch import nn
import torch.nn.functional as F

def create_causal_mask(
    seq_len,
    device,
    dtype
):
    return torch.triu(
        torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=device,
            dtype=dtype
        ),
        diagonal=1
    )

class RotaryPositionalEncoding(nn.Module):
    def __init__(
       self, 
       dim, 
       max_seq_len=1024
    ):
        super().__init__()
        N = 10000
        inv_freq = 1. / (
            N ** (
                torch.arange(
                    0, dim, 2
                ).float() / dim
            )
        )
        position = torch.arange(
            max_seq_len
        ).float()
        inv_freq = torch.cat(
            (
                inv_freq, inv_freq
            ), 
            dim=-1
        )
        sinusoid_inp = torch.outer(
            position, inv_freq
        )
        self.register_buffer(
            "cos", sinusoid_inp.cos()
        )
        self.register_buffer(
            "sin", sinusoid_inp.sin()
        ) 

    def rotate_half(self,x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat(
            (-x2, x1), dim=-1
        )

    def apply_rotary_pos_emb(
        self, x, cos, sin
    ):
        return (
            x * cos
        ) + (
            self.rotate_half(x) * sin
        )
    
    def forward(
       self, 
       x, 
       seq_len=None
    ):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(
            1, seq_len, 1, -1
        )
        sin = self.sin[:seq_len].view(
            1, seq_len, 1, -1
        )
        return self.apply_rotary_pos_emb(
            x, cos, sin
        )


class GQA(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads, 
        num_kv_heads, 
        dropout=0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(
            hidden_dim, 
            self.num_heads * self.head_dim
        )
        self.k_proj = nn.Linear(
            hidden_dim, 
            self.num_kv_heads * self.head_dim
        )
        self.v_proj = nn.Linear(
            hidden_dim, 
            self.num_kv_heads * self.head_dim
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            hidden_dim
        )
 
    def forward(
        self, 
        q, 
        k, 
        v, 
        mask=None, 
        rope=None
    ):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape
 
        # projection
        q = self.q_proj(q).view(
            q_batch_size, 
            q_seq_len, 
            -1, 
            self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(k).view(
            k_batch_size, 
            k_seq_len, 
            -1, 
            self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(v).view(
            v_batch_size, 
            v_seq_len, 
            -1, 
            self.head_dim
        ).transpose(1, 2)
 
        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)
 
        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(
            q, 
            k, 
            v,
            attn_mask=mask,
            dropout_p=self.dropout,
            enable_gqa=True
        )
        output = output.transpose(1, 2).reshape(
            q_batch_size, q_seq_len, hidden_dim
        ).contiguous()
        output = self.out_proj(output)
        return output
  
    
class SwiGLU(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        intermediate_dim
    ):
        super().__init__()
        self.gate = nn.Linear(
            hidden_dim, 
            intermediate_dim
        )
        self.up = nn.Linear(
            hidden_dim, 
            intermediate_dim
        )
        self.down = nn.Linear(
            intermediate_dim, 
            hidden_dim
        )
        self.act = nn.SiLU()
 
    def forward(self, x):
        x = self.act(
            self.gate(x)
        ) * self.up(x)
        x = self.down(x)
        return x
    

class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts,
        top_k=2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Routed experts
        self.experts = nn.ModuleList(
            [
                SwiGLU(
                    hidden_dim,
                    intermediate_dim
                )
                for _ in range(num_experts)
            ]
        )

        # Router for routed experts
        self.router = nn.Linear(
            hidden_dim,
            num_experts
        )

        # Shared expert
        self.shared_expert = SwiGLU(
            hidden_dim,
            intermediate_dim
        )

    def forward(self, hidden_states):

        batch_size, seq_len, hidden_dim = hidden_states.shape

        hidden_states_reshaped = hidden_states.view(
            -1,
            hidden_dim
        )

        # --------------------------------
        # Shared expert
        # --------------------------------

        shared_output = self.shared_expert(
            hidden_states_reshaped
        )

        # --------------------------------
        # Routed experts
        # --------------------------------

        router_logits = self.router(
            hidden_states_reshaped
        )

        top_k_logits, top_k_indices = torch.topk(
            router_logits,
            self.top_k,
            dim=-1
        )

        top_k_probs = F.softmax(
            top_k_logits,
            dim=-1
        )

        routed_output = torch.zeros(
            batch_size * seq_len,
            hidden_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )

        unique_experts = torch.unique(
            top_k_indices
        )

        for i in unique_experts:

            expert_id = int(i)

            mask = (
                top_k_indices == expert_id
            )

            token_mask = mask.any(dim=1)

            expert_input = (
                hidden_states_reshaped[token_mask]
            )

            expert_weight = (
                top_k_probs[mask]
                .unsqueeze(-1)
            )

            expert_output = self.experts[
                expert_id
            ](expert_input)

            routed_output[token_mask] += (
                expert_output * expert_weight
            )

        # --------------------------------
        # Combine
        # --------------------------------

        output = (
            shared_output
            + routed_output
        )

        output = output.view(
            batch_size,
            seq_len,
            hidden_dim
        )

        return output
    

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads, 
        num_kv_heads, 
        moe_experts, 
        moe_topk, 
        dropout=0.1
    ):
        super().__init__()
        self.self_attn = GQA(
            hidden_dim, 
            num_heads, 
            num_kv_heads, 
            dropout
        )
        self.mlp = MoELayer(
            hidden_dim, 
            4 * hidden_dim, 
            moe_experts, 
            moe_topk
        )
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
 
    def forward(
        self, 
        x, 
        mask=None, 
        rope=None
    ):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(
            out, 
            out, 
            out, 
            mask, 
            rope
        )
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x


class TextGenerationModel(nn.Module):
    def __init__(
        self, 
        num_layers, 
        num_heads, 
        num_kv_heads, 
        hidden_dim,
        moe_experts, 
        moe_topk,
        max_seq_len, 
        vocab_size, 
        dropout=0.1
    ):
        super().__init__()
        self.rope = RotaryPositionalEncoding(
            hidden_dim // num_heads, 
            max_seq_len
        )
        self.embedding = nn.Embedding(
            vocab_size, hidden_dim
        )
        self.decoders = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim, 
                    num_heads, 
                    num_kv_heads, 
                    moe_experts, 
                    moe_topk, 
                    dropout
                )
            for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, 
        ids, 
        mask=None
    ):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)