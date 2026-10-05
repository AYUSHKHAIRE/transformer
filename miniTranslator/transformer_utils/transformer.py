import torch
from torch import nn
import torch.nn.functional as F


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
    
    
class EncoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads, 
        num_kv_heads=None, 
        dropout=0.1
    ):
        super().__init__()
        self.self_attn = GQA(
            hidden_dim, 
            num_heads, 
            num_kv_heads, 
            dropout
        )
        self.mlp = SwiGLU(
            hidden_dim, 4 * hidden_dim
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
        out = x
        out = self.norm1(x)
        out = self.self_attn(
            out, out, out, 
            mask, rope
        )
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x
    
    
class DecoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads, 
        num_kv_heads=None, 
        dropout=0.1
    ):
        super().__init__()
        self.self_attn = GQA(
            hidden_dim, 
            num_heads, 
            num_kv_heads, 
            dropout
        )
        self.cross_attn = GQA(
            hidden_dim, 
            num_heads, 
            num_kv_heads, 
            dropout
        )
        self.mlp = SwiGLU(
            hidden_dim, 4 * hidden_dim
        )
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim)

    def forward(
        self, 
        x, 
        enc_out, 
        mask=None, 
        rope=None
    ):
        # self-attention sublayer
        out = x
        out = self.norm1(out)
        out = self.self_attn(
            out, out, out, mask, rope
        )
        x = out + x
        # cross-attention sublayer
        out = self.norm2(x)
        out = self.cross_attn(
            out, enc_out, enc_out, None, rope
        )
        x = out + x
        # MLP sublayer
        x = out + x
        out = self.norm3(x)
        out = self.mlp(out)
        return out + x


class Transformer(nn.Module):
    def __init__(
        self, 
        num_layers, 
        num_heads, 
        num_kv_heads, 
        hidden_dim,
        max_seq_len, 
        vocab_size_src, 
        vocab_size_tgt, 
        dropout=0.1
    ):
        super().__init__()
        self.rope = RotaryPositionalEncoding(
            hidden_dim // num_heads, 
            max_seq_len
        )
        self.src_embedding = nn.Embedding(
            vocab_size_src, 
            hidden_dim
        )
        self.tgt_embedding = nn.Embedding(
            vocab_size_tgt, 
            hidden_dim
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim, 
                    num_heads, 
                    num_kv_heads, 
                    dropout
                ) for _ in range(num_layers)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim, 
                    num_heads, 
                    num_kv_heads, 
                    dropout
                ) for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(
            hidden_dim, 
            vocab_size_tgt
        )

    def forward(
        self, 
        src_ids, 
        tgt_ids, 
        src_mask=None, 
        tgt_mask=None
    ):
        # Encoder
        x = self.src_embedding(src_ids)
        for encoder in self.encoders:
            x = encoder(
                x, src_mask, self.rope
            )
        enc_out = x
        # Decoder
        x = self.tgt_embedding(tgt_ids)
        for decoder in self.decoders:
            x = decoder(
                x, enc_out, tgt_mask, self.rope
            )
        return self.out(x)
    
    
def create_causal_mask(
    seq_len, device
):
    return torch.triu(
        torch.full(
            (seq_len, seq_len), 
            float('-inf'), 
            device=device
        ), 
        diagonal=1
    )
    

def create_padding_mask(
    batch, 
    padding_token_id
):
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(
        batch, device=device
    ).float().masked_fill(
        batch == padding_token_id, float('-inf')
    )
    mask = torch.zeros(
        batch_size, 
        seq_len, 
        seq_len, 
        device=device
    ) + padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]