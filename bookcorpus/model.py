import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List
from jaxtyping import Float, Int
from torch import Tensor

from config import ModelConfig
from rope import RotaryEmbedding
from modules import RMSNorm, HyperConnection

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wqkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=cfg.use_bias)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=cfg.use_bias)
        
    def forward(
        self, 
        x: Tensor, 
        rope: RotaryEmbedding, 
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        
        B, T, C = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        q = rope(q, start_pos)
        k = rope(k, start_pos)
        
        # KV Cache Logic
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            new_kv_cache = (k, v)
        else:
            # If we are generating but cache is None (first step), return current k, v
            # If training, we usually ignore this return value anyway
            new_kv_cache = (k, v)
        
        # Flash Attention
        is_causal = True if kv_cache is None else False
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            is_causal=is_causal
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y), new_kv_cache

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, 4 * cfg.dim, bias=cfg.use_bias)
        self.w2 = nn.Linear(4 * cfg.dim, cfg.dim, bias=cfg.use_bias)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)))

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.hc1 = HyperConnection(cfg.dim, cfg.hyper_type)
        
        self.ln2 = RMSNorm(cfg.dim)
        self.mlp = MLP(cfg)
        self.hc2 = HyperConnection(cfg.dim, cfg.hyper_type)

    def forward(
        self, 
        x: Tensor, 
        rope: RotaryEmbedding,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        start_pos: int = 0
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        
        norm_x = self.ln1(x)
        attn_out, new_cache = self.attn(norm_x, rope, kv_cache, start_pos)
        x = self.hc1(x, attn_out)
        
        mlp_out = self.mlp(self.ln2(x))
        x = self.hc2(x, mlp_out)
        
        return x, new_cache

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rope = RotaryEmbedding(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight

    def forward(
        self, 
        idx: Int[Tensor, "B T"], 
        targets: Optional[Int[Tensor, "B T"]] = None,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        start_pos: int = 0
    ):
        B, T = idx.shape
        x = self.token_emb(idx)
        
        new_caches = []
        
        for i, block in enumerate(self.blocks):
            # Safe access to cache
            layer_cache = kv_caches[i] if kv_caches is not None else None
            
            if self.training and self.config.use_checkpointing:
                # During training, we don't care about caching
                x, _ = checkpoint(block, x, self.rope, None, 0, use_reentrant=False)
            else:
                x, new_c = block(x, self.rope, layer_cache, start_pos)
                # FIX: Always append the new cache status if not training
                if not self.training:
                    new_caches.append(new_c)
                
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        logits = self.head(x[:, [-1], :]) 
        return logits, new_caches