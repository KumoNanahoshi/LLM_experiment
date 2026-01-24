import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

@dataclass
class MistralMHCConfig:
    vocab_size: int = 50257
    block_size: int = 64       # Can handle longer contexts now via SWA
    window_size: int = 128      # Mistral SWA Window (W)
    
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    bias: bool = False
    
    # mHC parameters
    n_stream: int = 4
    sinkhorn_iter: int = 20
    hyper_init_scale: float = 0.01
    
    # RoPE
    rope_base: float = 10000.0
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# 2. Math Helpers: RoPE & Sinkhorn
# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, T, D]
    # cos, sin: [T, D] -> Broadcast
    # Note: In inference with Rolling Cache, T=1, but cos/sin must correspond to absolute pos
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Rotate Half: [-x2, x1]
    q_rot = torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1)
    k_rot = torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1)
    
    q_embed = (q * cos) + (q_rot * sin)
    k_embed = (k * cos) + (k_rot * sin)
    return q_embed, k_embed

def sinkhorn_knopp(log_alpha, n_iters=20):
    P = torch.exp(log_alpha - log_alpha.max(dim=-1, keepdim=True)[0])
    for _ in range(n_iters):
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)
    return P

# -----------------------------------------------------------------------------
# 3. Rolling Buffer Cache (The Mistral Magic)
# -----------------------------------------------------------------------------

class RollingKVCache:
    """
    Fixed memory cache that acts as a ring buffer.
    Size: [Batch, n_head, window_size, head_dim]
    """
    def __init__(self, max_batch_size, max_seq_len, n_head, head_dim, window_size, device):
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.k_cache = torch.zeros(max_batch_size, n_head, window_size, head_dim, device=device)
        self.v_cache = torch.zeros(max_batch_size, n_head, window_size, head_dim, device=device)
        self.current_pos = 0 # Absolute position tracker
    
    def update(self, k, v, pos):
        """
        k, v: [B, H, 1, D] - Current step keys/values
        pos: current absolute position integer
        """
        # Calculate Ring Buffer Index: t % W
        idx = pos % self.window_size
        
        # Overwrite the slot
        self.k_cache[:, :, idx, :] = k.squeeze(2)
        self.v_cache[:, :, idx, :] = v.squeeze(2)
        
        self.current_pos = pos
        
    def get_view(self, pos):
        """
        Returns the valid keys and values for attention.
        For simplicity in PyTorch, we roll the buffer so the order is correct chronologically.
        Mistral CUDA kernels do this without rolling, but here we need contiguous tensors for matmul.
        """
        if pos < self.window_size:
            # Not full yet, return [:pos+1]
            return self.k_cache[:, :, :pos+1, :], self.v_cache[:, :, :pos+1, :]
        else:
            # Full buffer. We need to reorder it: [Older ... Newer]
            # Current tip is at idx = pos % W. 
            # The oldest item is at (idx + 1) % W.
            idx = pos % self.window_size
            k_rolled = torch.roll(self.k_cache, shifts=-(idx+1), dims=2)
            v_rolled = torch.roll(self.v_cache, shifts=-(idx+1), dims=2)
            return k_rolled, v_rolled

# -----------------------------------------------------------------------------
# 4. Neural Components
# -----------------------------------------------------------------------------

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SWAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.window_size = config.window_size
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, freqs_cos, freqs_sin, kv_cache: Optional[RollingKVCache] = None, current_pos=None):
        B, T, C = x.size()
        
        # 1. QKV Projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # [B, H, T, D]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 2. Apply RoPE (Absolute Position Logic)
        if kv_cache is not None:
            # Inference: T=1, current_pos is scalar
            # Get specific frequency for this single position
            cos = freqs_cos[current_pos : current_pos+1] 
            sin = freqs_sin[current_pos : current_pos+1]
        else:
            # Training: T is block_size
            cos, sin = freqs_cos[:T], freqs_sin[:T]
            
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Cache Management or Full Sequence
        if kv_cache is not None:
            # --- Inference with Rolling Buffer ---
            # Update cache at ring buffer index
            kv_cache.update(k, v, current_pos)
            
            # Retrieve historical keys/values (correctly ordered for attention)
            k_hist, v_hist = kv_cache.get_view(current_pos)
            
            # Attend: q is [B, H, 1, D], k_hist is [B, H, <=W, D]
            # No mask needed usually for inference as we only see past, 
            # providing cache is strictly causal.
            y = F.scaled_dot_product_attention(q, k_hist, v_hist, is_causal=False)
            
        else:
            # --- Training with Sliding Window Mask ---
            if self.flash:
                # Flash Attention 2 supports sliding window natively, but PyTorch 
                # generic SDPA relies on masking for broad compatibility.
                # Let's construct the Band Mask manually for safety.
                
                # Shape: [T, T]
                # Lower triangular AND Upper triangular shifted by window
                # mask[i, j] = 1 if i >= j AND i - j < W
                ones = torch.ones(T, T, device=x.device)
                mask = torch.tril(ones) # i >= j
                mask = torch.triu(mask, diagonal=-self.window_size + 1) # i - j < W
                
                # Expand mask for batch/heads: [1, 1, T, T]
                attn_mask = mask.unsqueeze(0).unsqueeze(0)
                
                # Use masked attention
                # Note: We pass is_causal=False because we manually provided the causal SWA mask
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                # Manual Fallback
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                mask = torch.tril(torch.ones(T, T, device=x.device))
                mask = torch.triu(mask, diagonal=-self.window_size + 1)
                att = att.masked_fill(mask == 0, float('-inf'))
                y = F.softmax(att, dim=-1) @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    def forward(self, x):
        return self.c_proj(new_gelu(self.c_fc(x)))

# -----------------------------------------------------------------------------
# 5. mHC Layer (Modified to pass Cache)
# -----------------------------------------------------------------------------

class MHCLayer(nn.Module):
    def __init__(self, config, block_type='attn'):
        super().__init__()
        self.n_stream = config.n_stream
        self.config = config
        self.block_type = block_type
        
        if block_type == 'attn':
            self.block = SWAAttention(config)
            self.ln = LayerNorm(config.n_embd, bias=config.bias)
        else:
            self.block = MLP(config)
            self.ln = LayerNorm(config.n_embd, bias=config.bias)
            
        # mHC Learnable Mappings
        self.alpha_pre = nn.Parameter(torch.tensor(config.hyper_init_scale))
        self.theta_pre = nn.Linear(config.n_embd, config.n_stream, bias=False)
        self.bias_pre  = nn.Parameter(torch.zeros(1, 1, 1, config.n_stream))

        self.alpha_post = nn.Parameter(torch.tensor(config.hyper_init_scale))
        self.theta_post = nn.Linear(config.n_embd, config.n_stream, bias=False)
        self.bias_post  = nn.Parameter(torch.zeros(1, 1, 1, config.n_stream))

        self.alpha_res = nn.Parameter(torch.tensor(config.hyper_init_scale))
        self.theta_res = nn.Linear(config.n_embd, config.n_stream * config.n_stream, bias=False)
        self.bias_res  = nn.Parameter(torch.zeros(1, 1, config.n_stream, config.n_stream))

    def forward(self, x_streams, freqs_cos, freqs_sin, kv_cache=None, current_pos=None):
        B, T, n, C = x_streams.shape
        x_summary = x_streams.mean(dim=2) 
        
        # 1. Dynamic Coefficients
        pre_logits = self.alpha_pre * self.theta_pre(x_summary).view(B, T, 1, n) + self.bias_pre
        H_pre = torch.sigmoid(pre_logits)
        
        post_logits = self.alpha_post * self.theta_post(x_summary).view(B, T, n, 1) + self.bias_post.transpose(-1, -2)
        H_post = 2.0 * torch.sigmoid(post_logits)
        
        res_logits = self.alpha_res * self.theta_res(x_summary).view(B, T, n, n) + self.bias_res
        H_res = sinkhorn_knopp(res_logits, n_iters=self.config.sinkhorn_iter)
        
        # 2. Block Input
        x_in = torch.matmul(H_pre, x_streams).squeeze(2) # Flatten streams
        x_in = self.ln(x_in)
        
        # 3. Block Execution (Passing Cache info)
        if self.block_type == 'attn':
            f_out = self.block(x_in, freqs_cos, freqs_sin, kv_cache, current_pos)
        else:
            f_out = self.block(x_in)
            
        # 4. Stream Update
        term_mix = torch.matmul(H_res, x_streams)
        term_new = H_post @ f_out.unsqueeze(2)
        return term_mix + term_new

class MistralMHCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            self.layers.append(MHCLayer(config, block_type='attn'))
            self.layers.append(MHCLayer(config, block_type='mlp'))
            
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        
        # RoPE Cache (Max length possible)
        max_len = config.block_size
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_freqs_cis(head_dim, max_len, config.rope_base)
        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches: List[RollingKVCache] = None, current_pos=0):
        B, T = idx.size()
        
        x = self.wte(idx)
        # Replicate for streams: [B, T, n, C]
        x_streams = x.unsqueeze(2).expand(-1, -1, self.config.n_stream, -1).clone()
        
        # Pass through layers
        cache_idx = 0
        for layer in self.layers:
            if layer.block_type == 'attn':
                # Pass specific cache for this layer
                cache = kv_caches[cache_idx] if kv_caches else None
                x_streams = layer(x_streams, self.freqs_cos, self.freqs_sin, cache, current_pos)
                cache_idx += 1
            else:
                x_streams = layer(x_streams, self.freqs_cos, self.freqs_sin)
                
        x_final = x_streams.mean(dim=2)
        x_final = self.ln_f(x_final)
        
        if targets is not None:
            logits = self.lm_head(x_final)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x_final[:, [-1], :])
            loss = None
            
        return logits, loss

# -----------------------------------------------------------------------------
# 6. Training & Inference Setup
# -----------------------------------------------------------------------------

class TinyStoriesIterableDataset(IterableDataset):
    def __init__(self, tokenizer, max_length, limit=None):
        self.dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.limit = limit
    def __iter__(self):
        for sample in self.dataset:
            enc = self.tokenizer(sample['text'], truncation=True, max_length=self.max_length+1, return_tensors="pt")
            ids = enc['input_ids'][0]
            if len(ids) < self.max_length + 1: continue
            yield ids[:-1], ids[1:]

def main():
    torch.manual_seed(42)
    # Config: Large Block Size (Context), Small Window (SWA)
    config = MistralMHCConfig(
        block_size=512, 
        window_size=128, # Sliding Window size
        n_layer=4, 
        n_embd=256,
        n_stream=4
    )
    
    print(f"Arch: mHC (n={config.n_stream}) + SWA (w={config.window_size}) + Rolling Cache")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = MistralMHCTransformer(config).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.1)
    
    # -- 1. Training Loop (Using SWA Mask) --
    dataset = TinyStoriesIterableDataset(tokenizer, config.block_size)
    dataloader = DataLoader(dataset, batch_size=12, num_workers=0)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == 'cuda'))
    
    model.train()
    print("\n--- Training (SWA Masked) ---")
    t0 = time.time()
    for i, (X, Y) in enumerate(dataloader):
        X, Y = X.to(config.device), Y.to(config.device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss = model(X, Y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        if i % 10 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")
            # -- 2. Inference Loop (Using Rolling Buffer Cache) --
            print("\n--- Inference (Rolling Buffer Cache) ---")
            model.eval()

            prompt = "Once upon a time"
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
            B = tokens.size(0)

            # Initialize Rolling Caches (One per Attention Layer)
            # Note: Only for Attn layers (layers 0, 2, 4, 6 in our alternating list)
            kv_caches = []
            attn_layer_count = config.n_layer # Since we add pair (Attn, MLP)
            head_dim = config.n_embd // config.n_head

            for _ in range(attn_layer_count):
                kv_caches.append(RollingKVCache(B, config.block_size, config.n_head, head_dim, config.window_size, config.device))

            # Prefill (Processing prompt) - In this demo we process token by token for simplicity
            # Ideally, you process prompt in one go (with mask) then switch to token-by-token
            # Here we simulate token-by-token generation from start to show Rolling Cache

            print(f"Generating from: '{prompt}'")
            curr_ids = tokens

            with torch.no_grad():
                # First, process prompt tokens one by one to fill cache (Simulation)
                # Real prefill is faster but requires handling cache.update differently
                for pos in range(curr_ids.size(1) - 1):
                    x_in = curr_ids[:, pos:pos+1]
                    model(x_in, kv_caches=kv_caches, current_pos=pos)

                # Now generate new tokens
                next_token = curr_ids[:, -1:]
                current_pos = curr_ids.size(1) - 1

                for _ in range(200): # Generate 200 tokens
                    logits, _ = model(next_token, kv_caches=kv_caches, current_pos=current_pos)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, 1)

                    curr_ids = torch.cat((curr_ids, next_id), dim=1)
                    next_token = next_id
                    current_pos += 1

            print(tokenizer.decode(curr_ids[0].tolist()))
        if i >= 2000: break # Demo stop

    

if __name__ == "__main__":
    main()