import math
import time
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# Import transformers for tokenizer and datasets for streaming
from transformers import AutoTokenizer
from datasets import load_dataset


@dataclass
class HyperConfig:
    vocab_size: int = 50257 # GPT-2 default
    block_size: int = 256   # Reduced context length for < 5GB VRAM
    n_layer: int = 6        # Shallow network
    n_head: int = 8
    n_embd: int = 384       # Small embedding dim
    dropout: float = 0.0
    bias: bool = False      # False: a bit better for modern architectures
    # Hyper-connection specific: Initial weight for the residual branch
    hyper_alpha_init: float = 0.01 
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# Functional helper for easy debugging and clarity
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# -----------------------------------------------------------------------------
# 2. The Muon Optimizer (Experimental / Advanced)
# -----------------------------------------------------------------------------
# Muon is a momentum-updated orthogonal optimizer. 
# It works best on 2D tensors (weights), while AdamW handles 1D (biases, layernorms).

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix G.
    Used to orthogonalize the update in Muon.

    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315) # Magic just as the original code !
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top spectral norm < 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalizer.
    This is a simplified version suitable for training transformers.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    continue 
                g = p.grad

                original_shape = p.shape
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if p.ndim > 2:
                    # such as Conv2d: [Out, In, K, K] -> [Out, In*K*K]
                    g_flat = g.view(original_shape[0], -1)
                else:
                    g_flat = g
                # Orthogonalize update

                g_flat = zeropower_via_newtonschulz5(g_flat, steps=ns_steps)
                g =  g_flat.view(original_shape)
                p.data.add_(g, alpha=-lr)

# -----------------------------------------------------------------------------
# 3. Model Components (Functional Style)
# -----------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support bias=False nicely. """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Flash attention is standard now
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels
        # Calculate q, k, v
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal Attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            # Fallback (Manual implementation) - omitting for brevity/memory safety
            raise ImportError("Flash Attention required for memory efficiency.")
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return x

class HyperBlock(nn.Module):
    """
    Implements a block with 'Hyper-connections'.
    Instead of x + f(x), we use:
    x_out = alpha * x_in + beta * f(norm(x_in))
    where alpha and beta are learnable scalars (or vectors).
    This allows the network to dynamically route signal strength.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Hyper-connection parameters (Learnable Skip Weights)
        # Initialized to act close to standard residual (1.0) but flexible
        self.alpha_attn = nn.Parameter(torch.tensor(1.0))
        self.beta_attn  = nn.Parameter(torch.tensor(config.hyper_alpha_init))
        
        self.alpha_mlp = nn.Parameter(torch.tensor(1.0))
        self.beta_mlp  = nn.Parameter(torch.tensor(config.hyper_alpha_init))

    def forward(self, x):
        # Attention Sub-layer with Hyper-connection
        # x = alpha * x + beta * Attention(Norm(x))
        x = self.alpha_attn * x + self.beta_attn * self.attn(self.ln_1(x))
        
        # MLP Sub-layer with Hyper-connection
        # x = alpha * x + beta * MLP(Norm(x))
        x = self.alpha_mlp * x + self.beta_mlp * self.mlp(self.ln_2(x))
        return x

class HyperTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([HyperBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward pass (Functional flow)
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
        # Apply HyperBlocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # Inference optimization
            loss = None
            
        return logits, loss

# -----------------------------------------------------------------------------
# 4. Data Pipeline (Streaming for Low RAM)
# -----------------------------------------------------------------------------

class TinyStoriesIterableDataset(IterableDataset):
    def __init__(self, tokenizer, max_length, limit=None):
        # Stream dataset so we don't download 4GB+ to disk at once
        self.dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.limit = limit

    def __iter__(self):
        count = 0
        for sample in self.dataset:
            text = sample['text']
            # Basic tokenization
            enc = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_length + 1, # +1 for target shifting
                return_tensors="pt"
            )
            ids = enc['input_ids'][0]
            if len(ids) < self.max_length + 1:
                continue # Skip short samples
                
            x = ids[:-1]
            y = ids[1:]
            
            yield x, y
            count += 1
            if self.limit and count >= self.limit:
                break

# -----------------------------------------------------------------------------
# 5. Training Loop (Optimized for 4.9GB VRAM)
# -----------------------------------------------------------------------------

def train():
    # Setup
    torch.manual_seed(1332)
    config = HyperConfig()
    print(f"Model Config: {config}")
    
    # 1. Initialize Tokenizer (GPT2 is compatible with TinyStories)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Initialize Model
    model = HyperTransformer(config)
    model.to(config.device)
    
    # Compile model 
   # model = torch.compile(model)  # torch.compile is not supported on Python 3.14+"

    
    USE_MUON = True # 
    
    if USE_MUON:
        # Separate 2D params (Linear weights) for Muon, 1D (Bias/LN) for AdamW
        muon_params = [p for n, p in model.named_parameters() if p.ndim == 2]
        adam_params = [p for n, p in model.named_parameters() if p.ndim < 2] # biases, layernorms, embeddings
        
        optimizers = [
            Muon(muon_params, lr=0.02, momentum=0.95),
            torch.optim.AdamW(adam_params, lr=3e-4, weight_decay=0.1)
        ]
    else:
        # Standard AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-1)
    
    # 4. DataLoader
    dataset = TinyStoriesIterableDataset(tokenizer, config.block_size)
    # Num_workers=0 is safer for streaming datasets in simple scripts
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0) 

    # 5. Loop with AMP
    scaler = torch.amp.GradScaler('cuda',enabled=(config.device == 'cuda'))
    model.train()
    
    max_steps = 1000 # Demo purpose
    accumulation_steps = 4 # Simulate batch size of 16 * 4 = 64
    
    print("Starting training...")
    t0 = time.time()
    
    step = 0
    if USE_MUON:
            # Manual stepping for mixed optimizers
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
    else:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    for i, (X, Y) in enumerate(dataloader):
        X, Y = X.to(config.device), Y.to(config.device)
        
        # Mixed Precision Context
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits, loss = model(X, Y)
            loss = loss / accumulation_steps # Scale loss

        # Backward
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            # Clip Gradients
            scaler.unscale_(optimizer) if not USE_MUON else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step
            if USE_MUON:
                # Manual stepping for mixed optimizers
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            step += 1
            if step % 10 == 0:
                dt = time.time() - t0
                # Restore loss for printing
                raw_loss = loss.item() * accumulation_steps
                print(f"Step {step} | Loss: {raw_loss:.4f} | Time: {dt*1000/10:.2f}ms")

                start_text = "Once upon a time"
                idx = tokenizer.encode(start_text, return_tensors="pt").to(config.device)
    
                print(f"\nGenerating from: '{start_text}'")
                with torch.no_grad():
                    for _ in range(50):
                        logits, _ = model(idx)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        next_idx = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, next_idx), dim=1)
            
                print(tokenizer.decode(idx[0].tolist()))
                t0 = time.time()
                
            if step >= max_steps:
                break
                
    print("Training finished.")
    
    # Simple Inference Test
    model.eval()
    start_text = "Once upon a time"
    idx = tokenizer.encode(start_text, return_tensors="pt").to(config.device)
    
    print(f"\nGenerating from: '{start_text}'")
    with torch.no_grad():
        for _ in range(50):
            logits, _ = model(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
            
    print(tokenizer.decode(idx[0].tolist()))

if __name__ == "__main__":
    train()