import torch
import torch.nn.functional as F
# Remove relative import if running as script, but keep for module
# from model import GPT 

@torch.no_grad()
def generate(
    model, 
    idx: torch.Tensor, 
    max_new_tokens: int, 
    temperature: float = 0.8, 
    top_k: int = 20
):
    model.eval()
    B, T = idx.shape
    
    # 1. Prefill
    # kv_caches will be populated by the model now
    logits, kv_caches = model(idx, kv_caches=None, start_pos=0)
    
    # 2. Sample first token
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    curr_idx = torch.cat((idx, next_token), dim=1)
    
    # 3. Generation loop
    for i in range(max_new_tokens - 1):
        # We start feeding from T (since we just generated T)
        current_pos = T + i
        
        # next_token is [B, 1]
        logits, kv_caches = model(next_token, kv_caches=kv_caches, start_pos=current_pos)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        curr_idx = torch.cat((curr_idx, next_token), dim=1)
        
    model.train()
    return curr_idx