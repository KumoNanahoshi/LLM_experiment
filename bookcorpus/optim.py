import torch
import torch.optim as optim
from jaxtyping import Float
from torch import Tensor

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to orthogonalize G.
    Inputs: G [N, M] (float32 or bfloat16 recommended)
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    # Internal calculation in bfloat16 or float32 for stability
    # FP16 is often too unstable for the matrix powers
    X = G.bfloat16() if torch.cuda.is_bf16_supported() else G.float()
    
    # Preconditioning
    X /= (X.norm() + 1e-7)
    
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    
    return X.to(G.dtype)

class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                if g.ndim != 2: continue 

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                
                # Nesterov
                update = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                
                # Orthogonalize
                update = zeropower_via_newtonschulz5(update, steps=group['ns_steps'])
                
                # Step
                p.data.add_(update, alpha=-group['lr'])

def configure_optimizers(model, muon_lr, adam_lr, weight_decay, device_type):
    # Strategy:
    # 1. 2D weights (Linear) -> Muon
    # 2. Embeddings, Norms, Biases -> AdamW
    
    muon_params = []
    adam_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        
        # Embeddings and Output Head usually stick to AdamW for better stability
        if p.ndim == 2 and "token_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    # Muon group
    optim1 = Muon(muon_params, lr=muon_lr, momentum=0.95, ns_steps=5)
    
    # Adam group
    optim2 = optim.AdamW(adam_params, lr=adam_lr, weight_decay=weight_decay, betas=(0.90, 0.95))
    
    return optim1, optim2