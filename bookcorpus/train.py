import time
import torch
import tiktoken
import numpy as np
from config import ModelConfig, TrainConfig
from model import GPT
from optim import configure_optimizers
from dataset import BinaryDataset
from generation import generate
from checkpoint import CheckpointManager # <--- Import

def train_loop():
    # 1. Setup
    m_cfg = ModelConfig(dim=512, n_layers=12, hyper_type="vector")
    t_cfg = TrainConfig(batch_size=1, grad_accum_steps=64) 
    # Use 'latest.pt' by default if resume_path is not set but file exists?
    # For now, let's keep it manual via config for safety.
    
    device = t_cfg.device
    torch.set_float32_matmul_precision('high')
    
    ckpt_manager = CheckpointManager(t_cfg.out_dir)
    
    # 2. Data
    print(f"Loading data from {t_cfg.data_dir}...")
    dataset = BinaryDataset(t_cfg.data_dir, m_cfg.max_seq_len)
    enc = tiktoken.get_encoding("gpt2")
    
    # 3. Model
    model = GPT(m_cfg).to(device)
    # Compile needs to happen AFTER load_state_dict ideally, 
    # or you re-compile. Simplest is to load weights then compile.
    
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 4. Optimizers & Scaler
    opt_muon, opt_adam = configure_optimizers(model, t_cfg.muon_lr, t_cfg.adam_lr, 0.01, device)
    scaler = torch.cuda.amp.GradScaler(enabled=(t_cfg.dtype=="float16"))
    
    # 5. Resume Logic
    start_step = 0
    if t_cfg.resume_path:
        start_step = ckpt_manager.load(
            t_cfg.resume_path, model, opt_muon, opt_adam, scaler, device
        )
        print("Loading existing model...")
    
    # Optional Compile
    if t_cfg.use_compile:
        print("Compiling model...")
        model = torch.compile(model)
        
    # 6. Loop
    step = start_step # Start from loaded step
    t0 = time.time()
    
    model.train()
    
    # Use while loop based on step count
    while step < t_cfg.total_steps:
        # A. Accumulation
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        loss_accum = 0.0
        grad_accum_steps = t_cfg.grad_accum_steps
        batch_size = t_cfg.batch_size
        for _ in range(grad_accum_steps):
            X, Y = dataset.get_batch(batch_size, device)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, loss = model(X, targets=Y)
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            
        # B. Optimization Step
        scaler.unscale_(opt_adam)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(opt_adam)
        scaler.step(opt_muon)
        scaler.update()
        
        step += 1
        
        # C. Logging
        if step % t_cfg.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            # Avoid division by zero on first step
            dt = max(dt, 1e-4)
            tok_sec = (t_cfg.batch_size * t_cfg.grad_accum_steps * m_cfg.max_seq_len) / dt
            print(f"Step {step:05d} | Loss: {loss_accum:.4f} | {tok_sec:.0f} tok/s")
            
        # D. Evaluation / Generation
        if step % t_cfg.eval_every == 0:
            print(f"--- Generating (Step {step}) ---")
            torch.cuda.empty_cache()
            ctx = torch.tensor(enc.encode("Once upon a time "), device=device).unsqueeze(0)
            # Switch to eval mode is handled inside generate, but let's be safe
            out = generate(model, ctx, max_new_tokens=50)
            print(enc.decode(out[0].tolist()))
            print("------------------------------")
            del ctx, out
            torch.cuda.empty_cache()
            t0 = time.time() 
            
        # E. Save Checkpoint
        if step % t_cfg.save_every == 0:
            ckpt_manager.save(step, model, opt_muon, opt_adam, scaler, loss_accum)

if __name__ == "__main__":
    train_loop()