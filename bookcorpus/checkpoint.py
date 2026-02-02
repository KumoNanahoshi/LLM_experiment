import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
from jaxtyping import Int

class CheckpointManager:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save(
        self, 
        step: int, 
        model: nn.Module, 
        opt_muon: optim.Optimizer, 
        opt_adam: optim.Optimizer, 
        scaler: torch.cuda.amp.GradScaler,
        loss: float
    ):
        """Saves a comprehensive checkpoint."""
        state = {
            "step": step,
            "model_state": model.state_dict(),
            "opt_muon_state": opt_muon.state_dict(),
            "opt_adam_state": opt_adam.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loss": loss
        }
        
        # Save numbered checkpoint
        path = os.path.join(self.out_dir, f"step_{step:06d}.pt")
        torch.save(state, path)
        
        # Update 'latest' pointer (optional but convenient)
        latest_path = os.path.join(self.out_dir, "latest.pt")
        torch.save(state, latest_path)
        print(f"--> Saved checkpoint to {path}")

    def load(
        self, 
        path: str, 
        model: nn.Module, 
        opt_muon: optim.Optimizer, 
        opt_adam: optim.Optimizer, 
        scaler: torch.cuda.amp.GradScaler,
        device: str
    ) -> int:
        """
        Loads state into objects in-place. Returns the step to resume from.
        """
        if not os.path.exists(path):
            print(f"!! Checkpoint path not found: {path}. Starting from scratch.")
            return 0
            
        print(f"--> Loading checkpoint from {path}...")
        # Map location is crucial to avoid loading directly to GPU 0 if configured otherwise
        checkpoint = torch.load(path, map_location=device)
        
        # Load Model
        # strict=False allows loading even if we added small buffers (like RoPE cache changes)
        keys = model.load_state_dict(checkpoint["model_state"], strict=False) 
        if keys.missing_keys: print(f"Warning: Missing keys: {keys.missing_keys}")
        
        # Load Optimizers
        opt_muon.load_state_dict(checkpoint["opt_muon_state"])
        opt_adam.load_state_dict(checkpoint["opt_adam_state"])
        
        # Load Scaler
        scaler.load_state_dict(checkpoint["scaler_state"])
        
        step = checkpoint["step"]
        loss = checkpoint.get("loss", "N/A")
        
        print(f"--> Resumed from step {step} (Last Loss: {loss})")
        return step