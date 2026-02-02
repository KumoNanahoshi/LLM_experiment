from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class ModelConfig:
    vocab_size: int = 50304  # GPT-2 vocab aligned to 128
    dim: int = 512           # Small efficient dim
    n_layers: int = 12       # Decent depth
    n_heads: int = 8         # 512 / 8 = 64 head_dim
    max_seq_len: int = 1024
    dropout: float = 0.0     # Disable dropout for pretraining usually
    use_bias: bool = False   # Bias-free is more stable for Muon
    use_checkpointing: bool = False # Gradient checkpointing
    # Advanced Arch
    hyper_type: Literal["scalar", "vector"] = "vector" # LayerScale logic
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    experiment_name: str = "bookcorpus_run"
    data_dir: str = "./data_bin"
    out_dir: str = "./checkpoints"
    
    # Checkpointing (New)
    resume_path: Optional[str] = "./checkpoints/step_004000.pt" # or None
    
    # Optimization
    muon_lr: float = 0.02
    adam_lr: float = 6e-4
    batch_size: int = 1
    grad_accum_steps: int = 64
    total_steps: int = 100000 # Extended for full training
    
    # System
    device: str = "cuda"
    dtype: str = "float16"
    use_compile: bool = False
    
    
    # Monitoring
    log_every: int = 10
    save_every: int = 2000
    eval_every: int = 50

