import os
import glob
import numpy as np
import tiktoken
from tqdm import tqdm
from typing import List
from jaxtyping import Int, UInt16

def save_shard(tokens: List[int], output_dir: str, shard_idx: int) -> None:
    arr = np.array(tokens, dtype=np.uint16)
    save_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.bin")
    with open(save_path, "wb") as f:
        f.write(arr.tobytes())
    print(f"Saved {save_path} ({len(tokens) / 1e6:.2f}M tokens)")

def process_large_corpus(input_files: List[str], output_dir: str, shard_size: int = 100_000_000):
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    
    # We will NOT append EOT after every line.
    # We will let the natural newlines separate sentences.
    
    token_buffer: List[int] = []
    shard_idx = 0
    
    for file_path in input_files:
        print(f"Processing file: {file_path}")
        total_size = os.path.getsize(file_path)
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(file_path)) as pbar:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    
                    # IMPORTANT: Do not strip() if you want to keep formatting.
                    # Or strip() and manually add '\n'. 
                    # BookCorpus lines are usually sentences.
                    text = line.strip() 
                    if not text:
                        continue
                    
                    # Add a newline character between sentences/lines
                    # ' ' + text helps GPT-2 tokenizer handle the start of words better
                    text_with_newline = " " + text + "\n"
                    
                    tokens = enc.encode_ordinary(text_with_newline)
                    token_buffer.extend(tokens)
                    
                    if len(token_buffer) >= shard_size:
                        save_shard(token_buffer[:shard_size], output_dir, shard_idx)
                        token_buffer = token_buffer[shard_size:]
                        shard_idx += 1

    # Only append EOT at the very end of the massive file
    token_buffer.append(enc.eot_token)
    
    if len(token_buffer) > 0:
        save_shard(token_buffer, output_dir, shard_idx)

if __name__ == "__main__":
    # Point to your actual files
    FILES = glob.glob("data_raw/*.txt") 
    process_large_corpus(FILES, "./data_bin")