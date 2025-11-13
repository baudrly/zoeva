from __future__ import annotations
import subprocess, sys, os, pandas as pd
from typing import List
from .config import EmbedConfig, ensure_dir

def run_clop_export(cfg: EmbedConfig) -> str:
    """Call CLOP's CLI to train/export and return path to final_embeddings.csv or .parquet."""
    ensure_dir(cfg.output_dir)
    clop_py = os.path.join(cfg.clop_repo, "clop.py")
    if not os.path.exists(clop_py):
        raise FileNotFoundError(f"Could not find clop.py in {cfg.clop_repo}. Clone https://github.com/baudrly/clop .")
    cmd = [
        sys.executable, clop_py,
        "--input_file", cfg.input_file,
        "--input_type", cfg.input_type,
        "--output_dir", cfg.output_dir,
        "--dna_tokenizer_type", "kmer",
        "--kmer_k", str(cfg.kmer_k),
        "--epochs", str(cfg.epochs),
        "--batch_size", str(cfg.batch_size)
    ]
    if cfg.input_type in {"gff3", "bed"} and cfg.reference_fasta:
        cmd += ["--reference_fasta", cfg.reference_fasta]
    print("[CLOP]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # find export
    for cand in ("final_embeddings.csv", "final_embeddings.parquet"):
        out = os.path.join(cfg.output_dir, cand)
        if os.path.exists(out): return out
    raise FileNotFoundError("CLOP export not found (final_embeddings.csv|parquet)." )
