import os, shutil, subprocess
from dataclasses import dataclass
from typing import Optional

@dataclass
class BlastConfig:
    fasta: str
    out_dir: str
    threads: int = 8
    blast_db: Optional[str] = None
    diamond_db: Optional[str] = None

def run_blast_or_diamond(cfg: BlastConfig) -> Optional[str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_tsv = os.path.join(cfg.out_dir, "blast_hits.tsv")
    if cfg.diamond_db and shutil.which("diamond") is not None:
        cmd = ["diamond", "blastx", "-q", cfg.fasta, "-d", cfg.diamond_db, "-o", out_tsv,
               "-f", "6", "qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend","sstart","send","evalue","bitscore",
               "--threads", str(cfg.threads)]
        subprocess.run(cmd, check=True)
        return out_tsv
    if cfg.blast_db and shutil.which("blastn") is not None:
        cmd = ["blastn", "-query", cfg.fasta, "-db", cfg.blast_db, "-outfmt",
               "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
               "-num_threads", str(cfg.threads), "-out", out_tsv]
        subprocess.run(cmd, check=True)
        return out_tsv
    return None
