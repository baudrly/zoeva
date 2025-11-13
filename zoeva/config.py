from dataclasses import dataclass
from typing import Optional, Dict
import os, yaml

@dataclass
class EmbedConfig:
    clop_repo: str
    input_file: str
    input_type: str = "fasta"   # fasta | gff3 | bed
    reference_fasta: Optional[str] = None
    output_dir: str = "run/embeddings"
    kmer_k: int = 6
    epochs: int = 5
    batch_size: int = 64

@dataclass
class HypothesisConfig:
    embeddings_csv: str
    out_dir: str = "run/hypotheses"
    use_faiss: bool = True
    n_neighbors: int = 50
    zero_shot_prompts: Optional[str] = None
    text_encoder_onnx: Optional[str] = None
    text_vocab_json: Optional[str] = None
    alpha_neighbor: float = 0.6
    beta_zeroshot: float = 0.4

@dataclass
class ValidationConfig:
    fasta: str
    out_dir: str = "run/validation"
    threads: int = 8
    blast_db: Optional[str] = None
    diamond_db: Optional[str] = None
    pfam_hmm: Optional[str] = None
    kofam_db: Optional[str] = None
    eggnog_db: Optional[str] = None

@dataclass
class ScoreConfig:
    hypotheses_parquet: str
    validation_parquet: Optional[str] = None
    graph_parquet: Optional[str] = None
    out_dir: str = "run/report"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
