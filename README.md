# Zoeva: sequence-annotation hypothesis henerator 

This repository provides a reproducible pipeline that builds on
[baudrly/clop](https://github.com/baudrly/clop) to:
1) ingest sequences with crude annotations (species/biotype/description),
2) train or load a joint sequence-annotation embedding using CLOP,
3) perform hypothesis generation (annotation & pathway candidates) via kNN and zero‑shot prompts,
4) validate candidates with traditional tools (BLAST+/DIAMOND, HMMER/Pfam, KOfamScan/eggNOG‑mapper),
5) score and report ranked hypotheses ready for human review,
6) propagate evidence over an annotation graph (sequences-keywords-GO/KEGG/Reactome) using personalized PageRank.

The pipeline is orchestrated by a single CLI (`zoeva`) and an optional Snakemake workflow.

> ⚠️ External tools are optional, auto‑detected at runtime. When installed,
> they are wrapped safely by the `validation/` module.

---

## Quick start

```bash
# 0) (optional) install this package in editable mode
pip install -e .

# 1) Prepare inputs (FASTA or GFF3+reference FASTA)
#    FASTA header convention compatible with CLOP:
#    >id|species=Homo sapiens|biotype=protein_coding|description=Example

# 2) Train or load a CLOP model & export embeddings
zoeva embed   --clop_repo /path/to/clop   --input your_data.fasta --input-type fasta   --out run1/embeddings

# 3) Build an index + generate hypotheses
zoeva hypothesize   --embeddings run1/embeddings/final_embeddings.csv   --out run1/hypotheses

# 4) (optional) Graph-based label propagation
zoeva graph   --hypotheses run1/hypotheses/hypotheses.parquet   --out run1/graph   --kw_to_label ref/kw_to_label.tsv   --label_edges ref/label_edges.tsv

# 5) (optional) Run classical validation
zoeva validate   --fasta your_data.fasta   --out run1/validation

# 6) Score & report
zoeva report   --hypotheses run1/hypotheses/hypotheses.parquet   --validation run1/validation/summary.parquet   --graph      run1/graph/graph_scores.parquet   --out run1/report
```