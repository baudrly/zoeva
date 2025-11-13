import os, pandas as pd

def summarize_validation(out_dir: str) -> str:
    rows = []
    blast = os.path.join(out_dir, "blast_hits.tsv")
    if os.path.exists(blast):
        df = pd.read_csv(blast, sep='\t', header=None)
        df.columns = ["qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend","sstart","send","evalue","bitscore"]
        rows.append({"source":"blast", "n_hits": int(df.shape[0]), "median_bitscore": float(df.bitscore.median()), "best_evalue": float(df.evalue.min())})
    domtbl = os.path.join(out_dir, "pfam.domtblout")
    if os.path.exists(domtbl):
        n_lines = sum(1 for ln in open(domtbl, 'r', encoding='utf-8') if not ln.startswith('#'))
        rows.append({"source":"pfam", "n_hits": int(n_lines)})
    kofam = os.path.join(out_dir, "kofam.tsv")
    if os.path.exists(kofam):
        n_lines = sum(1 for _ in open(kofam, 'r', encoding='utf-8'))
        rows.append({"source":"kofam", "n_hits": int(n_lines)})
    egg = os.path.join(out_dir, "eggnog.tsv")
    if os.path.exists(egg):
        n_lines = sum(1 for _ in open(egg, 'r', encoding='utf-8'))
        rows.append({"source":"eggnog", "n_hits": int(n_lines)})
    out = os.path.join(out_dir, "summary.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out
