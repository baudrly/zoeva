from __future__ import annotations
import os, json, numpy as np, pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Iterable
from collections import Counter
from .config import HypothesisConfig, ensure_dir

@dataclass
class Candidate:
    label: str
    score: float
    source: str  # 'neighbor' or 'zeroshot'

def load_embeddings_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"): df = pd.read_parquet(path)
    else: df = pd.read_csv(path)
    if "id" not in df.columns:
        for cand in ["seq_id","sequence_id","name"]:
            if cand in df.columns: df = df.rename(columns={cand: "id"}); break
    return df

def infer_matrix(df: pd.DataFrame) -> np.ndarray:
    if "embedding" in df.columns:
        return np.vstack(df["embedding"].apply(lambda x: np.array(json.loads(x), dtype=float)))
    numeric = [c for c in df.columns if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])]
    blacklist = {"id","species","biotype","length","gc","entropy"}
    numeric = [c for c in numeric if c not in blacklist]
    return df[numeric].to_numpy(dtype=float)

def _majority(labels: List[str], weights: Optional[np.ndarray], topk: int = 5) -> List[Candidate]:
    if weights is None: weights = np.ones(len(labels))
    w = Counter()
    for lab, ww in zip(labels, weights):
        if lab: w[lab] += float(ww)
    items = w.most_common(topk)
    if not items: return []
    vals = np.array([v for _,v in items], dtype=float)
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    return [Candidate(label=k, score=float(s), source="neighbor") for (k,_),s in zip(items, vals)]

def _keywordize(descs: Iterable[str], topk: int = 20) -> List[Candidate]:
    rx = __import__('re').compile(r"[A-Za-z][A-Za-z_\-]{3,}")
    ctr = Counter()
    for d in descs:
        if not d: continue
        for m in rx.findall(str(d).lower()):
            if m in {"species","biotype","unknown","uncharacterized","hypothetical"}: continue
            ctr[m]+=1
    kws = [k for k,_ in ctr.most_common(topk)]
    return [Candidate(label=k, score=float((i+1)/topk), source="neighbor") for i,k in enumerate(kws)]

def generate_hypotheses(cfg: HypothesisConfig) -> str:
    ensure_dir(cfg.out_dir)
    df = load_embeddings_table(cfg.embeddings_csv)
    X = infer_matrix(df)
    from .index import build_index, query_index
    idx = build_index(X, use_faiss=cfg.use_faiss)
    rows = []
    for i, row in df.iterrows():
        vec = X[i:i+1]
        D,I = query_index(idx, vec, k=cfg.n_neighbors)
        I = I[0]; D = D[0]
        w = 1.0 / (D + 1e-6)
        n_species = [df.iloc[j].get("species") for j in I]
        n_biotypes = [df.iloc[j].get("biotype") for j in I]
        n_descs = [df.iloc[j].get("description") or df.iloc[j].get("annotation") or "" for j in I]
        top_species = _majority(n_species, w, topk=5)
        top_biotypes = _majority(n_biotypes, w, topk=5)
        kw_cands = _keywordize(n_descs, topk=20)
        rows.append(dict(
            id=row.get("id", str(i)),
            top_species=[asdict(c) for c in top_species],
            top_biotypes=[asdict(c) for c in top_biotypes],
            keywords=[asdict(c) for c in kw_cands],
            pathways=[]
        ))
    out_parquet = os.path.join(cfg.out_dir, "hypotheses.parquet")
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    return out_parquet
