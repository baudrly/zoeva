import os, pandas as pd, numpy as np
from typing import Optional
from dataclasses import dataclass
from .config import ensure_dir

@dataclass
class ScoreWeights:
    neighbor: float = 0.6
    zeroshot: float = 0.4
    graph_weight: float = 0.35
    blast_bitscore_weight: float = 0.05
    pfam_hit_weight: float = 0.05
    kofam_hit_weight: float = 0.10
    eggnog_hit_weight: float = 0.10

def score_hypotheses(hypo_parquet: str, validation_summary: Optional[str], out_dir: str, weights: ScoreWeights = ScoreWeights(), graph_parquet: Optional[str] = None) -> str:
    ensure_dir(out_dir)
    hypo = pd.read_parquet(hypo_parquet)
    records = []
    for _,r in hypo.iterrows():
        for k in (r.top_species or []):
            records.append({"id": r.id, "type":"species", "label": k['label'], "score_raw": k['score'], "source": k['source']})
        for k in (r.top_biotypes or []):
            records.append({"id": r.id, "type":"biotype", "label": k['label'], "score_raw": k['score'], "source": k['source']})
        for k in (r.keywords or []):
            records.append({"id": r.id, "type":"keyword", "label": k['label'], "score_raw": k['score'], "source": k['source']})
        for k in (r.pathways or []):
            records.append({"id": r.id, "type":"pathway", "label": k['label'], "score_raw": k['score'], "source": k['source']})
    df = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=["id","type","label","score_raw","source"])

    # merge graph-based label scores if provided
    if graph_parquet and os.path.exists(graph_parquet):
        gp = pd.read_parquet(graph_parquet)
        gp['score_raw'] = gp.groupby('id')['score_graph'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        gp = gp.rename(columns={'label':'label'})
        gp['type'] = 'graph_label'
        gp['source'] = 'graph'
        df_graph = gp[['id','type','label','score_raw','source']]
        df_graph['score'] = df_graph['score_raw'] * weights.graph_weight
        df = pd.concat([df, df_graph], ignore_index=True, sort=False)

    if df.empty:
        out = os.path.join(out_dir, "scored.parquet")
        df.to_parquet(out, index=False); return out

    def wrow(row):
        if row.source == "neighbor": return row.score_raw * weights.neighbor
        if row.source == "zeroshot": return row.score_raw * weights.zeroshot
        if row.source == "graph": return row.score  # already weighted
        return row.score_raw

    if 'score' not in df.columns:
        df['score'] = df.apply(wrow, axis=1)
    else:
        # mix neighbor/zeroshot where needed
        mask = df['source'].isin(['neighbor','zeroshot']) & df['score'].isna()
        df.loc[mask, 'score'] = df[mask].apply(wrow, axis=1)

    if validation_summary and os.path.exists(validation_summary):
        vs = pd.read_parquet(validation_summary)
        boost = 0.0
        if any(vs.source == "pfam"): boost += weights.pfam_hit_weight
        if any(vs.source == "kofam"): boost += weights.kofam_hit_weight
        if any(vs.source == "eggnog"): boost += weights.eggnog_hit_weight
        df["score"] = np.clip(df["score"] + boost, 0, 1.5)

    out = os.path.join(out_dir, "scored.parquet")
    df.sort_values(["id","type","score"], ascending=[True, True, False]).to_parquet(out, index=False)
    return out
