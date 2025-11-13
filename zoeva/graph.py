from __future__ import annotations
import os, pandas as pd, networkx as nx
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from .config import ensure_dir

@dataclass
class GraphConfig:
    hypotheses_parquet: str
    out_dir: str = "run/graph"
    kw_to_label_tsv: Optional[str] = None
    label_edges_tsv: Optional[str] = None
    alpha: float = 0.85
    topk: int = 50

def _add_edge(G: nx.DiGraph, u: str, v: str, w: float, rel: str):
    if G.has_edge(u, v):
        G[u][v]["weight"] += float(w)
        G[u][v]["rels"].add(rel)
    else:
        G.add_edge(u, v, weight=float(w), rels=set([rel]))

def _node_type(x: str) -> str:
    if x.startswith("seq:"): return "seq"
    if x.startswith("kw:"): return "kw"
    if x.startswith("go:") or x.startswith("kegg:") or x.startswith("reactome:") or x.startswith("ko:") or x.startswith("label:"):
        return "label"
    return "other"

def build_graph(cfg: GraphConfig) -> Tuple[nx.DiGraph, pd.DataFrame]:
    df = pd.read_parquet(cfg.hypotheses_parquet)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        sid = f"seq:{row['id']}"
        if not G.has_node(sid): G.add_node(sid, ntype="seq")
        # keywords
        kws = (row.get("keywords") or [])
        for k in kws:
            kw = f"kw:{k['label']}"
            G.add_node(kw, ntype="kw")
            _add_edge(G, sid, kw, k.get("score", 0.1), "hypo_kw")
        # species/biotype/pathways
        for c in (row.get("top_species") or []):
            lab = f"label:species:{c['label']}"
            G.add_node(lab, ntype="label")
            _add_edge(G, sid, lab, c.get("score", 0.1), "hypo_species")
        for c in (row.get("top_biotypes") or []):
            lab = f"label:biotype:{c['label']}"
            G.add_node(lab, ntype="label")
            _add_edge(G, sid, lab, c.get("score", 0.1), "hypo_biotype")
        for c in (row.get("pathways") or []):
            lab = str(c['label']).lower()
            G.add_node(lab, ntype="label")
            _add_edge(G, sid, lab, c.get("score", 0.1), "hypo_pathway")

    if cfg.kw_to_label_tsv and os.path.exists(cfg.kw_to_label_tsv):
        m = pd.read_csv(cfg.kw_to_label_tsv, sep="\t", comment="#", header=0)
        for _, r in m.iterrows():
            kw = f"kw:{str(r['keyword']).strip().lower()}"
            lab = str(r['label']).strip().lower()
            if not G.has_node(kw): G.add_node(kw, ntype="kw")
            if not G.has_node(lab): G.add_node(lab, ntype="label")
            w = float(r.get("weight", 1.0))
            _add_edge(G, kw, lab, w, "kw_map")

    if cfg.label_edges_tsv and os.path.exists(cfg.label_edges_tsv):
        e = pd.read_csv(cfg.label_edges_tsv, sep="\t", comment="#", header=0)
        for _, r in e.iterrows():
            s = str(r['src']).strip().lower()
            t = str(r['dst']).strip().lower()
            rel = str(r.get('rel', 'related')).strip().lower()
            w = float(r.get('weight', 1.0))
            if not G.has_node(s): G.add_node(s, ntype="label")
            if not G.has_node(t): G.add_node(t, ntype="label")
            _add_edge(G, s, t, w, f"label_{rel}")

    edges = []
    for u, v, d in G.edges(data=True):
        rels = ";".join(sorted([str(x) for x in d.get("rels", set())]))
        edges.append(dict(u=u, v=v, weight=float(d.get("weight", 1.0)), rels=rels))
    edges_df = pd.DataFrame(edges)
    return G, edges_df

def _ppr_for_sequence(G: nx.DiGraph, seq_id: str, alpha: float):
    s = f"seq:{seq_id}"
    if s not in G: return {}
    pers = {n: 0.0 for n in G.nodes()}
    pers[s] = 1.0
    # make sure edge attrs are serializable
    for u, v, d in G.edges(data=True):
        if isinstance(d.get("rels"), set):
            d["rels"] = ";".join(sorted(d["rels"]))
    pr = nx.pagerank(G, alpha=alpha, personalization=pers, weight="weight")
    return pr

def propagate(cfg: GraphConfig) -> str:
    ensure_dir(cfg.out_dir)
    G, edges_df = build_graph(cfg)
    edges_df.to_parquet(os.path.join(cfg.out_dir, "graph_edges.parquet"), index=False)
    seq_nodes = [n for n, t in G.nodes(data="ntype") if t == "seq"]
    out_rows = []
    for s in seq_nodes:
        sid = s.split("seq:", 1)[1]
        pr = _ppr_for_sequence(G, sid, cfg.alpha)
        items = [(n, sc) for n, sc in pr.items() if n != s and _node_type(n) == "label"]
        items.sort(key=lambda x: -x[1])
        for n, sc in items[:cfg.topk]:
            out_rows.append(dict(id=sid, label=n, score_graph=float(sc)))
    out = os.path.join(cfg.out_dir, "graph_scores.parquet")
    pd.DataFrame(out_rows).to_parquet(out, index=False)
    return out
