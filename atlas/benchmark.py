"""Species, sequence-class and cell (species x class) recognition on held-out chromosomes.

Every arm scores the same query windows against the 44 cells; species and class scores are the
marginals of the cell scores. Choices are made on the validation chromosomes, frozen, and the
test chromosomes are scored once.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clop2.model import load

from .common import (BIOTYPE_KEYS, BIOTYPES, SEED, SPECIES_KEYS, cell_text, crop, dinucleotide_signature, encode,
                     file_sha256, kmer_profile, read_json, write_json)
from .models import HELD_OUT_CELLS, run_name

K_GRID = (5, 15, 51)
CORE = 200
N_BOOT = 1000
FEATURES = {
    "composition": dinucleotide_signature,
    "kmer4": lambda codes: kmer_profile(codes, 4),
    "kmer6": lambda codes: kmer_profile(codes, 6),
    "kmer5_core_window": lambda codes: np.hstack([kmer_profile(crop(codes, CORE), 5), kmer_profile(codes, 5)]),
}
TASKS = ("cell", "species", "biotype")


class Cells:
    """The label space: sorted cells with their species and sequence class, and the marginalising matrices."""

    def __init__(self, names):
        self.names = sorted(names)
        species, biotype = zip(*(name.split("|") for name in self.names))
        self.species = np.array([SPECIES_KEYS.index(s) for s in species])
        self.biotype = np.array([BIOTYPE_KEYS.index(b) for b in biotype])
        self.to_species = np.eye(len(SPECIES_KEYS), dtype=np.float32)[self.species]
        self.to_biotype = np.eye(len(BIOTYPE_KEYS), dtype=np.float32)[self.biotype]
        self.texts = [cell_text(s, b) for s, b in zip(species, biotype)]

    def index(self, cells) -> np.ndarray:
        return np.searchsorted(self.names, np.asarray(cells, dtype=str))


def nearest(ref: np.ndarray, query: np.ndarray, k: int, chunk: int = 1024):
    """The k most cosine-similar reference rows per query (rows are unit norm), most similar first."""
    nn, sims = np.empty((len(query), k), np.int64), np.empty((len(query), k), np.float32)
    ref_t = np.ascontiguousarray(ref.T)
    for lo in range(0, len(query), chunk):
        sim = query[lo:lo + chunk] @ ref_t
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        top_sim = np.take_along_axis(sim, top, 1)
        order = np.argsort(-top_sim, axis=1, kind="stable")
        nn[lo:lo + chunk], sims[lo:lo + chunk] = np.take_along_axis(top, order, 1), np.take_along_axis(top_sim, order, 1)
    return nn, sims


def vote_share(nn: np.ndarray, sims: np.ndarray, labels: np.ndarray, n_labels: int, k: int) -> np.ndarray:
    """Similarity-weighted share of the k nearest neighbours per label."""
    weight = np.clip(sims[:, :k], 0, None).astype(np.float64)
    rows = np.arange(len(nn))[:, None] * n_labels + labels[nn[:, :k]]
    votes = np.bincount(rows.ravel(), weight.ravel(), len(nn) * n_labels).reshape(len(nn), n_labels)
    return (votes / np.maximum(votes.sum(1, keepdims=True), 1e-9)).astype(np.float32)


@torch.no_grad()
def text_prototypes(checkpoint: str | Path, prompts) -> tuple[np.ndarray, float]:
    model, _ = load(checkpoint)
    return model.text(list(prompts)).numpy(), float(model.scale())


def confusion(true: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(true * n + pred, minlength=n * n).reshape(n, n)


def macro_f1(conf: np.ndarray) -> np.ndarray:
    """Mean F1 over the classes present in the truth, for confusion matrices of shape (..., n, n)."""
    tp, support = np.diagonal(conf, axis1=-2, axis2=-1), conf.sum(-1)
    f1 = 2 * tp / np.maximum(support + conf.sum(-2), 1)
    return (f1 * (support > 0)).sum(-1) / np.maximum((support > 0).sum(-1), 1)


def predictions(scores: np.ndarray, cells: Cells, true_species: np.ndarray) -> dict[str, np.ndarray]:
    """Top call per task, plus the class call when candidates are restricted to the true species' cells."""
    own = cells.species[None, :] == true_species[:, None]
    return {"cell": scores.argmax(1), "species": (scores @ cells.to_species).argmax(1),
            "biotype": (scores @ cells.to_biotype).argmax(1),
            "biotype_given_species": cells.biotype[np.where(own, scores, -np.inf).argmax(1)]}


def truth_of(table: pd.DataFrame, cells: Cells) -> dict[str, np.ndarray]:
    cell = cells.index(table["cell"])
    return {"cell": cell, "species": cells.species[cell], "biotype": cells.biotype[cell]}


def summarise(pred: dict, truth: dict, cells: Cells) -> dict:
    sizes = {"cell": len(cells.names), "species": len(SPECIES_KEYS), "biotype": len(BIOTYPE_KEYS)}
    out = {task: {"macro_f1": float(macro_f1(confusion(truth[task], pred[task], n))),
                  "accuracy": float((truth[task] == pred[task]).mean())} for task, n in sizes.items()}
    within = {SPECIES_KEYS[s]: float(macro_f1(confusion(truth["biotype"][rows], pred["biotype_given_species"][rows], sizes["biotype"])))
              for s in range(len(SPECIES_KEYS)) if (rows := truth["species"] == s).any()}
    out["biotype_given_species"] = {"macro_f1_mean_over_species": float(np.mean(list(within.values()))), "per_species": within}
    return out


def bootstrap(truth: dict, preds: dict[str, dict], anchor: str, cells: Cells) -> dict:
    """Percentile 95% intervals of macro F1 per task and arm, and of the paired difference to the anchor.
    Resamples query windows; conditional on the chromosomes, corpus and fitted models."""
    n_rows = len(truth["cell"])
    idx = np.random.default_rng(SEED).integers(0, n_rows, (N_BOOT, n_rows))
    offsets = np.arange(N_BOOT)[:, None]
    interval = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    out = {}
    for task, n in (("cell", len(cells.names)), ("species", len(SPECIES_KEYS)), ("biotype", len(BIOTYPE_KEYS))):
        resampled = {arm: macro_f1(np.bincount(((truth[task] * n + pred[task])[idx] + offsets * n * n).ravel(),
                                               minlength=N_BOOT * n * n).reshape(N_BOOT, n, n)) for arm, pred in preds.items()}
        for arm, values in resampled.items():
            out.setdefault(arm, {})[task] = {"macro_f1_ci95": interval(values),
                                             "minus_anchor_ci95": interval(values - resampled[anchor])}
    return out


def logistic(x_train: np.ndarray, y_train: np.ndarray) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(x_train, y_train)


def baseline_scores(manifest: pd.DataFrame, cells: Cells, cache: Path) -> dict[str, np.ndarray]:
    """Cell probabilities of every manifest row from each sequence-statistics baseline fitted on training rows."""
    cache.mkdir(parents=True, exist_ok=True)
    codes, train, y = None, (manifest["split"] == "train").to_numpy(), cells.index(manifest["cell"])
    out = {}
    for name, featurise in FEATURES.items():
        path = cache / f"baseline_{name}.npy"
        if not path.exists():
            codes = encode(manifest["sequence"]) if codes is None else codes
            x = featurise(codes)
            np.save(path, logistic(x[train], y[train]).predict_proba(x).astype(np.float32))
            if name == "kmer4":
                np.save(cache / "kmer4_profile.npy", x)
        out[f"{name}_logistic"] = np.load(path)
    return out


def clop2_scores(embeddings: np.ndarray, checkpoint: Path, manifest: pd.DataFrame, cells: Cells, query: np.ndarray,
                 shuffle: bool = False) -> dict[str, np.ndarray]:
    """Cell scores of the query rows: text-prototype probabilities and neighbour vote shares for every k."""
    train = np.flatnonzero((manifest["split"] == "train").to_numpy())
    labels = cells.index(manifest["cell"].to_numpy()[train])
    if shuffle:
        labels = np.random.default_rng(SEED).permutation(labels)
    nn, sims = nearest(embeddings[train], embeddings[query], max(K_GRID))
    prototypes, scale = text_prototypes(checkpoint, cells.texts)
    return {"text": softmax(scale * embeddings[query] @ prototypes.T, axis=1)} | {
        f"knn_k{k}": vote_share(nn, sims, labels, len(cells.names), k) for k in K_GRID}


def evaluate(results: str, split: str, seeds=(0, 1)) -> dict:
    """Score all arms on one split. Validation chooses k and the anchor baseline; test reuses the frozen choices."""
    root = Path(results)
    target = root / f"{split}_metrics.json"
    if split == "test" and target.exists():
        raise SystemExit(f"{target} exists: the test chromosomes are scored once.")
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False)
    cells = Cells(manifest["cell"].unique())
    query = np.flatnonzero((manifest["split"] == split).to_numpy())
    truth = truth_of(manifest.iloc[query], cells)
    protocol = read_json(root / "protocol.json") if split == "test" else None
    if protocol and protocol["manifest_sha256"] != file_sha256(root / "manifest.tsv"):
        raise SystemExit("manifest changed since the protocol was frozen")
    scores = {name: values[query] for name, values in baseline_scores(manifest, cells, root / "cache").items()}
    kmer4 = np.load(root / "cache" / "kmer4_profile.npy")
    train = np.flatnonzero((manifest["split"] == "train").to_numpy())
    scores["kmer4_knn"] = vote_share(*nearest(kmer4[train], kmer4[query], 15), cells.index(manifest["cell"].to_numpy()[train]),
                                     len(cells.names), 15)
    scores["majority"] = np.tile(np.bincount(cells.index(manifest["cell"].to_numpy()[train]), minlength=len(cells.names))
                                 .astype(np.float32), (len(query), 1))
    k_by_model = {}
    for seed in seeds:
        name = run_name("main", seed)
        embeddings = np.load(root / "models" / f"{name}.embeddings.npz")["dna"]
        arms = clop2_scores(embeddings, root / "models" / f"{name}.pt", manifest, cells, query)
        if protocol:
            k = protocol["k"][name]
        else:
            k = max(K_GRID, key=lambda v: macro_f1(confusion(truth["cell"], arms[f"knn_k{v}"].argmax(1), len(cells.names))))
        k_by_model[name] = int(k)
        scores |= {f"{name}_knn": arms[f"knn_k{k}"], f"{name}_text": arms["text"],
                   f"{name}_zoeva": 0.5 * arms[f"knn_k{k}"] + 0.5 * arms["text"]}
        if seed == seeds[0]:
            scores |= {f"{name}_knn_k{v}": arms[f"knn_k{v}"] for v in K_GRID}
            scores[f"{name}_knn_shuffled_labels"] = clop2_scores(embeddings, root / "models" / f"{name}.pt", manifest,
                                                                 cells, query, shuffle=True)[f"knn_k{k}"]
    preds = {arm: predictions(values, cells, truth["species"]) for arm, values in scores.items()}
    metrics = {arm: summarise(pred, truth, cells) for arm, pred in preds.items()}
    baselines = [arm for arm in metrics if arm.endswith("_logistic") or arm == "kmer4_knn"]
    anchor = protocol["anchor"] if protocol else max(baselines, key=lambda arm: metrics[arm]["cell"]["macro_f1"])
    for arm, intervals in bootstrap(truth, preds, anchor, cells).items():
        for task in TASKS:
            metrics[arm][task] |= intervals[task]
    lead = f"{run_name('main', seeds[0])}_zoeva"
    species_by_biotype = pd.crosstab(manifest["biotype"].to_numpy()[query], manifest["species"].to_numpy()[query],
                                     values=preds[lead]["species"] == truth["species"], aggfunc="mean")
    payload = {
        "split": split, "n_queries": int(len(query)), "created_utc": datetime.now(timezone.utc).isoformat(),
        "cells": cells.names, "species": list(SPECIES_KEYS), "biotypes": list(BIOTYPE_KEYS), "anchor": anchor,
        "k": k_by_model, "arms": metrics, "lead_arm": lead,
        "chance": {"cell": 1 / len(cells.names), "species": 1 / len(SPECIES_KEYS), "biotype": 1 / len(BIOTYPE_KEYS)},
        "seed_mean": {kind: {task: float(np.mean([metrics[f"{run_name('main', s)}_{kind}"][task]["macro_f1"] for s in seeds]))
                             for task in TASKS} for kind in ("knn", "text", "zoeva")},
        "confusion": {task: confusion(truth[task], preds[lead][task], n).tolist()
                      for task, n in (("cell", len(cells.names)), ("species", len(SPECIES_KEYS)), ("biotype", len(BIOTYPE_KEYS)))},
        "biotype_given_species_confusion": confusion(truth["biotype"], preds[lead]["biotype_given_species"], len(BIOTYPE_KEYS)).tolist(),
        "species_recall_by_biotype": {b: {s: (None if np.isnan(v) else float(v)) for s, v in row.items()}
                                      for b, row in species_by_biotype.to_dict("index").items()},
    }
    write_json(target, payload)
    table = manifest.iloc[query][["id", "species", "biotype", "cell"]].copy()
    table["predicted_cell"] = np.array(cells.names)[preds[lead]["cell"]]
    table["score"] = scores[lead].max(1)
    table.to_csv(root / f"{split}_predictions.tsv", sep="\t", index=False)
    np.save(root / f"{split}_scores_lead.npy", scores[lead])
    return payload


def freeze(results: str, seeds=(0, 1)) -> dict:
    """Write the protocol: validation choices and the hashes of everything the test run reads."""
    root = Path(results)
    validation = read_json(root / "validation_metrics.json")
    files = [root / "models" / f"{run_name('main', s)}{suffix}" for s in seeds for suffix in (".pt", ".embeddings.npz")]
    payload = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "k": validation["k"], "anchor": validation["anchor"],
               "lead_arm": validation["lead_arm"], "manifest_sha256": file_sha256(root / "manifest.tsv"),
               "files_sha256": {path.name: file_sha256(path) for path in files},
               "primary_metric": "macro F1 over the 44 cells; species and class macro F1 from marginal scores",
               "held_out_cells_of_compositional_model": list(HELD_OUT_CELLS),
               "class_names": {key: value[0] for key, value in BIOTYPES.items()}}
    write_json(root / "protocol.json", payload)
    return payload
