"""Generalisation beyond the training cells.

compositional: a model that never saw eight species x class cells must name them from prompts that
combine a seen species with a seen class.
leave_one_species_out: a model that never saw a species must name the classes of its windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from .benchmark import FEATURES, Cells, confusion, logistic, macro_f1, nearest, text_prototypes, vote_share
from .common import BIOTYPE_KEYS, BIOTYPES, SPECIES_KEYS, encode, write_json
from .models import HELD_OUT_CELLS, run_name, variant_rows

K = 15
BASELINE = "kmer5_core_window"


def load_run(root: Path, variant: str):
    name = run_name(variant, 0)
    return np.load(root / "models" / f"{name}.embeddings.npz")["dna"], root / "models" / f"{name}.pt"


def factorised(species: np.ndarray, biotype: np.ndarray, cells: Cells) -> np.ndarray:
    """Cell score as the product of a species score and a class score."""
    return species[:, cells.species] * biotype[:, cells.biotype]


def cell_report(scores: np.ndarray, truth: np.ndarray, cells: Cells) -> dict:
    """Recognition of the true cell among all cells, and of its two factors."""
    order = np.argsort(-scores, axis=1, kind="stable")
    rank = np.argmax(order == truth[:, None], axis=1) + 1
    own = cells.species[None, :] == cells.species[truth][:, None]
    return {"n": int(len(truth)), "cell_top1": float((rank == 1).mean()), "cell_top3": float((rank <= 3).mean()),
            "cell_mrr": float((1 / rank).mean()),
            "species": float(((scores @ cells.to_species).argmax(1) == cells.species[truth]).mean()),
            "biotype": float(((scores @ cells.to_biotype).argmax(1) == cells.biotype[truth]).mean()),
            "biotype_given_species": float((cells.biotype[np.where(own, scores, -np.inf).argmax(1)] == cells.biotype[truth]).mean())}


def compositional(results: str) -> dict:
    root = Path(results)
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False)
    cells = Cells(manifest["cell"].unique())
    query = np.flatnonzero(((manifest["split"] == "test") & manifest["cell"].isin(HELD_OUT_CELLS)).to_numpy())
    train = np.flatnonzero(variant_rows(manifest, "compositional").to_numpy())
    truth = cells.index(manifest["cell"].to_numpy()[query])
    species = np.array([SPECIES_KEYS.index(s) for s in manifest["species"]])
    biotype = np.array([BIOTYPE_KEYS.index(b) for b in manifest["biotype"]])
    arms = {}
    for variant, label in (("compositional", "zero_shot_text"), ("main", "trained_on_cell_text")):
        embeddings, checkpoint = load_run(root, variant)
        prototypes, scale = text_prototypes(checkpoint, cells.texts)
        arms[label] = softmax(scale * embeddings[query] @ prototypes.T, axis=1)
        if variant == "compositional":
            nn, sims = nearest(embeddings[train], embeddings[query], K)
            arms["zero_shot_factorised_knn"] = factorised(vote_share(nn, sims, species[train], len(SPECIES_KEYS), K),
                                                          vote_share(nn, sims, biotype[train], len(BIOTYPE_KEYS), K), cells)
            arms["zero_shot_fused"] = 0.5 * arms[label] + 0.5 * arms["zero_shot_factorised_knn"] / np.maximum(
                arms["zero_shot_factorised_knn"].sum(1, keepdims=True), 1e-9)
    x = FEATURES[BASELINE](encode(manifest["sequence"].to_numpy()[np.r_[train, query]]))
    halves = []
    for labels, size in ((species, len(SPECIES_KEYS)), (biotype, len(BIOTYPE_KEYS))):
        model = logistic(x[:len(train)], labels[train])
        proba = np.zeros((len(query), size), dtype=np.float32)
        proba[:, model.classes_] = model.predict_proba(x[len(train):])
        halves.append(proba)
    arms[f"zero_shot_factorised_{BASELINE}_logistic"] = factorised(*halves, cells)
    payload = {"held_out_cells": list(HELD_OUT_CELLS), "n_cells": len(cells.names), "chance_cell_top1": 1 / len(cells.names),
               "arms": {arm: {"overall": cell_report(scores, truth, cells)} | {
                   cell: cell_report(scores[truth == i], truth[truth == i], cells)
                   for cell in HELD_OUT_CELLS for i in [cells.names.index(cell)]} for arm, scores in arms.items()}}
    write_json(root / "compositional.json", payload)
    return payload


def leave_one_species_out(results: str) -> dict:
    root = Path(results)
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False)
    species, split = manifest["species"].to_numpy(), manifest["split"].to_numpy()
    biotype = np.array([BIOTYPE_KEYS.index(b) for b in manifest["biotype"]])
    codes, payload = encode(manifest["sequence"]), {}
    x = FEATURES[BASELINE](codes)
    main, _ = load_run(root, "main")
    for held in SPECIES_KEYS:
        train = np.flatnonzero((split == "train") & (species != held))
        candidates = np.intersect1d(biotype[train], biotype[species == held])
        query = np.flatnonzero((split == "test") & (species == held) & np.isin(biotype, candidates))
        own = np.flatnonzero((split == "train") & (species == held) & np.isin(biotype, candidates))
        truth, mask = biotype[query], np.isin(np.arange(len(BIOTYPE_KEYS)), candidates)
        embeddings, checkpoint = load_run(root, f"without_{held}")
        nn, sims = nearest(embeddings[train], embeddings[query], K)
        prototypes, scale = text_prototypes(checkpoint, [BIOTYPES[BIOTYPE_KEYS[b]][1] for b in candidates])
        text = np.zeros((len(query), len(BIOTYPE_KEYS)), dtype=np.float32)
        text[:, candidates] = softmax(scale * embeddings[query] @ prototypes.T, axis=1)
        votes = vote_share(nn, sims, biotype[train], len(BIOTYPE_KEYS), K)
        baseline = logistic(x[train], biotype[train])
        proba = np.zeros_like(text)
        proba[:, baseline.classes_] = baseline.predict_proba(x[query])
        arms = {"unseen_species_knn": votes, "unseen_species_text": text, "unseen_species_zoeva": 0.5 * votes + 0.5 * text,
                f"unseen_species_{BASELINE}_logistic": proba,
                "species_in_training_knn": vote_share(*nearest(main[own], main[query], K), biotype[own], len(BIOTYPE_KEYS), K)}
        report = {}
        for arm, scores in arms.items():
            conf = confusion(truth, np.where(mask, scores, -1).argmax(1), len(BIOTYPE_KEYS))
            report[arm] = {"macro_f1": float(macro_f1(conf)), "accuracy": float(np.trace(conf) / conf.sum()),
                           "recall": {BIOTYPE_KEYS[b]: float(conf[b, b] / conf[b].sum()) for b in candidates},
                           "confusion": conf[np.ix_(candidates, candidates)].tolist()}
        neighbours = pd.Series(species[train][nn[:, 0]]).value_counts(normalize=True)
        payload[held] = {"n_queries": int(len(query)), "classes": [BIOTYPE_KEYS[b] for b in candidates],
                         "chance": 1 / len(candidates), "arms": report,
                         "species_of_nearest_training_window": {s: float(neighbours.get(s, 0)) for s in SPECIES_KEYS if s != held}}
    write_json(root / "leave_one_species_out.json", payload)
    return payload
