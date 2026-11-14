"""Data for the explanatory figures: where windows and annotation labels sit in the embedding, and one worked
example of a hypothesis being proposed, tested and checked (an Arabidopsis tRNA window and its B box).

Everything is read from the frozen lead model and existing result files; nothing here is selected on test outcomes
beyond picking a replicated, literature-matching hypothesis as the worked example.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import HALF, SEED, encode, read_json, write_json
from .hypotheses import class_probability, control_starts, occurrences, randomise_blocks
from .pipeline import Zoeva

LABEL_NEIGHBOURS = 25
QUERY_NEIGHBOURS = 12
EXAMPLE = "arabidopsis|tRNA:GTTCGA"
DRAWS, NULL_DRAWS = 20, 200
TRACK_REACH = 60


def mutagenesis(zoeva: Zoeva, codes: np.ndarray, prototype: np.ndarray, start: int, length: int) -> np.ndarray:
    """In-silico mutagenesis: mean drop in similarity to the cell's text prototype over the three substitutions
    of each base in [start, start + length)."""
    positions = np.repeat(np.arange(start, start + length), 3)
    mutants = np.repeat(codes[None], len(positions), 0)
    mutants[np.arange(len(positions)), positions] = (codes[positions] + np.tile([1, 2, 3], length)) % 4
    return (zoeva.embed(codes[None]) @ prototype).item() - (zoeva.embed(mutants) @ prototype).reshape(length, 3).mean(1)


def run(results: str) -> dict:
    root = Path(results)
    zoeva = Zoeva(results)
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False).set_index("id")
    geometry = read_json(root / "mechanisms" / "geometry.json")["map"]
    ids, xy = np.asarray(geometry["id"]), np.column_stack([geometry["x"], geometry["y"]])
    embeddings = np.load(root / "models" / "clop2_main_s0.embeddings.npz")
    lookup = pd.Series(np.arange(len(embeddings["ids"])), index=embeddings["ids"])
    points = embeddings["dna"][lookup[ids].to_numpy()]

    # where each annotation label (text prototype) points: the 2D position of the windows most similar to it
    similarity = zoeva.prototypes @ points.T
    labels = []
    for c, cell in enumerate(zoeva.cells.names):
        top = np.argsort(-similarity[c])[:LABEL_NEIGHBOURS]
        labels.append({"cell": cell, "text": zoeva.cells.texts[c], "x": float(np.median(xy[top, 0])),
                       "y": float(np.median(xy[top, 1])),
                       "purity": float((np.asarray(geometry["cell"])[top] == cell).mean())})

    # worked example
    catalogue = {h["id"]: h for h in read_json(root / "hypotheses" / "tested.json")["hypotheses"]}
    hypothesis = catalogue[EXAMPLE]
    species, biotype = hypothesis["species"], hypothesis["biotype"]
    candidates = [i for i in np.flatnonzero(np.asarray(geometry["cell"]) == hypothesis["cell"])
                  if occurrences(encode([manifest.loc[ids[i], "sequence"]]), hypothesis["seed"]).any()]
    query = candidates[0]
    codes = encode([manifest.loc[ids[query], "sequence"]])
    hits = occurrences(codes, hypothesis["seed"])
    rng = np.random.default_rng(SEED)
    intact = float(class_probability(zoeva, codes, species, biotype)[0])
    scrambled = class_probability(zoeva, np.concatenate([randomise_blocks(codes, *np.nonzero(hits), hypothesis["seed"], rng)
                                                         for _ in range(DRAWS)]), species, biotype)
    controls = class_probability(zoeva, np.concatenate([randomise_blocks(codes, *control_starts(hits, rng), hypothesis["seed"], rng)
                                                        for _ in range(NULL_DRAWS)]), species, biotype)
    first = int(np.flatnonzero(hits[0])[0])
    lo = max(0, min(first + 3 - TRACK_REACH, codes.shape[1] - 2 * TRACK_REACH))
    importance = mutagenesis(zoeva, codes[0], zoeva.prototypes[zoeva.cells.names.index(hypothesis["cell"])], lo, 2 * TRACK_REACH)
    neighbours = np.argsort(-(points @ points[query]))[1:QUERY_NEIGHBOURS + 1]
    scores = zoeva.hypothesize({"query": manifest.loc[ids[query], "sequence"]})[0]
    payload = {
        "labels": labels,
        "example": {
            "id": str(ids[query]), "cell": hypothesis["cell"], "seed": hypothesis["seed"], "pfm": hypothesis["pfm"],
            "literature": hypothesis["literature"], "tests": hypothesis["tests"],
            "x": float(xy[query, 0]), "y": float(xy[query, 1]),
            "neighbours": [{"x": float(xy[j, 0]), "y": float(xy[j, 1]), "cell": geometry["cell"][j]} for j in neighbours],
            "species_scores": scores["species"], "class_scores": scores["sequence_class_given_top_species"],
            "track_start": lo, "sequence": manifest.loc[ids[query], "sequence"][lo:lo + 2 * TRACK_REACH],
            "importance": importance.round(5).tolist(),
            "seed_positions": [int(p) - lo for p in np.flatnonzero(hits[0]) if lo <= p < lo + 2 * TRACK_REACH],
            "intact_probability": intact, "scrambled_probabilities": scrambled.round(4).tolist(),
            "control_probabilities": controls.round(4).tolist(),
            "window_centre": HALF - lo,
        },
    }
    write_json(root / "mechanisms" / "concept.json", payload)
    return payload
