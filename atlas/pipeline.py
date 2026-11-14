"""Zoeva: from a DNA sequence to ranked species and sequence-class hypotheses with their evidence.

The cell score is the mean of two views of one CLOP2 embedding: the similarity-weighted vote of the
nearest training windows, and the softmax over the text prototypes of the 44 species x class cells.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from clop2.data import encode_ragged
from clop2.model import embed, load

from .benchmark import K_GRID, Cells, nearest, text_prototypes, vote_share
from .common import BIOTYPE_KEYS, BIOTYPES, SPECIES, SPECIES_KEYS, WINDOW, dinucleotide_signature, read_json
from .models import run_name

TOP = 3
SCAN_STEP = 100
N_NEIGHBOURS = 5


class Zoeva:
    def __init__(self, results: str, name: str = run_name("main", 0)):
        root = Path(results)
        manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False)
        rows = np.flatnonzero((manifest["split"] == "train").to_numpy())
        self.references = manifest.iloc[rows][["id", "cell"]].reset_index(drop=True)
        self.cells = Cells(manifest["cell"].unique())
        self.labels = self.cells.index(self.references["cell"])
        self.index = np.load(root / "models" / f"{name}.embeddings.npz")["dna"][rows]
        self.model, _ = load(root / "models" / f"{name}.pt")
        self.prototypes, self.scale = text_prototypes(root / "models" / f"{name}.pt", self.cells.texts)
        protocol = root / "protocol.json"
        self.k = read_json(protocol)["k"][name] if protocol.exists() else K_GRID[1]

    def embed(self, codes: np.ndarray, lengths: np.ndarray | None = None) -> np.ndarray:
        return embed(self.model, codes, lengths)

    def text_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity of each embedding to each cell's text prototype."""
        return embeddings @ self.prototypes.T

    def scores(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fused cell scores (rows sum to one), and the neighbours behind the retrieval half."""
        nn, sims = nearest(self.index, embeddings, max(self.k, N_NEIGHBOURS))
        votes = vote_share(nn, sims, self.labels, len(self.cells.names), self.k)
        return 0.5 * votes + 0.5 * softmax(self.scale * self.text_similarity(embeddings), axis=1), nn, sims

    def class_given_species(self, scores: np.ndarray, species: str) -> np.ndarray:
        """Class probabilities when the species is known: the species' cells, renormalised. Columns follow BIOTYPE_KEYS."""
        own = self.cells.species == SPECIES_KEYS.index(species)
        out = (scores * own) @ self.cells.to_biotype
        return out / np.maximum(out.sum(1, keepdims=True), 1e-9)

    def hypothesize(self, sequences: dict[str, str]) -> list[dict]:
        """One report per sequence. Inputs longer than the training window are cut to their central 500 bp."""
        trimmed = [s[max(0, (len(s) - WINDOW) // 2):][:WINDOW] for s in sequences.values()]
        codes, lengths = encode_ragged(trimmed)
        scores, nn, sims = self.scores(self.embed(codes, lengths))
        species, biotype = scores @ self.cells.to_species, scores @ self.cells.to_biotype
        signature = dinucleotide_signature(np.where(codes > 3, 4, codes))
        reports = []
        for i, name in enumerate(sequences):
            best = SPECIES_KEYS[int(species[i].argmax())]
            within = self.class_given_species(scores[i:i + 1], best)[0]
            ranked = lambda values, keys: [{"label": keys[j], "score": float(values[j])} for j in np.argsort(-values)[:TOP]]
            reports.append({
                "id": name, "length": int(lengths[i]), "species": ranked(species[i], SPECIES_KEYS),
                "sequence_class": ranked(biotype[i], BIOTYPE_KEYS),
                "sequence_class_given_top_species": ranked(within, BIOTYPE_KEYS),
                "cell": ranked(scores[i], self.cells.names),
                "nearest_training_windows": [{"id": self.references["id"][j], "cell": self.references["cell"][j],
                                              "cosine": float(s)} for j, s in zip(nn[i, :N_NEIGHBOURS], sims[i, :N_NEIGHBOURS])],
                "composition": {"gc": float(signature[i, 16]), "cpg_observed_over_expected": float(signature[i, 6])},
            })
        return reports

    def scan(self, codes: np.ndarray, step: int = SCAN_STEP) -> tuple[np.ndarray, np.ndarray]:
        """Tile a long sequence: window centres and fused cell scores per tile."""
        starts = np.arange(0, len(codes) - WINDOW + 1, step)
        tiles = np.asarray(codes)[starts[:, None] + np.arange(WINDOW)]
        return starts + WINDOW // 2, self.scores(self.embed(tiles))[0]


def markdown(reports: list[dict]) -> str:
    name = lambda label: BIOTYPES[label][0] if label in BIOTYPES else SPECIES[label][0] if label in SPECIES else label
    line = lambda items: ", ".join(f"{name(item['label'])} {item['score']:.2f}" for item in items)
    blocks = []
    for r in reports:
        blocks.append(
            f"## {r['id']} ({r['length']} bp)\n\n- Species: {line(r['species'])}\n- Sequence class: {line(r['sequence_class'])}\n"
            f"- Class within {name(r['species'][0]['label'])}: {line(r['sequence_class_given_top_species'])}\n"
            f"- GC {r['composition']['gc']:.2f}, CpG observed/expected {r['composition']['cpg_observed_over_expected']:.2f}\n"
            "- Nearest training windows: " + "; ".join(f"{n['id']} ({n['cosine']:.2f})" for n in r["nearest_training_windows"]) + "\n")
    return "# Zoeva hypotheses\n\nExploratory rankings from sequence alone; not an annotation of record.\n\n" + "\n".join(blocks)
