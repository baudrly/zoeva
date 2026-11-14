"""Bundle the CLOP2 encoder, a demo Zoeva index, examples and result tables into one self-contained page."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from clop2.model import load

from .benchmark import Cells, text_prototypes
from .common import BIOTYPES, SEED, SPECIES, decode, encode, read_json, revcomp
from .figures import ARM_NAMES, SHOWN_ARMS, SPECIES_SHORT
from .literature import library
from .models import HELD_OUT_CELLS

REFERENCES_PER_CELL = 60
EXAMPLES_PER_CELL = 3
FIGURE_WIDTH = 1500
GALLERY = {"fig2_benchmark": "Benchmark on held-out chromosomes", "fig3_confusions": "Confusions of the lead arm",
           "fig4_species_by_class": "Which sequence classes carry species identity",
           "fig5_transfer": "Compositional and cross-species transfer", "fig6_shuffles": "Shuffle ladder and reading frame",
           "fig7_landmarks": "Occlusion around landmarks", "fig9_geometry": "Embedding geometry",
           "fig10_signatures": "Dinucleotide signatures", "fig12_hypotheses": "Automatic hypotheses", "fig13_asthma": "Asthma variants", "fig11_loci": "Asthma-associated loci (exploratory)"}
PLACEHOLDER = "/*__BUNDLE__*/"


def b64(array: np.ndarray, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


@torch.no_grad()
def folded_weights(path: Path) -> dict:
    """Convolution weights with batch-norm statistics folded in, so the browser needs no normalisation layer."""
    model, state = load(path)
    dna, out = model.dna, {"arch": state["arch"]}
    layers = [("conv1", dna.conv1, dna.norms[0]), ("conv2", dna.conv2, dna.norms[1]),
              *[(f"block{i}", block, norm) for i, (block, norm) in enumerate(zip(dna.blocks, dna.norms[2:]))]]
    for name, conv, norm in layers:
        gain = norm.weight / (norm.running_var + norm.eps).sqrt()
        out[f"{name}_w"] = b64((conv.weight * gain[:, None, None]).numpy())
        out[f"{name}_b"] = b64(((conv.bias - norm.running_mean) * gain + norm.bias).numpy())
    out["fc_w"], out["fc_b"] = b64(dna.fc.weight.numpy()), b64(dna.fc.bias.numpy())
    return out


def figure(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    if image.width > FIGURE_WIDTH:
        image = image.resize((FIGURE_WIDTH, round(image.height * FIGURE_WIDTH / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.quantize(256, method=Image.Quantize.MEDIANCUT).save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def web_hypotheses(root: Path, manifest: pd.DataFrame) -> dict:
    """Tested catalogue for the page, each motif with one held-out window that carries it."""
    tested = read_json(root / "hypotheses" / "tested.json")
    test = manifest[manifest["split"] == "test"]
    items = []
    for h in tested["hypotheses"]:
        item = {key: h[key] for key in ("id", "family", "cell", "verdict", "status")} | {
            "tests": {split: {k: v for k, v in t.items() if not k.startswith("p_")} for split, t in h["tests"].items()}}
        if h["family"] == "motif":
            rows = test.loc[test["cell"] == h["cell"], "sequence"]
            rc = decode(revcomp(encode([h["seed"]])))[0]
            carriers = rows[rows.str.contains(h["seed"]) | rows.str.contains(rc)]
            item |= {key: h[key] for key in ("seed", "pfm", "literature", "jaspar", "sufficient")} | {
                "example": carriers.iloc[0] if len(carriers) else None}
        items.append(item)
    order = {"replicated": 0, "one split only": 1, "not supported": 2}
    items.sort(key=lambda h: (order[h["verdict"]], -(h["tests"]["test"].get("necessity_relative") or 0)))
    return {"summary": tested["summary"], "items": items, "recall": tested["expected_recall"]}


def web_asthma(root: Path) -> dict:
    """Variant readouts, pair tests and the published-mechanism checks, without the locus tracks."""
    data = read_json(root / "asthma" / "asthma.json")
    keep = ("name", "gene", "consequence", "group", "risk", "ref", "alt", "chrom", "pos", "held_out_chromosome",
            "literature_mechanism", "references", "largest_class_shift", "largest_shift_class", "class_shift_percentile",
            "attribution_percentile", "seeds_changed", "literature_check", "motif_leads", "motifs_present_above_95th",
            "motifs_present_above_95th_expected", "windows", "allele_note")
    return {"summary": {k: v for k, v in data["summary"].items() if k != "motif_leads"},
            "variants": [{k: v[k] for k in keep} for v in data["variants"]], "pairs": data["pairs"]}


def build(results: str, template: str, out: str, name: str = "clop2_main_s0") -> None:
    root, mech = Path(results), Path(results) / "mechanisms"
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False)
    cells = Cells(manifest["cell"].unique())
    embeddings = np.load(root / "models" / f"{name}.embeddings.npz")["dna"]
    shuffled = manifest.sample(frac=1, random_state=SEED)
    references = shuffled[shuffled["split"] == "train"].groupby("cell").head(REFERENCES_PER_CELL)
    examples = shuffled[shuffled["split"] == "test"].groupby("cell").head(EXAMPLES_PER_CELL).sort_values("cell")
    prototypes, scale = text_prototypes(root / "models" / f"{name}.pt", cells.texts)
    test, protocol = read_json(root / "test_metrics.json"), read_json(root / "protocol.json")
    geometry = read_json(mech / "geometry.json")
    by_id = manifest.set_index("id")["sequence"]
    called = pd.read_csv(root / "test_predictions.tsv", sep="\t").set_index("id")["predicted_cell"]
    filters = read_json(mech / "filters.json")
    bundle = {
        "model": folded_weights(root / "models" / f"{name}.pt") | {"prototypes": b64(prototypes), "scale": scale},
        "k": protocol["k"][name], "cells": cells.names,
        "species": {key: {"name": value[0], "short": SPECIES_SHORT[key]} for key, value in SPECIES.items()},
        "classes": {key: value[0] for key, value in BIOTYPES.items()},
        "index": {"embeddings": b64(embeddings[references.index.to_numpy()]), "cell": cells.index(references["cell"]).tolist(),
                  "id": references["id"].tolist()},
        "examples": [{"id": r.id, "cell": r.cell, "sequence": r.sequence, "called": called[r.id]} for r in examples.itertuples()],
        "preparation": read_json(root / "preparation.json")["cells"],
        "benchmark": {"arms": {arm: {task: test["arms"][arm][task] for task in ("species", "biotype", "cell", "biotype_given_species")}
                               for arm in SHOWN_ARMS if arm in test["arms"]},
                      "names": ARM_NAMES, "chance": test["chance"], "n": test["n_queries"],
                      "species_recall_by_biotype": test["species_recall_by_biotype"]},
        "compositional": read_json(root / "compositional.json"), "loso": read_json(root / "leave_one_species_out.json"),
        "heldOutCells": list(HELD_OUT_CELLS),
        "ladder": read_json(mech / "shuffle_ladder.json"), "frame": read_json(mech / "coding_frame.json"),
        "landmarks": read_json(mech / "landmarks.json"),
        "filters": [{key: item[key] for key in ("filter", "pfm", "match", "closest_consensus", "match_cosine", "top_cells",
                                               "information_bits")} for item in filters["filters"][:24]],
        "map": geometry["map"] | {"sequence": [by_id[i] for i in geometry["map"]["id"]]},
        "geometry": {key: geometry[key] for key in ("variance_explained", "ten_neighbour_purity", "purity_chance")},
        "subOntology": read_json(mech / "sub_ontology.json"), "signatures": read_json(mech / "signatures.json"),
        "hypotheses": web_hypotheses(root, manifest), "asthma": web_asthma(root), "literature": library()[["name", "consensus"]].to_dict("records"),
        "figures": {key: {"title": title, "src": figure(root / "figures" / f"{key}.png")} for key, title in GALLERY.items()},
    }
    page = Path(template).read_text().replace(PLACEHOLDER, json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/"))
    Path(out).write_text(page)
