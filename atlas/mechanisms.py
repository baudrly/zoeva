"""What the encoder reads: perturbation, occlusion and probing analyses on held-out windows.

All analyses use the frozen lead model and test chromosomes. They were designed after the benchmark
and are descriptive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.linear_model import RidgeCV
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

from clop2.model import one_hot

from . import genomes
from .benchmark import confusion, macro_f1, nearest, vote_share
from .common import (BIOTYPE_KEYS, HALF, N, SEED, SPECIES, SPECIES_KEYS, WINDOW, dinucleotide_signature, encode,
                     kmer_profile, revcomp, split_of, write_json)
from .corpus import windows
from .features import covered, merged
from .literature import MATCH_MIN, aligned_similarity, library
from .pipeline import Zoeva

PER_CELL = 150
BLOCK, STRIDE = 10, 5
LANDMARK_N = 200
LOGO_REACH = 60
FILTER_TOP = 1000
FILTER_WINDOWS_PER_CELL = 35
K = 15
REDUNDANT_K = 32
TE_MIN_TRAIN, TE_MIN_TEST = 40, 10
# landmark -> (species, feature class, scored cell class, offset of the window centre from the landmark, description)
LANDMARKS = {
    "transcription start site": (("human", "fly", "worm", "arabidopsis"), "promoter", "promoter", 0),
    "start codon": (("yeast", "ecoli"), "promoter", "promoter", None),
    "tRNA 5' end": (("yeast", "fly", "arabidopsis"), "tRNA", "tRNA", 36),
    "exon start (3' splice site)": (("human", "fly", "worm", "arabidopsis"), "CDS", "CDS", 50),
    "exon end (5' splice site)": (("human", "fly", "worm", "arabidopsis"), "CDS", "CDS", -50),
    "3' UTR end (cleavage site)": (("human", "fly", "worm"), "three_prime_UTR", "three_prime_UTR", -100),
}
PROMOTER_OFFSET = {"yeast": 100, "ecoli": 50}  # as in features.yeast / features.ecoli
LOCI = {"2q12 IL1RL1-IL18R1": ("chr2", 102_341_256, "rs1420101", "C", "T"),
        "11q13.5 EMSY-LRRC32": ("chr11", 76_588_150, "rs2155219", "G", "T")}
LOCUS_REACH = 150_000
VARIANT_REACH = 50


def sample_test(manifest: pd.DataFrame, per_cell: int) -> pd.DataFrame:
    test = manifest[manifest["split"] == "test"]
    return test.sample(frac=1, random_state=SEED).groupby("cell").head(per_cell).sort_index()


# ---------------------------------------------------------------- shuffles

def dinucleotide_shuffle(row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random sequence with exactly the same dinucleotide counts and end bases (Altschul & Erickson 1985)."""
    successors = [row[1:][row[:-1] == base] for base in range(4)]
    last = row[-1]
    while True:
        final = {base: int(rng.integers(len(successors[base]))) for base in range(4) if base != last and len(successors[base])}
        reach = {base: successors[base][choice] for base, choice in final.items()}
        if all(_reaches(base, reach, last) for base in reach):
            break
    queues = []
    for base in range(4):
        edges = successors[base]
        if base in final:
            edges = np.r_[rng.permutation(np.delete(edges, final[base])), edges[final[base]]]
        else:
            edges = rng.permutation(edges)
        queues.append(edges.tolist()[::-1])
    out, base = [int(row[0])], int(row[0])
    for _ in range(len(row) - 1):
        base = queues[base].pop()
        out.append(base)
    return np.array(out, dtype=np.uint8)


def _reaches(base: int, reach: dict, last: int) -> bool:
    for _ in range(4):
        if base == last:
            return True
        base = reach.get(base, -1)
    return base == last


def permute_blocks(codes: np.ndarray, width: int, rng: np.random.Generator) -> np.ndarray:
    """Permute consecutive blocks of ``width`` bases within each row (width 1 = mononucleotide shuffle)."""
    blocks = codes.reshape(len(codes), -1, width)
    order = np.argsort(rng.random(blocks.shape[:2]), axis=1)
    return np.take_along_axis(blocks, order[:, :, None], 1).reshape(codes.shape)


def shuffle_ladder(zoeva: Zoeva, manifest: pd.DataFrame) -> dict:
    """Species and class recognition after shuffles that keep progressively less sequence order."""
    rows = sample_test(manifest, PER_CELL)
    codes, rng = encode(rows["sequence"]), np.random.default_rng(SEED)
    truth = zoeva.cells.index(rows["cell"])
    species, biotype = zoeva.cells.species[truth], zoeva.cells.biotype[truth]
    versions = {"intact": codes, "50 bp blocks permuted": permute_blocks(codes, 50, rng),
                "10 bp blocks permuted": permute_blocks(codes, 10, rng),
                "dinucleotide-preserving shuffle": np.stack([dinucleotide_shuffle(row, rng) for row in codes]),
                "mononucleotide shuffle": permute_blocks(codes, 1, rng)}
    out = {"n_windows": int(len(rows)), "versions": {}}
    for name, version in versions.items():
        scores = zoeva.scores(zoeva.embed(version))[0]
        species_hit = (scores @ zoeva.cells.to_species).argmax(1) == species
        own = zoeva.cells.species[None, :] == species[:, None]
        class_hit = zoeva.cells.biotype[np.where(own, scores, -np.inf).argmax(1)] == biotype
        by = lambda hit, keys, labels: {keys[i]: float(hit[labels == i].mean()) for i in np.unique(labels)}
        out["versions"][name] = {
            "species_accuracy": float(species_hit.mean()), "class_given_species_accuracy": float(class_hit.mean()),
            "species_accuracy_by_species": by(species_hit, SPECIES_KEYS, species),
            "species_accuracy_by_class": by(species_hit, BIOTYPE_KEYS, biotype),
            "class_accuracy_by_class": by(class_hit, BIOTYPE_KEYS, biotype)}
    counts = pd.Series(species).map(lambda s: int((zoeva.cells.species == s).sum()))
    out["chance"] = {"species": 1 / len(SPECIES_KEYS), "class_given_species": float((1 / counts).mean())}
    return out


# ---------------------------------------------------------------- reading frame

def coding_frame(zoeva: Zoeva, manifest: pd.DataFrame) -> dict:
    """In windows lying wholly inside one open reading frame (yeast, E. coli), what keeps the "coding" call:
    codon order, codon identity, or only the three-base periodicity of composition?"""
    rng, out = np.random.default_rng(SEED), {}
    cds = BIOTYPE_KEYS.index("CDS")
    for species in ("yeast", "ecoli"):
        rows = manifest[(manifest["cell"] == f"{species}|CDS") & (manifest["split"] == "test")
                        & (manifest["start"] <= manifest["center"] - HALF) & (manifest["end"] >= manifest["center"] + HALF)]
        plus = (rows["strand"] == "+").to_numpy()
        codes = encode(rows["sequence"])
        codes[~plus] = revcomp(codes[~plus])
        # distance from the window's first base (coding orientation) to the next codon boundary
        first = np.where(plus, rows["center"] - HALF - rows["start"], rows["end"] - rows["center"] - HALF)
        phase = (-first) % 3
        frames = np.stack([row[p:p + WINDOW - 2] for row, p in zip(codes, phase)])
        codons = frames.reshape(len(frames), -1, 3)
        stops = np.logical_or.reduce([(codons == stop).all(2) for stop in ((3, 0, 0), (3, 0, 2), (3, 2, 0))]).any(1)
        versions = {
            "intact": frames, "codons permuted": permute_blocks(frames, 3, rng),
            "bases shuffled within codon position": np.stack(
                [permute_blocks(np.ascontiguousarray(codons[:, :, j]), 1, rng) for j in range(3)], axis=2).reshape(frames.shape),
            "triplets permuted out of frame": np.hstack([frames[:, :1], permute_blocks(frames[:, 1:-2], 3, rng), frames[:, -2:]]),
            "dinucleotide-preserving shuffle": np.stack([dinucleotide_shuffle(row, rng) for row in frames]),
            "mononucleotide shuffle": permute_blocks(frames, 1, rng)}
        versions = {name: version[~stops] for name, version in versions.items()}  # drop windows crossing an intron
        report = {}
        for name, version in versions.items():
            within = zoeva.class_given_species(zoeva.scores(zoeva.embed(version))[0], species)
            report[name] = {"coding_call_rate": float((within.argmax(1) == cds).mean()),
                            "mean_coding_probability": float(within[:, cds].mean())}
        out[species] = {"n_windows": int((~stops).sum()), "excluded_with_in_frame_stop": int(stops.sum()), "versions": report}
    return out


# ---------------------------------------------------------------- occlusion at landmarks

def landmark_windows(genome: dict, chrom: np.ndarray, site: np.ndarray, strand: np.ndarray, offset: int) -> np.ndarray:
    """Windows in transcript orientation in which the landmark base sits at index HALF - offset."""
    plus = strand == "+"
    codes = windows(genome, chrom, np.where(plus, site + offset, site + 1 - offset))
    codes[~plus] = revcomp(codes[~plus])
    return codes


def landmark_sites(features: pd.DataFrame, species: str, landmark: str) -> tuple[pd.DataFrame, int]:
    """Rows carrying the landmark, the landmark base (``site``) and the window-centre offset."""
    _, source, _, offset = LANDMARKS[landmark]
    rows = features[features["biotype"] == source].copy()
    sign = np.where(rows["strand"] == "+", 1, -1)
    five, three = np.where(sign > 0, rows["start"], rows["end"] - 1), np.where(sign > 0, rows["end"] - 1, rows["start"])
    if landmark == "start codon":
        offset = -PROMOTER_OFFSET[species]
        rows["site"] = five - sign * offset
    elif landmark.startswith("exon"):
        size = rows["end"] - rows["start"]
        extreme = rows.groupby("group")["start"].transform("min").eq(rows["start"]) | rows.groupby("group")["end"].transform("max").eq(rows["end"])
        keep = ((size >= 100) & (size <= 300) & (~extreme if species != "arabidopsis" else True)).to_numpy()
        rows["site"] = five if "start" in landmark else three
        rows = rows[keep]
    elif landmark.startswith("3' UTR"):
        rows["site"] = three
        rows = rows[(rows["end"] - rows["start"] >= 150).to_numpy()]
        rows = rows[rows["site"].eq(rows.groupby("group")["site"].transform("max" if rows["strand"].iloc[0] == "+" else "min"))
                    | (rows["strand"] != rows["strand"].iloc[0])]
    else:
        rows["site"] = five
    return rows, offset


def occlusion(zoeva: Zoeva, codes: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop in cosine similarity to a cell's text prototype when each BLOCK-wide stretch is replaced by N.
    Returns (n, n_blocks) drops and the intact similarity."""
    starts = np.arange(0, WINDOW - BLOCK + 1, STRIDE)
    masked = np.repeat(codes[:, None, :], len(starts), 1)
    masked[:, np.arange(len(starts))[:, None], starts[:, None] + np.arange(BLOCK)] = N
    intact = zoeva.text_similarity(zoeva.embed(codes))[:, cell]
    occluded = zoeva.text_similarity(zoeva.embed(masked.reshape(-1, WINDOW)))[:, cell].reshape(len(codes), -1)
    return intact[:, None] - occluded, intact


def landmarks(zoeva: Zoeva, results: Path, cache: str) -> dict:
    """Mean occlusion profile around annotated landmarks on test chromosomes, with the base frequencies there."""
    out, axis = {}, np.arange(0, WINDOW - BLOCK + 1, STRIDE) + BLOCK // 2
    for species in SPECIES_KEYS:
        genome = genomes.load(species, cache)
        features = pd.read_csv(results / "features" / f"{species}.tsv.gz", sep="\t", keep_default_na=False)
        features = features[split_of(species, features["chrom"].to_numpy(), features["center"].to_numpy()) == "test"]
        for landmark, (hosts, _, scored, _) in LANDMARKS.items():
            if species not in hosts or f"{species}|{scored}" not in zoeva.cells.names:
                continue
            rows, offset = landmark_sites(features, species, landmark)
            rows = rows.drop_duplicates(["chrom", "site"]).sample(frac=1, random_state=SEED).head(2 * LANDMARK_N)
            codes = landmark_windows(genome, rows["chrom"].to_numpy(), rows["site"].to_numpy(), rows["strand"].to_numpy(), offset)
            codes = codes[(codes < 4).all(1)][:LANDMARK_N]
            drop, intact = occlusion(zoeva, codes, zoeva.cells.names.index(f"{species}|{scored}"))
            at = HALF - offset
            logo = np.stack([(codes[:, at - LOGO_REACH:at + LOGO_REACH + 1] == base).mean(0) for base in range(4)], 1)
            out.setdefault(landmark, {})[species] = {
                "n_windows": int(len(codes)), "position": (axis - at).tolist(), "mean_drop": drop.mean(0).tolist(),
                "sem_drop": (drop.std(0) / np.sqrt(len(drop))).tolist(), "mean_intact_similarity": float(intact.mean()),
                "logo_position": list(range(-LOGO_REACH, LOGO_REACH + 1)), "base_frequency": logo.round(4).tolist()}
    return out


# ---------------------------------------------------------------- first-layer filters

def best_match(pfm: np.ndarray) -> tuple[str, float]:
    """Closest curated literature signal (either strand, any offset) and its background-subtracted cosine."""
    table = library()
    scores = [aligned_similarity(pfm, matrix, True) for matrix in table["matrix"]]
    i = int(np.argmax(scores))
    return table["name"].iloc[i], float(scores[i])


@torch.no_grad()
def filters(zoeva: Zoeva, manifest: pd.DataFrame) -> dict:
    """What each first-layer filter responds to: the frequency matrix of its strongest responses on held-out
    windows, its closest literature consensus, and the cells whose windows excite it most."""
    rows = sample_test(manifest, FILTER_WINDOWS_PER_CELL)
    codes, dna = encode(rows["sequence"]), zoeva.model.dna
    reach = dna.conv1.kernel_size[0] // 2
    activation = dna.norms[0](dna.conv1(one_hot(codes)))[:, :, reach:-reach].numpy()
    cell = zoeva.cells.index(rows["cell"])
    peak = activation.max(2)
    by_cell = np.stack([peak[cell == c].mean(0) for c in range(len(zoeva.cells.names))])
    z = (by_cell - by_cell.mean(0)) / by_cell.std(0)
    out = []
    for f in range(activation.shape[1]):
        flat = np.argpartition(-activation[:, f].ravel(), FILTER_TOP)[:FILTER_TOP]
        row, start = np.unravel_index(flat, activation[:, f].shape)
        kmers = codes[row[:, None], start[:, None] + np.arange(2 * reach + 1)]
        pfm = np.stack([(kmers == base).mean(0) for base in range(4)], 1)
        information = 2 + (pfm * np.log2(np.maximum(pfm, 1e-9))).sum(1)
        name, score = best_match(pfm)
        out.append({"filter": f, "pfm": pfm.round(4).tolist(), "information_bits": float(information.sum()),
                    "match": name if score >= MATCH_MIN else None, "closest_consensus": name, "match_cosine": score,
                    "top_cells": [{"cell": zoeva.cells.names[c], "z": float(z[c, f])} for c in np.argsort(-z[:, f])[:3]]})
    return {"n_windows": int(len(rows)), "top_responses_per_filter": FILTER_TOP, "match_threshold": MATCH_MIN,
            "consensus_library": dict(zip(library()["name"], library()["consensus"])), "filters": sorted(out, key=lambda item: -item["information_bits"])}


# ---------------------------------------------------------------- composition, geometry, sub-ontology

def signatures(manifest: pd.DataFrame, embeddings: np.ndarray) -> dict:
    """Dinucleotide odds ratios per species and class, and how linearly the embedding encodes composition."""
    codes = encode(manifest["sequence"])
    signature = dinucleotide_signature(codes)
    names = [a + b for a in "ACGT" for b in "ACGT"] + ["gc_fraction"]
    test, train = (manifest["split"] == "test").to_numpy(), (manifest["split"] == "train").to_numpy()
    table = pd.DataFrame(signature[test], columns=names).assign(species=manifest["species"].to_numpy()[test],
                                                               cell=manifest["cell"].to_numpy()[test])
    probe = {}
    for name in ("gc_fraction", "CG", "TA"):
        column = names.index(name)
        model = RidgeCV(np.logspace(-3, 2, 6)).fit(embeddings[train], signature[train, column])
        probe[name] = float(model.score(embeddings[test], signature[test, column]))
    return {"columns": names, "by_species": table.drop(columns="cell").groupby("species").mean().round(4).to_dict("index"),
            "by_cell": table.drop(columns="species").groupby("cell")[["gc_fraction", "CG", "TA"]].mean().round(4).to_dict("index"),
            "embedding_linear_probe_r2": probe}


def geometry(zoeva: Zoeva, manifest: pd.DataFrame, embeddings: np.ndarray) -> dict:
    """How the embedding is organised: cell centroids and their clustering, variance explained by species
    and by class, neighbourhood purity against a 4-mer space, and a two-dimensional map."""
    test = np.flatnonzero((manifest["split"] == "test").to_numpy())
    x, cell = embeddings[test], zoeva.cells.index(manifest["cell"].to_numpy()[test])
    groups = {"cell": cell, "species": zoeva.cells.species[cell], "class": zoeva.cells.biotype[cell]}
    total = ((x - x.mean(0)) ** 2).sum()
    explained = {}
    for name, labels in groups.items():
        onehot = np.eye(labels.max() + 1)[labels]
        means = (onehot.T @ x) / np.maximum(onehot.sum(0)[:, None], 1)
        explained[name] = float((onehot.sum(0) * ((means - x.mean(0)) ** 2).sum(1)).sum() / total)
    centroids = np.stack([x[cell == c].mean(0) for c in range(len(zoeva.cells.names))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    tree = linkage(centroids, "average", metric="cosine")
    spaces = {"clop2": x, "kmer4": kmer_profile(encode(manifest["sequence"].to_numpy()[test]), 4)}
    purity = {space: {name: float((labels[nearest(values, values, 11)[0][:, 1:]] == labels[:, None]).mean())
                      for name, labels in groups.items()} for space, values in spaces.items()}
    shown = sample_test(manifest, 90)
    xy = TSNE(2, perplexity=40, random_state=0, init="pca").fit_transform(embeddings[shown.index.to_numpy()])
    return {"variance_explained": explained, "ten_neighbour_purity": purity,
            "purity_chance": {name: float((np.bincount(labels) / len(labels)) @ (np.bincount(labels) / len(labels)))
                              for name, labels in groups.items()},
            "centroid_cosine": (centroids @ centroids.T).round(4).tolist(), "cells": zoeva.cells.names,
            "dendrogram_order": leaves_list(tree).tolist(), "linkage": tree.round(5).tolist(),
            "map": {"id": shown["id"].tolist(), "cell": shown["cell"].tolist(), "x": xy[:, 0].round(2).tolist(),
                    "y": xy[:, 1].round(2).tolist()}}


def sub_ontology(manifest: pd.DataFrame, embeddings: np.ndarray) -> dict:
    """Transposable-element superfamilies of Arabidopsis, which no training text mentions: are they separable
    by nearest neighbours in the embedding, compared with a 4-mer profile?"""
    rows = manifest[(manifest["cell"] == "arabidopsis|transposable_element") & (manifest["detail"] != "")]
    train, test = (rows["split"] == "train").to_numpy(), (rows["split"] == "test").to_numpy()
    counts = rows["detail"][train].value_counts()
    kept = sorted(set(counts[counts >= TE_MIN_TRAIN].index)
                  & set(rows["detail"][test].value_counts().loc[lambda s: s >= TE_MIN_TEST].index))
    usable = rows["detail"].isin(kept).to_numpy()
    labels = np.searchsorted(kept, rows["detail"].to_numpy()[usable])
    train, test = train[usable], test[usable]
    spaces = {"clop2": embeddings[rows.index.to_numpy()[usable]], "kmer4": kmer_profile(encode(rows["sequence"][usable]), 4)}
    out = {"superfamilies": kept, "n_train": int(train.sum()), "n_test": int(test.sum()),
           "majority_accuracy": float(np.bincount(labels[test]).max() / test.sum())}
    for name, values in spaces.items():
        votes = vote_share(*nearest(values[train], values[test], K), labels[train], len(kept), K)
        conf = confusion(labels[test], votes.argmax(1), len(kept))
        out[name] = {"macro_f1": float(macro_f1(conf)), "accuracy": float(np.trace(conf) / conf.sum()),
                     "recall": {kept[i]: float(conf[i, i] / conf[i].sum()) for i in range(len(kept))}}
    return out


# ---------------------------------------------------------------- redundancy across the split

def packed_kmers(codes: np.ndarray, k: int) -> np.ndarray:
    """(n, L - k + 1) uint64 2-bit packing of every k-mer (k <= 32), with the reverse complement folded in by min."""
    width = codes.shape[1] - k + 1
    forward, reverse = np.zeros((len(codes), width), np.uint64), np.zeros((len(codes), width), np.uint64)
    for j in range(k):
        column = codes[:, j:j + width].astype(np.uint64)
        forward = (forward << np.uint64(2)) | column
        reverse |= (np.uint64(3) - column) << np.uint64(2 * j)
    return np.minimum(forward, reverse)


def redundancy(manifest: pd.DataFrame, results: Path) -> dict:
    """Held-out chromosomes do not stop multicopy families (tRNA, snRNA, transposons, pseudogenes) from sharing
    sequence with training windows on other chromosomes. A test window counts as redundant when its central
    REDUNDANT_K bases occur exactly, on either strand, in a training window of the same species. Recall of the
    frozen lead arm is then split by that flag."""
    predictions = pd.read_csv(results / "test_predictions.tsv", sep="\t", keep_default_na=False)
    flags = []
    for species, rows in manifest.groupby("species"):
        train = np.unique(packed_kmers(encode(rows.loc[rows["split"] == "train", "sequence"]), REDUNDANT_K))
        test = rows[rows["split"] == "test"]
        core = packed_kmers(encode(test["sequence"]), REDUNDANT_K)[:, HALF - REDUNDANT_K // 2]
        flags.append(pd.Series(np.isin(core, train), index=test["id"].to_numpy()))
    predictions["redundant"] = predictions["id"].map(pd.concat(flags))
    predictions["correct"] = predictions["cell"] == predictions["predicted_cell"]
    table = predictions.groupby("cell").agg(n=("id", "size"), redundant=("redundant", "mean"),
                                            recall=("correct", "mean")).round(4)
    split = predictions.groupby(["cell", "redundant"])["correct"].agg(["mean", "size"]).unstack()
    return {"k": REDUNDANT_K, "overall_redundant": float(predictions["redundant"].mean()),
            "cell_recall_redundant": float(predictions.loc[predictions["redundant"], "correct"].mean()),
            "cell_recall_unique": float(predictions.loc[~predictions["redundant"], "correct"].mean()),
            "by_cell": table.to_dict("index"),
            "recall_unique_by_cell": {c: (None if np.isnan(v) else float(v)) for c, v in split[("mean", False)].items()},
            "recall_redundant_by_cell": {c: (None if np.isnan(v) else float(v)) for c, v in split[("mean", True)].items()}}


# ---------------------------------------------------------------- disease loci

def human_classes(zoeva: Zoeva) -> tuple[list[str], list[int]]:
    classes = [b for b in BIOTYPE_KEYS if f"human|{b}" in zoeva.cells.names]
    return classes, [BIOTYPE_KEYS.index(b) for b in classes]


def human_profile(zoeva: Zoeva, codes: np.ndarray) -> np.ndarray:
    """P(class | human) per window over the human classes."""
    return zoeva.class_given_species(zoeva.scores(zoeva.embed(codes))[0], "human")[:, human_classes(zoeva)[1]]


def locus_track(zoeva: Zoeva, genome: dict, features: pd.DataFrame, chrom: str, lo: int, hi: int) -> dict:
    """Human class tracks (500 bp tiles every 100 bp) over [lo, hi) and their AUROC against the annotation."""
    classes, _ = human_classes(zoeva)
    centers, scores = zoeva.scan(np.asarray(genome[chrom][lo:hi]))
    centers = centers + lo
    tracks = zoeva.class_given_species(scores, "human")[:, human_classes(zoeva)[1]]
    local = features[(features["chrom"] == chrom) & (features["end"] > lo) & (features["start"] < hi)]
    auroc, labels = {}, {}
    for j, biotype in enumerate(classes):
        own = local[local["biotype"] == biotype]
        if biotype == "promoter":
            own = own.assign(start=own["start"] - HALF, end=own["end"] + HALF)
        if biotype == "intergenic" or own.empty:
            continue
        labels[biotype] = covered(merged(own), np.full(len(centers), chrom), centers)
        if 10 <= labels[biotype].sum() < len(centers):
            auroc[biotype] = {"auroc": float(roc_auc_score(labels[biotype], tracks[:, j])), "n_tiles": int(labels[biotype].sum())}
    return {"chrom": chrom, "classes": classes, "centers": centers.tolist(), "tracks": tracks.round(4).tolist(),
            "annotation": {b: np.flatnonzero(v).tolist() for b, v in labels.items()}, "auroc": auroc}


def variant_window(genome: dict, chrom: str, position: int, ref: str) -> np.ndarray:
    """500 bp centred on a 1-based position, the variant at index HALF; checks the reference allele."""
    window = np.asarray(genome[chrom][position - 1 - HALF:position - 1 + HALF]).copy()
    if "ACGT"[window[HALF]] != ref:
        raise ValueError(f"{chrom}:{position}: reference allele mismatch")
    return window


def nearby_substitutions(window: np.ndarray, reach: int = VARIANT_REACH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every single-base substitution within reach of the centre: (positions, bases, mutated windows)."""
    positions = np.repeat(np.arange(HALF - reach, HALF + reach + 1), 3)
    bases = ((window[positions] + 1 + np.tile(np.arange(3), len(positions) // 3)) % 4).astype(np.uint8)
    mutants = np.repeat(window[None, :], len(positions), 0)
    mutants[np.arange(len(positions)), positions] = bases
    return positions, bases, mutants


def loci(zoeva: Zoeva, results: Path, cache: str) -> dict:
    """Class tracks across two asthma-associated loci on held-out human chromosomes, scored against the
    annotation, and the effect of the lead variant at each. Exploratory."""
    genome = genomes.load("human", cache)
    features = pd.read_csv(results / "features" / "human.tsv.gz", sep="\t", keep_default_na=False)
    classes, _ = human_classes(zoeva)
    out = {}
    for name, (chrom, position, rsid, ref, alt) in LOCI.items():
        site = position - 1
        window = variant_window(genome, chrom, position, ref)
        positions, bases, mutants = nearby_substitutions(window)
        reference, shifted = human_profile(zoeva, window[None, :])[0], human_profile(zoeva, mutants)
        effect = np.abs(shifted - reference).max(1)
        this = np.flatnonzero((positions == HALF) & (bases == "ACGT".index(alt)))[0]
        out[name] = locus_track(zoeva, genome, features, chrom, site - LOCUS_REACH, site + LOCUS_REACH) | {
            "variant": {"rsid": rsid, "position": position, "ref": ref, "alt": alt,
                        "reference_classes": dict(zip(classes, map(float, reference))),
                        "alt_classes": dict(zip(classes, map(float, shifted[this]))),
                        "largest_class_shift": float(effect[this]),
                        "percentile_among_nearby_substitutions": float((effect < effect[this]).mean() * 100),
                        "n_nearby_substitutions": int(len(effect))}}
    return out


def run(results: str, cache: str) -> None:
    root = Path(results)
    out = root / "mechanisms"
    zoeva = Zoeva(results)
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False)
    embeddings = np.load(root / "models" / "clop2_main_s0.embeddings.npz")["dna"]
    for name, payload in (("shuffle_ladder", lambda: shuffle_ladder(zoeva, manifest)),
                          ("coding_frame", lambda: coding_frame(zoeva, manifest)),
                          ("landmarks", lambda: landmarks(zoeva, root, cache)),
                          ("filters", lambda: filters(zoeva, manifest)),
                          ("signatures", lambda: signatures(manifest, embeddings)),
                          ("geometry", lambda: geometry(zoeva, manifest, embeddings)),
                          ("sub_ontology", lambda: sub_ontology(manifest, embeddings)),
                          ("loci", lambda: loci(zoeva, root, cache)),
                          ("redundancy", lambda: redundancy(manifest, root))):
        write_json(out / f"{name}.json", payload())
        print("wrote", name, flush=True)
