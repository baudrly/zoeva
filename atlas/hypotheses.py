"""Automatic hypothesis generation and testing (Zoeva's original purpose), checked against the literature.

1. Generate, on training chromosomes only. For every species x class cell, rank canonical 6-mers by how much the
   model relies on them (gradient x input attribution summed over each occurrence) and by how enriched they are
   in the cell against the species' other cells. Keep a few non-redundant seeds per cell and turn each into a
   frequency matrix from its occurrences. Separately, propose "composition" hypotheses for cells whose calls
   survive a dinucleotide-preserving shuffle.
2. Register: write every hypothesis with its generation statistics and a hash before any test.
3. Test, on validation and test chromosomes separately. Motif hypotheses: necessity (randomise the occurrences;
   control randomises as many 6 bp blocks elsewhere in the same windows), sufficiency (implant the seed into the
   species' other windows; control implants a letter-shuffled seed), held-out enrichment (Fisher). Composition
   hypotheses: call after a dinucleotide-preserving shuffle against a mononucleotide shuffle. Benjamini-Hochberg
   per test family and split.
4. Check: compare each matrix with curated literature signals (and, for annotation, JASPAR), and score how many
   signals the literature expects in a cell the pipeline recovers.

Queries get the same treatment one sequence at a time (``explain``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact, wilcoxon

from clop2.model import one_hot

from .common import (BIOTYPE_KEYS, HALF, SEED, SPECIES_KEYS, WINDOW, canonical_map, decode, encode, file_sha256,
                     kmer_counts, read_json, revcomp, sha256, write_json)
from .literature import JASPAR_MIN, LIBRARY_PATH, jaspar_match, library, literature_match
from .mechanisms import dinucleotide_shuffle, permute_blocks
from .pipeline import Zoeva

K = 6
ATTRIBUTION_WINDOWS = 300
MIN_WINDOWS = 25
TOP_PER_CELL = 3
CANDIDATES = 40
FLANK = 4
TRIM_BITS = 0.3
TEST_WINDOWS = 200
MIN_TEST_WINDOWS = 15
COMPOSITION_WINDOWS = 150
COMPOSITION_RETENTION = 0.8
IMPLANT_REACH = 50
FDR = 0.05
QUERY_DRAWS, QUERY_NULL = 8, 40
POSITION_BINS = np.arange(-250, 251, 25)
SPLITS = ("validation", "test")
NULL_PER_CELL = 20
BATCH = 64


# ---------------------------------------------------------------- k-mer bookkeeping

def kmer_ids(codes: np.ndarray, canonical: bool = True) -> np.ndarray:
    """(n, L - K + 1) k-mer index per position (canonical dense ids by default); -1 where a base is not ACGT."""
    width = codes.shape[1] - K + 1
    raw, valid = np.zeros((len(codes), width), np.int64), np.ones((len(codes), width), bool)
    for j in range(K):
        column = codes[:, j:j + width]
        raw, valid = (raw << 2) | np.minimum(column, 3), valid & (column < 4)
    return np.where(valid, canonical_map(K)[0][raw] if canonical else raw, -1)


def seed_codes(seed: str) -> np.ndarray:
    return encode([seed])[0]


def occurrences(codes: np.ndarray, seed: str) -> np.ndarray:
    """(n, L - K + 1) mask of positions where the seed or its reverse complement starts."""
    forward = kmer_ids(codes, canonical=False)
    target = [int("".join(map(str, c)), 4) for c in (seed_codes(seed), revcomp(seed_codes(seed)))]
    return np.isin(forward, target)


def similar(a: str, b: str) -> bool:
    """Seeds that overlap by at least four bases with at most one mismatch, on either strand, at shifts up to 2."""
    for other in (b, decode(revcomp(seed_codes(b)))[0]):
        for shift in range(-2, 3):
            pairs = [(a[i], other[i - shift]) for i in range(len(a)) if 0 <= i - shift < len(other)]
            if len(pairs) >= 4 and sum(x != y for x, y in pairs) <= 1:
                return True
    return False


def frequency_matrix(codes: np.ndarray, seed: str) -> np.ndarray:
    """Seed-anchored frequency matrix (seed forward, FLANK bases each side), trimmed of uninformative ends."""
    hits = occurrences(codes, seed)
    rows, starts = np.nonzero(hits)
    forward = (codes[rows[:, None], starts[:, None] + np.arange(K)] == seed_codes(seed)).all(1)
    keep = (starts >= FLANK) & (starts + K + FLANK <= codes.shape[1])
    rows, starts, forward = rows[keep], starts[keep], forward[keep]
    stretch = codes[rows[:, None], starts[:, None] + np.arange(-FLANK, K + FLANK)]
    stretch[~forward] = revcomp(stretch[~forward])
    pfm = np.stack([(stretch == b).sum(0) for b in range(4)], 1) + 0.25
    pfm /= pfm.sum(1, keepdims=True)
    bits = 2 + (pfm * np.log2(pfm)).sum(1)
    informative = np.flatnonzero(bits >= TRIM_BITS)
    lo, hi = min(informative.min(), FLANK), max(informative.max(), FLANK + K - 1)
    return pfm[lo:hi + 1]


# ---------------------------------------------------------------- model readouts

def class_probability(zoeva: Zoeva, codes: np.ndarray, species: str, biotype: str) -> np.ndarray:
    """Zoeva's probability of the class given the species, per window."""
    return zoeva.class_given_species(zoeva.scores(zoeva.embed(codes))[0], species)[:, BIOTYPE_KEYS.index(biotype)]


def attribution(zoeva: Zoeva, codes: np.ndarray, cell: int) -> np.ndarray:
    """Gradient x input of the cosine similarity to a cell's text prototype, per base (n, L)."""
    zoeva.model.eval()
    prototype = torch.from_numpy(zoeva.prototypes[cell])
    out = []
    for lo in range(0, len(codes), BATCH):
        x = one_hot(codes[lo:lo + BATCH]).requires_grad_(True)
        (zoeva.model.dna(x) @ prototype).sum().backward()
        out.append((x.grad * x).sum(1).detach().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- generation

def generate(zoeva: Zoeva, manifest: pd.DataFrame) -> list[dict]:
    rng, out = np.random.default_rng(SEED), []
    train = manifest[manifest["split"] == "train"]
    size = canonical_map(K)[1]
    for species, rows in train.groupby("species"):
        codes = encode(rows["sequence"])
        present = kmer_counts(codes, K) > 0
        cells = rows["cell"].to_numpy()
        for cell in sorted(set(cells)):
            biotype = cell.split("|")[1]
            own = cells == cell
            a, b = present[own].sum(0), present[~own].sum(0)
            log_odds = np.log2((a + 0.5) / (own.sum() - a + 0.5)) - np.log2((b + 0.5) / ((~own).sum() - b + 0.5))
            sample = rng.choice(np.flatnonzero(own), min(ATTRIBUTION_WINDOWS, own.sum()), replace=False)
            scores = attribution(zoeva, codes[sample], zoeva.cells.names.index(cell))
            ids = kmer_ids(codes[sample])
            summed = np.cumsum(np.pad(scores, ((0, 0), (1, 0))), 1)
            occurrence = summed[:, K:] - summed[:, :-K]
            valid = ids >= 0
            totals = np.bincount(ids[valid], occurrence[valid], size)
            counts = np.bincount(ids[valid], minlength=size)
            mean = totals / np.maximum(counts, 1)
            z = (mean - occurrence[valid].mean()) / occurrence[valid].std()
            score = np.where((a >= MIN_WINDOWS) & (log_odds > 0) & (z > 0), z * log_odds, 0)
            accepted = []
            for kmer in np.argsort(-score)[:CANDIDATES]:
                if score[kmer] <= 0 or len(accepted) == TOP_PER_CELL:
                    break
                seed = canonical_kmer(kmer)
                if any(similar(seed, other["seed"]) for other in accepted):
                    continue
                hits = occurrences(codes[own], seed)
                rows_hit, starts = np.nonzero(hits)
                strand = rows["strand"].to_numpy()[own][rows_hit]
                offset = np.where(strand == "-", -1, 1) * (starts + K / 2 - HALF)
                accepted.append({
                    "id": f"{cell}:{seed}", "family": "motif", "cell": cell, "species": species, "biotype": biotype,
                    "seed": seed, "pfm": frequency_matrix(codes[own], seed).round(4).tolist(),
                    "train_windows_with_seed": int(a[kmer]), "train_cell_windows": int(own.sum()),
                    "train_log2_odds": float(log_odds[kmer]), "attribution_z": float(z[kmer]), "score": float(score[kmer]),
                    "position_histogram": np.histogram(offset, POSITION_BINS)[0].tolist(),
                    "central_100bp_share": float((np.abs(offset) <= 50).mean()),
                    "statement": f"In {species} {biotype} windows the model relies on {seed}: it is enriched over the "
                                 f"species' other classes and carries positive attribution."})
            out += accepted
        out += composition_candidates(zoeva, rows, codes, rng)
    return out


def annotate(catalogue: list[dict]) -> None:
    """Literature check of every motif hypothesis (closest curated signal; JASPAR for annotation)."""
    for item in catalogue:
        if item["family"] == "motif":
            item["literature"] = literature_match(np.asarray(item["pfm"]), item["species"], item["biotype"])
            item["jaspar"] = jaspar_match(np.asarray(item["pfm"]), item["species"])


def canonical_kmer(index: int) -> str:
    """The lexicographically smaller strand of a canonical k-mer id."""
    dense, _ = canonical_map(K)
    raw = int(np.flatnonzero(dense == index)[0])
    return "".join("ACGT"[(raw >> (2 * (K - 1 - j))) & 3] for j in range(K))


def composition_candidates(zoeva: Zoeva, rows: pd.DataFrame, codes: np.ndarray, rng) -> list[dict]:
    """Cells whose class call on training windows survives a dinucleotide-preserving shuffle."""
    out, species = [], rows["species"].iloc[0]
    n_classes = rows["biotype"].nunique()
    for cell, part in rows.groupby("cell"):
        biotype = cell.split("|")[1]
        sample = codes[rng.choice(np.flatnonzero((rows["cell"] == cell).to_numpy()), min(COMPOSITION_WINDOWS, len(part)), replace=False)]
        intact = class_probability(zoeva, sample, species, biotype).mean()
        shuffled = class_probability(zoeva, np.stack([dinucleotide_shuffle(r, rng) for r in sample]), species, biotype).mean()
        if intact > 1 / n_classes and shuffled / intact >= COMPOSITION_RETENTION:
            out.append({"id": f"{cell}:composition", "family": "composition", "cell": cell, "species": species,
                        "biotype": biotype, "train_intact": float(intact), "train_dinucleotide_shuffled": float(shuffled),
                        "statement": f"The {species} {biotype} call rests on dinucleotide composition: it survives a "
                                     "dinucleotide-preserving shuffle."})
    return out


# ---------------------------------------------------------------- tests

def randomise_blocks(codes: np.ndarray, rows: np.ndarray, starts: np.ndarray, seed: str, rng) -> np.ndarray:
    """Replace K-base blocks with random bases, redrawing any block that recreates the seed."""
    out = codes.copy()
    banned = {seed, decode(revcomp(seed_codes(seed)))[0]}
    for r, s in zip(rows, starts):
        while True:
            block = rng.integers(0, 4, K).astype(np.uint8)
            if decode(block)[0] not in banned:
                out[r, s:s + K] = block
                break
    return out


def control_starts(hits: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """As many K-base blocks per row as the row has occurrences, placed away from every occurrence."""
    rows, starts = [], []
    for r in range(len(hits)):
        n = int(hits[r].sum())
        blocked = np.convolve(hits[r], np.ones(2 * K - 1), "same") > 0
        free = np.flatnonzero(~blocked)
        chosen = rng.choice(free, min(n, len(free)), replace=False)
        rows += [r] * len(chosen)
        starts += chosen.tolist()
    return np.array(rows, int), np.array(starts, int)


def one_sided(differences: np.ndarray) -> float:
    return float(wilcoxon(differences, alternative="greater").pvalue) if np.any(differences != 0) else 1.0


def enrichment(present: np.ndarray, own: np.ndarray) -> dict:
    """One-sided Fisher test of seed presence in the cell against the species' other windows; the odds ratio is
    Haldane-corrected (+0.5 per cell) so that it stays finite."""
    table = np.array([[(own & present).sum(), (own & ~present).sum()], [(~own & present).sum(), (~own & ~present).sum()]])
    _, p = fisher_exact(table, alternative="greater")
    a, b, c, d = table.ravel() + 0.5
    return {"windows_with_seed": int(table[0, 0]), "cell_windows": int(own.sum()), "contingency": table.tolist(),
            "odds_ratio": float(a * d / (b * c)), "p_enrichment": float(p)}


def test_motif(zoeva: Zoeva, hypothesis: dict, split_rows: pd.DataFrame, rng) -> dict:
    species, biotype, seed = hypothesis["species"], hypothesis["biotype"], hypothesis["seed"]
    rows = split_rows[split_rows["species"] == species]
    codes = encode(rows["sequence"])
    present = occurrences(codes, seed).any(1)
    own = (rows["cell"] == hypothesis["cell"]).to_numpy()
    result = enrichment(present, own)
    carriers = np.flatnonzero(own & present)
    if len(carriers) < MIN_TEST_WINDOWS:
        return result | {"testable": False}
    carriers = rng.choice(carriers, min(TEST_WINDOWS, len(carriers)), replace=False)
    sample = codes[carriers]
    hits = occurrences(sample, seed)
    intact = class_probability(zoeva, sample, species, biotype)
    mutated = class_probability(zoeva, randomise_blocks(sample, *np.nonzero(hits), seed, rng), species, biotype)
    control = class_probability(zoeva, randomise_blocks(sample, *control_starts(hits, rng), seed, rng), species, biotype)
    necessity = control - mutated
    recipients = np.flatnonzero(~own & ~present)
    recipients = codes[rng.choice(recipients, min(TEST_WINDOWS, len(recipients)), replace=False)]
    where = HALF - K // 2 + rng.integers(-IMPLANT_REACH, IMPLANT_REACH + 1, len(recipients))
    flip = rng.random(len(recipients)) < 0.5
    motif = np.where(flip[:, None], revcomp(seed_codes(seed))[None, :], seed_codes(seed)[None, :])
    scrambled = permute_blocks(motif, 1, rng)
    implanted, decoy = recipients.copy(), recipients.copy()
    span = where[:, None] + np.arange(K)
    np.put_along_axis(implanted, span, motif, 1)
    np.put_along_axis(decoy, span, scrambled, 1)
    sufficiency = class_probability(zoeva, implanted, species, biotype) - class_probability(zoeva, decoy, species, biotype)
    return result | {
        "testable": True, "n_tested": int(len(sample)), "intact_probability": float(intact.mean()),
        "necessity_effect": float(necessity.mean()), "necessity_relative": float(necessity.mean() / max(intact.mean(), 1e-9)),
        "p_necessity": one_sided(necessity), "n_recipients": int(len(recipients)),
        "sufficiency_effect": float(sufficiency.mean()), "p_sufficiency": one_sided(sufficiency)}


def test_composition(zoeva: Zoeva, hypothesis: dict, split_rows: pd.DataFrame, rng) -> dict:
    rows = split_rows[split_rows["cell"] == hypothesis["cell"]]
    sample = encode(rows["sequence"].to_numpy()[rng.choice(len(rows), min(TEST_WINDOWS, len(rows)), replace=False)])
    probability = lambda codes: class_probability(zoeva, codes, hypothesis["species"], hypothesis["biotype"])
    intact = probability(sample)
    di = probability(np.stack([dinucleotide_shuffle(r, rng) for r in sample]))
    mono = probability(permute_blocks(sample, 1, rng))
    return {"testable": True, "n_tested": int(len(sample)), "intact_probability": float(intact.mean()),
            "retention_dinucleotide": float(di.mean() / max(intact.mean(), 1e-9)),
            "retention_mononucleotide": float(mono.mean() / max(intact.mean(), 1e-9)),
            "composition_effect": float((di - mono).mean()), "p_composition": one_sided(di - mono)}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def adjust(results: list[dict], key: str) -> None:
    rows = [r for r in results if key in r]
    if rows:
        for r, q in zip(rows, benjamini_hochberg(np.array([r[key] for r in rows]))):
            r[key.replace("p_", "q_")] = float(q)


def passed(result: dict, family: str) -> bool:
    if not result.get("testable"):
        return False
    if family == "composition":
        return result["q_composition"] < FDR and result["retention_dinucleotide"] >= COMPOSITION_RETENTION
    return result["q_necessity"] < FDR and result["necessity_effect"] > 0 and result["q_enrichment"] < FDR


def verdict(item: dict) -> str:
    hits = sum(passed(item["tests"][split], item["family"]) for split in SPLITS)
    return ("replicated", "one split only", "not supported")[2 - hits]


def status(item: dict) -> str:
    """Where a supported hypothesis stands against the literature."""
    if item["family"] == "composition":
        return "composition"
    lit = item["literature"]
    if lit["name"] is None:
        return "novel lead"
    return "recovers known signal" if lit["expected_in_cell"] else "known signal, new context"


# ---------------------------------------------------------------- literature recall

def chance_matches(manifest: pd.DataFrame, per_cell: int = NULL_PER_CELL) -> dict:
    """How often a matrix built around a random 6-mer of a cell (present in at least MIN_WINDOWS training windows)
    matches a curated signal at all, and matches one the literature expects in that cell: the rate a hypothesis
    would reach by chance."""
    rng, any_match, expected, expected_any = np.random.default_rng(SEED + 2), [], [], []
    train = manifest[manifest["split"] == "train"]
    for cell, rows in train.groupby("cell"):
        species, biotype = cell.split("|")
        codes = encode(rows["sequence"])
        frequent = np.flatnonzero((kmer_counts(codes, K) > 0).sum(0) >= MIN_WINDOWS)
        for kmer in rng.choice(frequent, min(per_cell, len(frequent)), replace=False):
            match = literature_match(frequency_matrix(codes, canonical_kmer(kmer)), species, biotype)
            any_match.append(match["name"] is not None)
            expected.append(match["expected_in_cell"])
            expected_any.append(match["expected_signal_above_threshold"])
    return {"n_random_seeds": len(any_match), "any_signal": float(np.mean(any_match)),
            "expected_signal": float(np.mean(expected)), "expected_signal_any_above_threshold": float(np.mean(expected_any))}


def expected_recall(catalogue: list[dict], cells: list[str]) -> list[dict]:
    """For every (signal, cell) pair the literature predicts, whether a hypothesis matching it was generated,
    and whether it was replicated."""
    out = []
    for _, row in library().iterrows():
        for species in row["species"]:
            for biotype in row["classes"]:
                cell = f"{species}|{biotype}"
                if cell not in cells:
                    continue
                matching = [h for h in catalogue if h["family"] == "motif" and h["cell"] == cell and h["literature"]["name"] == row["name"]]
                out.append({"signal": row["name"], "cell": cell, "references": row["references"],
                            "generated": bool(matching), "replicated": any(h["verdict"] == "replicated" for h in matching),
                            "hypotheses": [h["id"] for h in matching]})
    return out


# ---------------------------------------------------------------- driver

def run(results: str) -> dict:
    root = Path(results)
    out = root / "hypotheses"
    out.mkdir(parents=True, exist_ok=True)
    zoeva = Zoeva(results)
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", keep_default_na=False)
    registered = out / "registered.json"
    if not registered.exists():
        catalogue = generate(zoeva, manifest)
        write_json(registered, {"generated_on": "train", "model": "clop2_main_s0", "k": K, "top_per_cell": TOP_PER_CELL,
                                "manifest_sha256": file_sha256(root / "manifest.tsv"), "hypotheses": catalogue})
    catalogue = read_json(registered)["hypotheses"]
    previous = out / "tested.json"
    if previous.exists() and read_json(previous)["summary"]["registered_sha256"] == file_sha256(registered):
        for h, old in zip(catalogue, read_json(previous)["hypotheses"]):  # tests are deterministic; reuse them
            h["tests"] = old["tests"]
    else:
        rng = np.random.default_rng(SEED + 1)
        for split in SPLITS:
            split_rows = manifest[manifest["split"] == split]
            tests = [(test_motif if h["family"] == "motif" else test_composition)(zoeva, h, split_rows, rng) for h in catalogue]
            for key in ("p_necessity", "p_sufficiency", "p_enrichment", "p_composition"):
                adjust(tests, key)
            for h, t in zip(catalogue, tests):
                h.setdefault("tests", {})[split] = t
    annotate(catalogue)
    for h in catalogue:
        h["verdict"] = verdict(h)
        h["status"] = status(h)
        if h["family"] == "motif":
            h["sufficient"] = all(h["tests"][s].get("q_sufficiency", 1) < FDR and h["tests"][s].get("sufficiency_effect", 0) > 0
                                  for s in SPLITS)
    recall = expected_recall(catalogue, sorted(manifest["cell"].unique()))
    chance = chance_matches(manifest)
    motifs = [h for h in catalogue if h["family"] == "motif"]
    replicated_motifs = [h for h in motifs if h["verdict"] == "replicated"]
    supported = [h for h in catalogue if h["verdict"] == "replicated"]
    summary = {
        "registered_sha256": file_sha256(registered), "n_hypotheses": len(catalogue),
        "n_motif": sum(h["family"] == "motif" for h in catalogue),
        "n_composition": sum(h["family"] == "composition" for h in catalogue),
        "verdicts": pd.Series([h["verdict"] for h in catalogue]).value_counts().to_dict(),
        "replicated_by_status": pd.Series([h["status"] for h in supported]).value_counts().to_dict(),
        "replicated_sufficient": sum(h.get("sufficient", False) for h in supported),
        "expected_pairs": len(recall), "expected_generated": sum(r["generated"] for r in recall),
        "expected_replicated": sum(r["replicated"] for r in recall),
        "fdr": FDR, "splits": list(SPLITS), "chance_match": chance, "literature_library_sha256": file_sha256(LIBRARY_PATH),
        "generated_any_signal": float(np.mean([h["literature"]["name"] is not None for h in motifs])),
        "generated_expected_signal": float(np.mean([h["literature"]["expected_in_cell"] for h in motifs])),
        "replicated_any_signal": float(np.mean([h["literature"]["name"] is not None for h in replicated_motifs])) if replicated_motifs else None,
        "replicated_expected_signal": float(np.mean([h["literature"]["expected_in_cell"] for h in replicated_motifs])) if replicated_motifs else None,
        "generated_expected_any_above_threshold": float(np.mean([h["literature"]["expected_signal_above_threshold"] for h in motifs])),
        "replicated_expected_any_above_threshold": float(np.mean([h["literature"]["expected_signal_above_threshold"]
                                                                  for h in replicated_motifs])) if replicated_motifs else None,
        "jaspar_threshold": JASPAR_MIN}
    write_json(out / "tested.json", {"summary": summary, "hypotheses": catalogue, "expected_recall": recall})
    (out / "report.md").write_text(report(summary, catalogue, recall))
    return summary


def report(summary: dict, catalogue: list[dict], recall: list[dict]) -> str:
    lines = ["# Zoeva automatic hypotheses", "",
             f"{summary['n_hypotheses']} hypotheses generated on training chromosomes and registered "
             f"(sha256 {summary['registered_sha256'][:12]}), tested on validation and test chromosomes "
             f"(Benjamini-Hochberg, FDR {summary['fdr']}). Exploratory.", "",
             f"Literature baseline: {summary['expected_generated']} of {summary['expected_pairs']} signal-cell pairs the "
             f"literature predicts were generated, {summary['expected_replicated']} replicated. A matrix built around a random "
             f"frequent 6-mer matches some curated signal {100 * summary['chance_match']['any_signal']:.0f}% of the time and one "
             f"expected in its cell {100 * summary['chance_match']['expected_signal']:.0f}% of the time; generated hypotheses "
             f"{100 * summary['generated_any_signal']:.0f}% and {100 * summary['generated_expected_signal']:.0f}%.", "",
             "| Hypothesis | Verdict | Literature | Necessity (test) | Enrichment OR (test) | Sufficient |", "|---|---|---|---|---|---|"]
    order = {"replicated": 0, "one split only": 1, "not supported": 2}
    for h in sorted(catalogue, key=lambda h: (order[h["verdict"]], h["cell"])):
        t = h["tests"]["test"]
        if h["family"] == "motif":
            lit = h["literature"]["name"] or f"none (closest {h['literature']['closest']} {h['literature']['similarity']:.2f})"
            effect = f"{t['necessity_effect']:+.3f}" if t.get("testable") else "untestable"
            lines.append(f"| {h['cell']} {h['seed']} | {h['verdict']} | {lit} ({h['status']}) | {effect} | "
                         f"{t['odds_ratio']:.2f} | {'yes' if h.get('sufficient') else 'no'} |")
        else:
            lines.append(f"| {h['cell']} composition | {h['verdict']} | composition | retention {t['retention_dinucleotide']:.2f} "
                         f"(mono {t['retention_mononucleotide']:.2f}) | | |")
    lines += ["", "## Signals the literature predicts", "", "| Signal | Cell | Generated | Replicated |", "|---|---|---|---|"]
    lines += [f"| {r['signal']} | {r['cell']} | {'yes' if r['generated'] else 'no'} | {'yes' if r['replicated'] else 'no'} |" for r in recall]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- one query

def explain(zoeva: Zoeva, sequence: str, catalogue: list[dict], rng=None) -> dict:
    """Testable hypotheses for one sequence: which replicated motifs of its top cell it carries, how much the
    class call drops when they are randomised, and where that drop falls among drops from randomising as
    many random blocks; plus whether the call survives a dinucleotide-preserving shuffle."""
    rng = rng or np.random.default_rng(SEED)
    report = zoeva.hypothesize({"query": sequence})[0]
    species = report["species"][0]["label"]
    biotype = report["sequence_class_given_top_species"][0]["label"]
    codes = encode([sequence[max(0, (len(sequence) - WINDOW) // 2):][:WINDOW]])
    probability = lambda batch: class_probability(zoeva, batch, species, biotype)
    intact = float(probability(codes)[0])
    tested = []
    for h in catalogue:
        if h["cell"] != f"{species}|{biotype}" or h["verdict"] == "not supported":
            continue
        if h["family"] == "composition":
            shuffled = probability(np.stack([dinucleotide_shuffle(codes[0], rng) for _ in range(QUERY_DRAWS)]))
            tested.append({"id": h["id"], "family": "composition", "retention": float(shuffled.mean() / max(intact, 1e-9))})
            continue
        hits = occurrences(codes, h["seed"])
        if not hits.any():
            tested.append({"id": h["id"], "family": "motif", "present": False, "literature": h["literature"]["name"]})
            continue
        drop = intact - probability(np.concatenate([randomise_blocks(codes, *np.nonzero(hits), h["seed"], rng)
                                                    for _ in range(QUERY_DRAWS)])).mean()
        null = intact - probability(np.concatenate([randomise_blocks(codes, *control_starts(hits, rng), h["seed"], rng)
                                                    for _ in range(QUERY_NULL)]))
        tested.append({"id": h["id"], "family": "motif", "present": True, "occurrences": np.flatnonzero(hits[0]).tolist(),
                       "drop": float(drop), "null_percentile": float((null < drop).mean() * 100),
                       "literature": h["literature"]["name"], "status": h["status"]})
    return {"species": species, "class": biotype, "class_probability": intact, "hypotheses": tested,
            "query_sha256": sha256(sequence)}
