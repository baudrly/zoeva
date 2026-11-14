"""Exploratory asthma mechanisms, starting from the variants of Eliasen et al. 2022 (CDHR3 x GSDMB epistasis in
early severe childhood asthma) and the functional 17q21 variants around them.

For each variant, both alleles are read the same way Zoeva reads any sequence, and every readout is ranked
against all single-base substitutions within 50 bp of the variant:
  model     shift in P(class | human), attribution at the variant base, hypothesis seeds the allele makes or breaks,
            and the per-sequence hypothesis tests of ``hypotheses.explain`` on each allele;
  motifs    best log-odds score of every JASPAR 2026 vertebrate profile and every curated literature signal on
            the windows covering the variant, risk allele minus other allele;
  pairs     for the interacting pairs, whether the two risk alleles move their windows in the same direction in
            embedding and class space, against pairs of random substitutions at the two loci.
Literature check: where a mechanism is published (CTCF switching at rs12936231 and rs4065275, the GSDMB splice
acceptor at rs11078928), does the sequence readout reproduce its direction? Most of these loci lie on training
chromosomes, so model readouts are in-sample there; motif readouts do not depend on the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from clop2.model import one_hot

from . import genomes
from .common import HALF, SEED, SPECIES, decode, read_json, revcomp, write_json
from .hypotheses import attribution, explain, occurrences
from .literature import jaspar_profiles, library
from .mechanisms import human_classes, human_profile, locus_track, nearby_substitutions, variant_window
from .pipeline import Zoeva

VARIANTS = Path(__file__).resolve().parent.parent / "data" / "asthma_variants.tsv"
PAIRS = (("rs6967330", "rs2305480", "replicated interaction"), ("rs6967330", "rs12936231", "similar interaction reported"),
         ("rs7758229", "rs2305480", "not replicated"), ("rs2230641", "rs2280134", "not replicated"))
LOCI = {"CDHR3 (7q22.3)": ("chr7", 105_868_000, 106_168_000), "17q12-21 (IKZF3-ZPBP2-GSDMB-ORMDL3)": ("chr17", 39_740_000, 40_040_000)}
BACKGROUND = np.array([0.295, 0.205, 0.205, 0.295])  # human genome base composition
PSEUDO = 0.01
MOTIF_PRESENT = 0.85   # relative score at which a motif counts as present
STRONG = 90            # percentile among nearby substitutions for a "strong" change
TOP_MOTIFS = 8
PAIR_NULL = 4000
MOTIF_REACH = 50
LEAD_PERCENTILE = 95
# Transcription factor families with established roles in T-cell or innate immune programmes (name prefixes).
IMMUNE_FAMILIES = ("NFAT", "STAT", "IRF", "RUNX", "ETS", "ELF", "ETV", "SPI1", "IKZF", "ROR", "NFKB", "REL", "JUN", "FOS",
                   "BATF", "GATA3", "TBX21", "EOMES", "FOXP3", "MAF", "BCL6", "PRDM1", "TCF7", "LEF1")


# ---------------------------------------------------------------- motif scanning

def motif_bank() -> tuple[list[str], list[str], torch.Tensor, np.ndarray, np.ndarray]:
    """Log-odds matrices of JASPAR vertebrate profiles and curated literature signals, padded to one width.
    Returns names, sources, weights (M, 4, W), widths, and (min, max) attainable score per motif."""
    names, sources, matrices = [], [], []
    for name, identifier, frequency in jaspar_profiles("vertebrates"):
        names.append(f"{name} ({identifier})")
        sources.append("JASPAR")
        matrices.append(frequency)
    for _, row in library().iterrows():
        names.append(row["name"])
        sources.append("literature")
        matrices.append(row["matrix"])
    width = max(len(m) for m in matrices)
    weights = np.zeros((len(matrices), 4, width), np.float32)
    widths, bounds = np.array([len(m) for m in matrices]), np.zeros((len(matrices), 2))
    for i, m in enumerate(matrices):
        log_odds = np.log2((m + PSEUDO) / (1 + 4 * PSEUDO) / BACKGROUND)
        weights[i, :, :len(m)] = log_odds.T
        bounds[i] = log_odds.min(1).sum(), log_odds.max(1).sum()
    return names, sources, torch.from_numpy(weights), widths, bounds


@torch.no_grad()
def covering_scores(codes: np.ndarray, weights: torch.Tensor, widths: np.ndarray) -> np.ndarray:
    """Best score per window and motif over every placement (either strand) that covers the centre base."""
    width = weights.shape[2]
    lo = HALF - width + 1
    segment = one_hot(np.ascontiguousarray(codes[:, lo:HALF + width]))
    both = torch.cat([weights, weights.flip(1, 2)])  # reverse-complement matrices are right-aligned after the flip
    scores = F.conv1d(segment, both).numpy()          # (n, 2M, positions); start of placement = lo + index
    starts = lo + np.arange(scores.shape[2])
    m = len(widths)
    shift = width - widths                             # flipped matrices begin `shift` columns into the window
    forward_ok = (starts[None, :] <= HALF) & (starts[None, :] + widths[:, None] > HALF)
    reverse_ok = (starts[None, :] + shift[:, None] <= HALF) & (starts[None, :] + width > HALF)
    masked = np.where(np.concatenate([forward_ok, reverse_ok])[None], scores, -np.inf)
    return np.maximum(masked[:, :m].max(2), masked[:, m:].max(2))


def recentred_substitutions(context: np.ndarray, reach: int) -> tuple[np.ndarray, np.ndarray]:
    """Every substitution within reach of the variant other than at the variant itself, each in a window centred
    on the substituted base, with the matching unmutated windows. ``context`` is WINDOW + 2 reach bases long."""
    offsets = np.repeat(np.concatenate([np.arange(-reach, 0), np.arange(1, reach + 1)]), 3)
    starts = reach + offsets
    windows = context[starts[:, None] + np.arange(2 * HALF)]
    mutated = windows.copy()
    mutated[:, HALF] = (windows[:, HALF] + 1 + np.tile(np.arange(3), len(offsets) // 3)) % 4
    return mutated, windows


def motif_readout(ref: np.ndarray, alt: np.ndarray, context: np.ndarray, bank, risk_is_alt: bool):
    """Relative scores of both alleles, risk-minus-other log-odds differences, and the same differences for every
    nearby substitution scored on placements covering its own base."""
    names, sources, weights, widths, bounds = bank
    relative = lambda s: (s - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    scores = covering_scores(np.stack([ref, alt]), weights, widths)
    mutated, unmutated = recentred_substitutions(context, MOTIF_REACH)
    after, before = covering_scores(mutated, weights, widths), covering_scores(unmutated, weights, widths)
    null_present = np.maximum(relative(after), relative(before)) >= MOTIF_PRESENT
    sign = 1 if risk_is_alt else -1
    return relative(scores), sign * (scores[1] - scores[0]), after - before, null_present


# ---------------------------------------------------------------- one variant

def variant_report(zoeva: Zoeva, genome: dict, row: pd.Series, catalogue: list[dict], bank) -> dict:
    classes, _ = human_classes(zoeva)
    ref = variant_window(genome, row["chrom"], int(row["pos"]), row["ref"])
    alt = ref.copy()
    alt[HALF] = "ACGT".index(row["alt"])
    positions, bases, mutants = nearby_substitutions(ref)
    elsewhere = positions != HALF
    risk_is_alt = row["risk"] == row["alt"]
    risk, other = (alt, ref) if risk_is_alt else (ref, alt)

    # model: class profile, embedding shift, attribution, seeds, per-allele hypothesis tests
    profiles = human_profile(zoeva, np.stack([risk, other]))
    shift = profiles[0] - profiles[1]
    null_profiles = human_profile(zoeva, mutants) - human_profile(zoeva, ref[None])[0]
    effect, null_effect = np.abs(shift).max(), np.abs(null_profiles[elsewhere]).max(1)
    embeddings = zoeva.embed(np.stack([risk, other]))
    top = classes[int(profiles[1].argmax())]
    cell = zoeva.cells.names.index(f"human|{top}")
    saliency = np.abs(attribution(zoeva, other[None], cell)[0])
    core_saliency = saliency[HALF - MOTIF_REACH:HALF + MOTIF_REACH + 1]  # the model weighs the centre more; rank within it
    seeds = []
    for h in catalogue:
        if h["family"] != "motif" or h["species"] != "human" or h["verdict"] == "not supported":
            continue
        hits = [occurrences(w[None], h["seed"])[0] for w in (risk, other)]
        cover = [bool(x[max(0, HALF - 5):HALF + 1].any()) for x in hits]
        if cover[0] != cover[1]:
            seeds.append({"hypothesis": h["id"], "seed": h["seed"], "status": h["status"], "verdict": h["verdict"],
                          "change_on_risk": "creates" if cover[0] else "breaks", "literature": h["literature"]["name"]})

    # motifs
    names, sources, *_ = bank
    position = int(row["pos"]) - 1
    context = np.asarray(genome[row["chrom"]][position - HALF - MOTIF_REACH:position + HALF + MOTIF_REACH])
    relative, delta, null, null_present = motif_readout(ref, alt, context, bank, risk_is_alt)
    percentile = (np.abs(null) < np.abs(delta)[None, :]).mean(0) * 100
    present = relative.max(0) >= MOTIF_PRESENT
    order = np.lexsort((-relative.max(0), -np.abs(delta) * present))
    top_motifs = [{"motif": names[i], "source": sources[i], "risk_minus_other": float(delta[i]),
                   "relative_ref": float(relative[0, i]), "relative_alt": float(relative[1, i]),
                   "percentile_nearby": float(percentile[i])} for i in order[:TOP_MOTIFS] if present[i]]
    literature = expected_check(row, names, relative, delta, percentile, risk_is_alt)
    leads = [m for m in top_motifs if m["source"] == "JASPAR" and m["percentile_nearby"] >= LEAD_PERCENTILE]
    for m in leads:
        m["immune_family"] = any(m["motif"].upper().startswith(f) for f in IMMUNE_FAMILIES)
        on_risk, on_other = (m["relative_alt"], m["relative_ref"]) if risk_is_alt else (m["relative_ref"], m["relative_alt"])
        m["statement"] = (f"The {row['name']} risk allele ({row['risk']}) {'strengthens' if m['risk_minus_other'] > 0 else 'weakens'} "
                          f"a {m['motif'].split(' (')[0]} site (relative score {on_other:.2f} on the other allele, {on_risk:.2f} on the risk allele).")
    return {
        "name": row["name"], "gene": row["gene"], "consequence": row["consequence"], "group": row["group"],
        "chrom": row["chrom"], "pos": int(row["pos"]), "ref": row["ref"], "alt": row["alt"], "risk": row["risk"],
        "held_out_chromosome": row["chrom"] in SPECIES["human"][2], "allele_note": row["allele_note"],
        "literature_mechanism": row["literature_mechanism"], "references": row["references"].split(","),
        "classes": classes, "risk_profile": profiles[0].round(4).tolist(), "other_profile": profiles[1].round(4).tolist(),
        "top_class_other_allele": top, "largest_class_shift": float(effect),
        "largest_shift_class": classes[int(np.abs(shift).argmax())], "shift_toward": float(shift[np.abs(shift).argmax()]),
        "class_shift_percentile": float((null_effect < effect).mean() * 100),
        "embedding_cosine_risk_other": float(embeddings[0] @ embeddings[1]),
        "attribution_percentile": float((core_saliency < saliency[HALF]).mean() * 100),
        "seeds_changed": seeds, "top_motif_changes": top_motifs, "literature_check": literature, "motif_leads": leads,
        "motifs_present": int(present.sum()), "motifs_present_above_95th": int((present & (percentile >= LEAD_PERCENTILE)).sum()),
        "motifs_present_above_95th_expected": expected_leads(null, null_present),
        "windows": {"risk": decode(risk[None])[0], "other": decode(other[None])[0]},
        "canonical_splice_dinucleotide": splice_dinucleotide(ref, alt) if "splice" in row["consequence"] else None,
        "explain": {"risk": explain(zoeva, decode(risk[None])[0], catalogue), "other": explain(zoeva, decode(other[None])[0], catalogue)},
        "_shift_embedding": (embeddings[0] - embeddings[1]).tolist(), "_shift_profile": shift.tolist(),
        "_null_embedding": (zoeva.embed(mutants[elsewhere]) - zoeva.embed(ref[None])[0]).tolist(),
        "_null_profile": null_profiles[elsewhere].tolist(),
    }


def expected_leads(null: np.ndarray, null_present: np.ndarray) -> float:
    """How many present motifs reach the lead percentile when a random nearby substitution plays the variant:
    each substitution is ranked against all the others (leave one out), with the same presence filter."""
    magnitude = np.abs(null)
    ranks = (magnitude[None, :, :] < magnitude[:, None, :]).sum(1) / (len(null) - 1) * 100
    return float(((ranks >= LEAD_PERCENTILE) & null_present).sum(1).mean())


def splice_dinucleotide(ref: np.ndarray, alt: np.ndarray) -> dict | None:
    """Whether the variant sits in a GT or AG dinucleotide (either strand) that the other allele destroys; a plain
    sequence fact used to interpret splice-site variants, not a model readout."""
    out = {}
    for label, pair in (("GT", (2, 3)), ("AG", (0, 2))):
        for strand, seq_ref, seq_alt in (("+", ref, alt), ("-", revcomp_window(ref), revcomp_window(alt))):
            for offset in (-1, 0):
                i = HALF + offset
                if tuple(seq_ref[i:i + 2]) == pair and tuple(seq_alt[i:i + 2]) != pair:
                    out = {"dinucleotide": label, "strand": strand, "destroyed_by": "alt"}
                if tuple(seq_alt[i:i + 2]) == pair and tuple(seq_ref[i:i + 2]) != pair:
                    out = {"dinucleotide": label, "strand": strand, "destroyed_by": "ref"}
    return out or None


def revcomp_window(codes: np.ndarray) -> np.ndarray:
    """Reverse complement that keeps the variant at index HALF (windows are even-length: shift by one)."""
    return np.roll(revcomp(codes), 1)


def expected_check(row, names, relative, delta, percentile, risk_is_alt) -> dict | None:
    """Does the motif readout reproduce the published direction of the expected mechanism?"""
    if not row["expected_motif"]:
        return None
    wanted = [i for i, n in enumerate(names) if n == row["expected_motif"] or n.startswith(row["expected_motif"] + " (")]
    change = row["expected_change"]
    # convert every expectation to "risk minus other" sign
    sign = {"gain_on_risk": 1, "loss_on_risk": -1, "loss_on_alt": 1 if not risk_is_alt else -1}[change]
    best = max(wanted, key=lambda i: sign * delta[i])
    present = relative[:, best].max() >= MOTIF_PRESENT
    direction = np.sign(delta[best]) == sign
    verdict = ("recovers published direction" if direction and present and percentile[best] >= STRONG else
               "direction consistent, weak" if direction else "not recovered")
    return {"motif": names[best], "expected": change, "risk_minus_other": float(delta[best]),
            "relative_ref": float(relative[0, best]), "relative_alt": float(relative[1, best]),
            "percentile_nearby": float(percentile[best]), "verdict": verdict}


# ---------------------------------------------------------------- pairs and loci

def pair_report(a: dict, b: dict, label: str, rng) -> dict:
    """Do the two risk alleles push their windows the same way? Cosine of risk-minus-other shifts, in embedding
    and in class space, ranked against cosines of random substitution pairs at the two loci."""
    out = {"pair": f"{a['name']} x {b['name']}", "genes": f"{a['gene']} x {b['gene']}", "evidence": label}
    for space in ("embedding", "profile"):
        x, y = np.asarray(a[f"_shift_{space}"]), np.asarray(b[f"_shift_{space}"])
        nx, ny = np.asarray(a[f"_null_{space}"]), np.asarray(b[f"_null_{space}"])
        cosine = lambda u, v: (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
        observed = float(cosine(x, y))
        null = cosine(nx[rng.integers(0, len(nx), PAIR_NULL)], ny[rng.integers(0, len(ny), PAIR_NULL)])
        out[space] = {"cosine": observed, "percentile": float((null < observed).mean() * 100),
                      "null_quantiles": np.percentile(null, [5, 25, 50, 75, 95]).round(4).tolist()}
    return out


def run(results: str, cache: str) -> dict:
    root = Path(results)
    zoeva = Zoeva(results)
    genome = genomes.load("human", cache)
    catalogue = read_json(root / "hypotheses" / "tested.json")["hypotheses"]
    variants = pd.read_csv(VARIANTS, sep="\t", keep_default_na=False)
    bank = motif_bank()
    reports = {row["name"]: variant_report(zoeva, genome, row, catalogue, bank) for _, row in variants.iterrows()}
    rng = np.random.default_rng(SEED)
    pairs = [pair_report(reports[a], reports[b], label, rng) for a, b, label in PAIRS]
    features = pd.read_csv(root / "features" / "human.tsv.gz", sep="\t", keep_default_na=False)
    loci = {name: locus_track(zoeva, genome, features, chrom, lo, hi) | {
        "variants": [{"name": r["name"], "pos": r["pos"]} for r in reports.values() if r["chrom"] == chrom and lo <= r["pos"] < hi]}
        for name, (chrom, lo, hi) in LOCI.items()}
    checks = [r["literature_check"] for r in reports.values() if r["literature_check"]]
    summary = {"n_variants": len(reports), "n_motifs_scanned": len(bank[0]),
               "literature_checks": {r["name"]: r["literature_check"]["verdict"] for r in reports.values() if r["literature_check"]},
               "recovered": sum(c["verdict"] == "recovers published direction" for c in checks), "n_checks": len(checks),
               "model_sees_variant": [r["name"] for r in reports.values() if r["class_shift_percentile"] >= 95 or r["seeds_changed"]],
               "in_sample_variants": [r["name"] for r in reports.values() if not r["held_out_chromosome"]],
               "motif_leads": [{"variant": r["name"], "gene": r["gene"], **m} for r in reports.values() for m in r["motif_leads"]]}
    payload = {"summary": summary, "variants": [{k: v for k, v in r.items() if not k.startswith("_")} for r in reports.values()],
               "pairs": pairs, "loci": loci}
    write_json(root / "asthma" / "asthma.json", payload)
    (root / "asthma" / "report.md").write_text(report(payload))
    return summary


def report(payload: dict) -> str:
    s = payload["summary"]
    lines = ["# Exploratory asthma mechanisms (starting from Eliasen et al. 2022)", "",
             f"{s['n_variants']} variants; {s['n_motifs_scanned']} motif matrices (JASPAR 2026 vertebrates + curated signals). "
             f"Published mechanisms reproduced at sequence level: {s['recovered']} of {s['n_checks']}. Model readouts are "
             f"in-sample for {', '.join(s['in_sample_variants'])} (training chromosomes).", "",
             "| Variant | Gene | Class shift (pct) | Attribution pct | Seeds changed | Strongest motif change (risk - other) | Literature |",
             "|---|---|---|---|---|---|---|"]
    for v in payload["variants"]:
        motif = v["top_motif_changes"][0] if v["top_motif_changes"] else None
        motif_text = f"{motif['motif']} {motif['risk_minus_other']:+.2f} ({motif['percentile_nearby']:.0f}th)" if motif else "none present"
        seeds = "; ".join(f"{x['change_on_risk']} {x['seed']} ({x['hypothesis'].split(':')[0]})" for x in v["seeds_changed"]) or "none"
        lit = v["literature_check"]["verdict"] if v["literature_check"] else "no sequence-level prediction"
        lines.append(f"| {v['name']} | {v['gene']} | {v['largest_class_shift']:.3f} {v['largest_shift_class']} ({v['class_shift_percentile']:.0f}) | "
                     f"{v['attribution_percentile']:.0f} | {seeds} | {motif_text} | {lit} |")
    lines += ["", "## Motif leads (JASPAR, >= 95th percentile of nearby substitutions, motif present on one allele)", ""]
    lines += [f"- {m['statement']} {m['percentile_nearby']:.0f}th percentile{' · immune TF family' if m['immune_family'] else ''}"
              for m in s["motif_leads"]] or ["- none"]
    for v in payload["variants"]:
        if v["canonical_splice_dinucleotide"]:
            d = v["canonical_splice_dinucleotide"]
            lines.append(f"- {v['name']}: the {'alternative' if d['destroyed_by'] == 'alt' else 'reference'} allele destroys a "
                         f"{d['dinucleotide']} dinucleotide on the {d['strand']} strand (sequence fact, not a model readout).")
    lines += ["", "## Interacting pairs", "", "| Pair | Evidence | Embedding cosine (pct) | Class-shift cosine (pct) |", "|---|---|---|---|"]
    for p in payload["pairs"]:
        lines.append(f"| {p['pair']} ({p['genes']}) | {p['evidence']} | {p['embedding']['cosine']:+.2f} ({p['embedding']['percentile']:.0f}) | "
                     f"{p['profile']['cosine']:+.2f} ({p['profile']['percentile']:.0f}) |")
    return "\n".join(lines) + "\n"
