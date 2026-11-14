"""Figures from the result files. Orange = CLOP2 / Zoeva, blue = sequence-statistics baselines, grey = controls."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from .common import BIOTYPE_KEYS, BIOTYPES, SPECIES, SPECIES_KEYS, read_json

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#6250d6", "#e34948")
GREY, INK, MUTED, GRID, SURFACE = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPECIES_COLOUR = dict(zip(SPECIES_KEYS, (GREY, YELLOW, GREEN, AQUA, VIOLET, ORANGE)))
SPECIES_SHORT = {"ecoli": "E. coli", "yeast": "S. cerevisiae", "arabidopsis": "A. thaliana", "worm": "C. elegans",
                 "fly": "D. melanogaster", "human": "H. sapiens"}
BASE_COLOUR = {"A": GREEN, "C": BLUE, "G": YELLOW, "T": RED}
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f6f5f2", "#f6c8ad", "#eb6834", "#7a2a0b"])
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#f6f5f2", ORANGE])
ARM_NAMES = {
    "majority_logistic": "Majority", "majority": "Majority", "composition_logistic": "Dinucleotide odds ratios",
    "kmer4_logistic": "4-mer profile", "kmer6_logistic": "6-mer profile", "kmer5_core_window_logistic": "5-mer core + window",
    "kmer4_knn": "4-mer retrieval", "clop2_main_s0_knn": "CLOP2 retrieval", "clop2_main_s0_text": "CLOP2 text prototypes",
    "clop2_main_s0_zoeva": "Zoeva (retrieval + text)", "clop2_main_s0_knn_shuffled_labels": "CLOP2, shuffled labels",
}
SHOWN_ARMS = ("majority", "clop2_main_s0_knn_shuffled_labels", "composition_logistic", "kmer4_knn", "kmer4_logistic",
              "kmer6_logistic", "kmer5_core_window_logistic", "clop2_main_s0_text", "clop2_main_s0_knn", "clop2_main_s0_zoeva")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "grid.color": GRID, "grid.linewidth": 0.8,
    "svg.fonttype": "none", "legend.frameon": False,
})


def class_name(key: str) -> str:
    return BIOTYPES[key][0]


def arm_colour(arm: str) -> str:
    if "shuffled" in arm or arm.startswith("majority"):
        return GREY
    return ORANGE if arm.startswith("clop2") else BLUE


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def present_classes(cells) -> list[str]:
    return [b for b in BIOTYPE_KEYS if any(c.endswith("|" + b) for c in cells)]


def corpus_atlas(prep: dict, out: Path) -> None:
    """Fig 1: the corpus, species by sequence class, sized by held-out windows."""
    cells = prep["cells"]
    classes = present_classes(cells)
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    for i, species in enumerate(SPECIES_KEYS):
        for j, biotype in enumerate(classes):
            counts = cells.get(f"{species}|{biotype}")
            if counts:
                ax.scatter(j, i, s=counts["test"] * 0.55, color=SPECIES_COLOUR[species], alpha=0.9, lw=0)
                ax.text(j, i - 0.36, f"{counts['train']}", ha="center", va="top", fontsize=5.5, color=MUTED)
    ax.set_xticks(range(len(classes)), [class_name(b) for b in classes], rotation=40, ha="right")
    ax.set_yticks(range(len(SPECIES_KEYS)), [SPECIES_SHORT[s] for s in SPECIES_KEYS], fontstyle="italic")
    ax.set_xlim(-0.6, len(classes) - 0.4)
    ax.set_ylim(len(SPECIES_KEYS) - 0.4, -0.75)
    ax.grid(True, lw=0.5)
    ax.set_axisbelow(True)
    n_cells, n_rows = len(cells), sum(sum(v.values()) for v in cells.values())
    ax.set_title(f"{n_cells} species x class cells, {n_rows:,} windows of 500 bp; dot area = held-out windows, "
                 "number = training windows", fontsize=8.5, fontweight="normal", color=MUTED)
    save(fig, out, "fig1_corpus_atlas")


def benchmark(test: dict, out: Path) -> None:
    """Fig 2: macro F1 for species, class within species, and cell, held-out chromosomes, with 95% intervals."""
    tasks = (("species", "Species (6)"), ("biotype_given_species", "Class, species given"), ("cell", f"Cell ({len(test['cells'])})"))
    arms = [a for a in SHOWN_ARMS if a in test["arms"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, tasks):
        for y, arm in enumerate(arms):
            entry = test["arms"][arm][task]
            value = entry["macro_f1_mean_over_species"] if task == "biotype_given_species" else entry["macro_f1"]
            ax.barh(y, value, color=arm_colour(arm), height=0.68)
            if "macro_f1_ci95" in entry:
                ax.plot(entry["macro_f1_ci95"], [y, y], color=INK, lw=0.9)
            ax.text(max(value, entry.get("macro_f1_ci95", [0, value])[1]) + 0.015, y, f"{value:.2f}", va="center",
                    fontsize=5.5, color=MUTED)
        if task in test["chance"]:
            ax.axvline(test["chance"][task], color=GREY, ls=":", lw=1)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x")
    axes[0].set_yticks(range(len(arms)), [ARM_NAMES.get(a, a) for a in arms], fontsize=6.5)
    axes[0].invert_yaxis()
    fig.supxlabel("Macro F1 on held-out chromosomes (dotted: chance)", fontsize=7, color=MUTED)
    save(fig, out, "fig2_benchmark")


def confusions(test: dict, out: Path) -> None:
    """Fig 3: species confusion and class-within-species confusion of the lead arm, row-normalised."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.9]})
    species = np.array(test["confusion"]["species"], float)
    within = np.array(test["biotype_given_species_confusion"], float)
    keep = np.flatnonzero(within.sum(1) > 0)
    for ax, matrix, labels, title in ((axes[0], species, [SPECIES_SHORT[s] for s in SPECIES_KEYS], "Species"),
                                      (axes[1], within[np.ix_(keep, keep)], [class_name(BIOTYPE_KEYS[i]) for i in keep],
                                       "Sequence class, species given")):
        rows = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        ax.imshow(rows, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(rows):
            if v >= 0.05:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6 if len(labels) > 8 else 7.5,
                        color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Called", fontsize=8)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("True", fontsize=8)
    save(fig, out, "fig3_confusions")


def species_by_class(test: dict, out: Path) -> None:
    """Fig 4: how well each sequence class carries its species' identity."""
    table = test["species_recall_by_biotype"]
    classes = [b for b in BIOTYPE_KEYS if b in table]
    matrix = np.array([[np.nan if table[b].get(s) is None else table[b][s] for s in SPECIES_KEYS] for b in classes])
    order = np.argsort(-np.nanmean(matrix, 1))
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.imshow(np.ma.masked_invalid(matrix[order]), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(matrix[order]):
        if not np.isnan(v):
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(SPECIES_KEYS)), [SPECIES_SHORT[s] for s in SPECIES_KEYS], rotation=35, ha="right", fontstyle="italic")
    ax.set_yticks(range(len(classes)), [class_name(classes[i]) for i in order])
    ax.set_title("Species recall (%) by sequence class")
    ax.spines[:].set_visible(False)
    save(fig, out, "fig4_species_by_class")


def transfer(comp: dict, loso: dict, out: Path) -> None:
    """Fig 5: naming never-seen species x class combinations, and classifying windows of a never-seen species."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained")
    ax = axes[0]
    arms = (("zero_shot_text", "CLOP2 text, cell never seen", ORANGE),
            ("zero_shot_factorised_knn", "CLOP2 retrieval, species x class", AQUA),
            ("zero_shot_factorised_kmer5_core_window_logistic", "5-mer logistic, species x class", BLUE),
            ("trained_on_cell_text", "CLOP2 text, cell seen (reference)", GREY))
    held = comp["held_out_cells"]
    width = 0.8 / len(arms)
    for a, (arm, label, colour) in enumerate(arms):
        values = [comp["arms"][arm][cell]["cell_top1"] for cell in held]
        ax.bar(np.arange(len(held)) + (a - (len(arms) - 1) / 2) * width, values, width, color=colour, label=label)
    ax.axhline(comp["chance_cell_top1"], color=GREY, ls=":", lw=1)
    ax.set_xticks(range(len(held)), [f"{SPECIES_SHORT[c.split('|')[0]]} {class_name(c.split('|')[1])}" for c in held],
                  fontsize=7.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(f"Top-1 among {comp['n_cells']} cells")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_title("Compositional: cells withheld from training")
    ax.legend(fontsize=6.5, loc="upper left", ncols=2)
    ax.grid(axis="y")
    ax = axes[1]
    arms = (("unseen_species_zoeva", "Zoeva, species unseen", ORANGE),
            ("unseen_species_kmer5_core_window_logistic", "5-mer logistic, species unseen", BLUE),
            ("species_in_training_knn", "CLOP2 retrieval, species seen", GREY))
    width = 0.8 / len(arms)
    for a, (arm, label, colour) in enumerate(arms):
        ax.bar(np.arange(len(SPECIES_KEYS)) + (a - 1) * width, [loso[s]["arms"][arm]["macro_f1"] for s in SPECIES_KEYS],
               width, color=colour, label=label)
    for i, s in enumerate(SPECIES_KEYS):
        ax.plot([i - 0.42, i + 0.42], [loso[s]["chance"]] * 2, color=INK, ls=":", lw=1)
    ax.set_xticks(range(len(SPECIES_KEYS)), [f"{SPECIES_SHORT[s]} ({len(loso[s]['classes'])})" for s in SPECIES_KEYS],
                  fontsize=7.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Species withheld (number of classes); dotted: chance", fontsize=8)
    ax.set_ylabel("Class macro F1")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_title("Leave one species out")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(axis="y")
    save(fig, out, "fig5_transfer")


def shuffles(ladder: dict, frame: dict, out: Path) -> None:
    """Fig 6: recognition as sequence order is destroyed, and what the "coding" call needs."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1]}, layout="constrained")
    highlight = {"CDS": BLUE, "intron": GREY, "promoter": ORANGE, "tRNA": VIOLET}
    names = list(ladder["versions"])
    short = ["intact", "50 bp\nblocks", "10 bp\nblocks", "dinucl.\nshuffle", "mononucl.\nshuffle"][:len(names)]
    for ax, key, by, title, chance in (
            (axes[0], "species_accuracy_by_species", SPECIES_KEYS, "Species accuracy", ladder["chance"]["species"]),
            (axes[1], "class_accuracy_by_class", None, "Class accuracy, species given", None)):
        groups = by or [b for b in BIOTYPE_KEYS if b in ladder["versions"][names[0]][key]]
        for g in groups:
            values = [ladder["versions"][n][key][g] for n in names]
            colour = SPECIES_COLOUR[g] if by else highlight.get(g, GRID)
            ax.plot(range(len(names)), values, color=colour, lw=1.6 if (by or colour != GRID) else 1, marker="o", ms=3,
                    label=SPECIES_SHORT[g] if by else (class_name(g) if colour != GRID else None), zorder=3 if colour != GRID else 2)
        overall = "species_accuracy" if by else "class_given_species_accuracy"
        ax.plot(range(len(names)), [ladder["versions"][n][overall] for n in names], color=INK, lw=2.4, label="all windows")
        if chance:
            ax.axhline(chance, color=GREY, ls=":", lw=1)
        ax.set_xticks(range(len(names)), short, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3)
    ax = axes[2]
    versions = list(next(iter(frame.values()))["versions"])
    width = 0.38
    for a, species in enumerate(frame):
        ax.barh(np.arange(len(versions)) + (a - 0.5) * width,
                [frame[species]["versions"][v]["coding_call_rate"] for v in versions], width, color=SPECIES_COLOUR[species],
                label=f"{SPECIES_SHORT[species]} (n={frame[species]['n_windows']})")
    ax.set_yticks(range(len(versions)), [textwrap.fill(v, 18) for v in versions], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Windows still called coding", fontsize=8)
    ax.set_title("Inside open reading frames")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)
    ax.grid(axis="x")
    save(fig, out, "fig6_shuffles")


def draw_logo(ax, freq: np.ndarray, x0: float = 0, information: bool = True) -> None:
    """Sequence logo from a (positions, 4) frequency matrix."""
    font = FontProperties(family="DejaVu Sans", weight="bold")
    heights = freq * (2 + (freq * np.log2(np.maximum(freq, 1e-9))).sum(1, keepdims=True)) if information else freq
    for i, row in enumerate(heights):
        y = 0.0
        for b in np.argsort(row):
            if row[b] < 1e-3:
                continue
            letter = "ACGT"[b]
            path = TextPath((0, 0), letter, size=1, prop=font)
            box = path.get_extents()
            transform = (Affine2D().translate(-box.x0, -box.y0).scale(0.9 / box.width, row[b] / box.height)
                         .translate(x0 + i - 0.45, y))
            ax.add_patch(PathPatch(transform.transform_path(path), color=BASE_COLOUR[letter], lw=0))
            y += row[b]
    ax.set_xlim(x0 - 0.6, x0 + len(freq) - 0.4)


def landmarks(data: dict, out: Path, name: str = "fig7_landmarks", keep=None) -> None:
    """Fig 7: which bases around annotated landmarks the class call depends on (occlusion of 10 bp blocks)."""
    data = {k: v for k, v in data.items() if keep is None or k in keep}
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 1.9 * len(data)), sharex=True)
    for ax, (landmark, hosts) in zip(np.atleast_1d(axes), data.items()):
        for species, entry in hosts.items():
            x, y, e = map(np.asarray, (entry["position"], entry["mean_drop"], entry["sem_drop"]))
            ax.fill_between(x, y - e, y + e, color=SPECIES_COLOUR[species], alpha=0.2, lw=0)
            ax.plot(x, y, color=SPECIES_COLOUR[species], lw=1.5, label=f"{SPECIES_SHORT[species]} (n={entry['n_windows']})")
        ax.axvline(0, color=INK, lw=0.7)
        ax.axhline(0, color=GRID, lw=0.7)
        ax.set_title(landmark, fontsize=9)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_ylabel("Similarity\ndrop", fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("Position relative to landmark (bp, transcript orientation)")
    save(fig, out, name)


def landmark_logos(data: dict, out: Path, reach: int = 25) -> None:
    """Fig 7b: base composition at the landmarks, one logo per landmark and species."""
    panels = [(landmark, species, entry) for landmark, hosts in data.items() for species, entry in hosts.items()]
    fig, axes = plt.subplots(int(np.ceil(len(panels) / 3)), 3, figsize=(12, 1.15 * np.ceil(len(panels) / 3)))
    for ax, (landmark, species, entry) in zip(axes.ravel(), panels):
        freq = np.asarray(entry["base_frequency"])
        middle = len(freq) // 2
        draw_logo(ax, freq[middle - reach:middle + reach + 1], x0=-reach)
        ax.set_ylim(0, 2)
        ax.set_yticks([])
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.tick_params(labelsize=6)
        ax.set_title(f"{landmark} · {SPECIES_SHORT[species]}", fontsize=7, fontweight="normal")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    save(fig, out, "fig7b_landmark_logos")


def filters(data: dict, out: Path, n: int = 12) -> None:
    """Fig 8: the most informative first-layer filters, their closest literature consensus and favourite cells."""
    chosen = data["filters"][:n]
    fig, axes = plt.subplots(n // 3, 3, figsize=(12, 1.35 * (n // 3)))
    for ax, item in zip(axes.ravel(), chosen):
        draw_logo(ax, np.asarray(item["pfm"]))
        ax.set_ylim(0, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        match = item["match"] or f"({item['closest_consensus']})"
        cells = ", ".join(f"{SPECIES_SHORT[c['cell'].split('|')[0]]} {class_name(c['cell'].split('|')[1])}" for c in item["top_cells"][:2])
        ax.set_title(f"#{item['filter']}  {match} {item['match_cosine']:.2f}\n{cells}", fontsize=6.5, fontweight="normal",
                     color=INK if item["match"] else MUTED)
    fig.tight_layout()
    save(fig, out, "fig8_filters")


def geometry(data: dict, out: Path) -> None:
    """Fig 9: map of held-out windows and the clustering of cell centroids."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [1, 1.1]}, layout="constrained")
    ax = axes[0]
    shown = data["map"]
    species = np.array([c.split("|")[0] for c in shown["cell"]])
    for s in SPECIES_KEYS:
        mask = species == s
        ax.scatter(np.asarray(shown["x"])[mask], np.asarray(shown["y"])[mask], s=3, color=SPECIES_COLOUR[s], alpha=0.75,
                   lw=0, label=SPECIES_SHORT[s])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=4, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=3)
    ax.set_title("t-SNE of held-out windows (CLOP2)")
    ax = axes[1]
    order = data["dendrogram_order"]
    matrix = np.asarray(data["centroid_cosine"])[np.ix_(order, order)]
    ax.imshow(matrix, cmap=SEQUENTIAL, vmin=np.percentile(matrix, 2), vmax=1)
    labels = [data["cells"][i] for i in order]
    ax.set_yticks(range(len(labels)), [f"{SPECIES_SHORT[c.split('|')[0]]} · {class_name(c.split('|')[1])}" for c in labels], fontsize=6)
    ax.set_xticks([])
    for tick, cell in zip(ax.get_yticklabels(), labels):
        tick.set_color(SPECIES_COLOUR[cell.split("|")[0]] if cell.split("|")[0] != "ecoli" else MUTED)
    explained = data["variance_explained"]
    ax.set_title(f"Cell centroid cosine\nvariance explained: species {explained['species']:.0%}, class {explained['class']:.0%}",
                 fontsize=9)
    ax.spines[:].set_visible(False)
    save(fig, out, "fig9_geometry")


def signatures(data: dict, out: Path) -> None:
    """Fig 10: genomic dinucleotide signatures, the composition axis that species recognition can use."""
    names = data["columns"][:16]
    pairs = sorted({min(n, n[::-1].translate(str.maketrans("ACGT", "TGCA"))) for n in names})  # strand-symmetric
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    matrix = np.array([[data["by_species"][s][p] for p in pairs] for s in SPECIES_KEYS])
    ax.imshow(np.log2(matrix), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(len(pairs)), pairs, fontfamily="DejaVu Sans Mono")
    ax.set_yticks(range(len(SPECIES_KEYS)), [SPECIES_SHORT[s] for s in SPECIES_KEYS], fontstyle="italic")
    probe = data["embedding_linear_probe_r2"]
    ax.set_title("Dinucleotide odds ratio (observed / expected), held-out windows. Linear probe of CLOP2 embedding: "
                 + ", ".join(f"{k} R²={v:.2f}" for k, v in probe.items()), fontsize=7.5, fontweight="normal")
    ax.spines[:].set_visible(False)
    save(fig, out, "fig10_signatures")


def loci(data: dict, out: Path) -> None:
    """Fig 11: class tracks across two asthma-associated loci on held-out chromosomes (exploratory)."""
    fig, axes = plt.subplots(len(data), 1, figsize=(11, 3.3 * len(data)), layout="constrained")
    colours = {"CDS": BLUE, "intron": GREY, "promoter": ORANGE, "enhancer": VIOLET, "open_chromatin": AQUA,
               "five_prime_UTR": GREEN, "three_prime_UTR": YELLOW, "lncRNA": MAGENTA}
    for ax, (name, entry) in zip(np.atleast_1d(axes), data.items()):
        x = np.asarray(entry["centers"]) / 1e6
        tracks = np.asarray(entry["tracks"])
        for j, biotype in enumerate(entry["classes"]):
            if biotype in colours:
                ax.plot(x, tracks[:, j], color=colours[biotype], lw=0.9, label=class_name(biotype))
        for k, (biotype, tiles) in enumerate(entry["annotation"].items()):
            if biotype in colours and tiles:
                ax.scatter(x[tiles], np.full(len(tiles), -0.05 - 0.04 * k), marker="|", s=14, color=colours[biotype])
        variant = entry["variant"]
        ax.axvline(variant["position"] / 1e6, color=RED, lw=0.8, ls="--")
        ax.text(variant["position"] / 1e6, 0.97, f" {variant['rsid']}", color=RED, fontsize=7, ha="left", va="top",
                transform=ax.get_xaxis_transform())
        scores = ", ".join(f"{class_name(b)} {v['auroc']:.2f}" for b, v in entry["auroc"].items())
        ax.set_title(f"{name} ({entry['chrom']})\nAUROC against annotation: {scores}", fontsize=8.5)
        ax.set_ylim(-0.05 - 0.04 * len(entry["annotation"]) - 0.02, 1)
        ax.set_ylabel("P(class | human)", fontsize=7.5)
        ax.ticklabel_format(axis="x", useOffset=False)
    np.atleast_1d(axes)[-1].set_xlabel("Position (Mb, GRCh38)")
    handles = {l: h for ax in np.atleast_1d(axes) for h, l in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), fontsize=7, ncols=len(handles), loc="outside lower center")
    save(fig, out, "fig11_loci")


STATUS_COLOUR = {"recovers known signal": BLUE, "known signal, new context": VIOLET, "novel lead": ORANGE, "composition": GREY}


def hypothesis_label(h: dict) -> str:
    species, biotype = h["cell"].split("|")
    return f"{SPECIES_SHORT[species]} {class_name(biotype)}"


def hypotheses(data: dict, out: Path, n_logos: int = 8, name: str = "fig12_hypotheses") -> None:
    """Fig 12: automatically generated hypotheses, their held-out tests and their standing against the literature."""
    catalogue, summary = data["hypotheses"], data["summary"]
    motifs = [h for h in catalogue if h["family"] == "motif" and h["tests"]["test"].get("testable")]
    rows = 1 + int(np.ceil(n_logos / 4))
    fig = plt.figure(figsize=(7.1, 2.7 + 1.25 * (rows - 1)), layout="constrained")
    grid = fig.add_gridspec(rows, 4, height_ratios=[2.2] + [1] * (rows - 1))
    ax = fig.add_subplot(grid[0, :2])
    for h in motifs:
        t = h["tests"]["test"]
        replicated = h["verdict"] == "replicated"
        ax.scatter(t["necessity_relative"] * 100, np.log2(max(t["odds_ratio"], 1e-2)), s=16 if replicated else 7,
                   color=STATUS_COLOUR[h["status"]] if replicated else "none", edgecolor=STATUS_COLOUR[h["status"]], lw=0.7,
                   alpha=0.95 if replicated else 0.6, zorder=3 if replicated else 2)
    ax.axvline(0, color=GRID, lw=1)
    ax.axhline(0, color=GRID, lw=1)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Necessity: class-probability drop when the motif is\nrandomised, beyond random blocks (% of intact)", fontsize=6.5)
    ax.set_ylabel("log2 odds ratio, held-out enrichment", fontsize=6.5)
    ax.set_title(f"a  {summary['n_motif']} motif hypotheses (filled: replicated)", fontsize=7.5)
    for label, colour in STATUS_COLOUR.items():
        if label != "composition":
            ax.scatter([], [], color=colour, s=14, label=label)
    ax.set_ylim(top=ax.get_ylim()[1] + 1.6)  # headroom so the legend sits above the points
    ax.legend(fontsize=6, loc="upper left", handletextpad=0.1)
    ax = fig.add_subplot(grid[0, 2:])
    chance = summary["chance_match"]
    groups = [("random 6-mer\n(chance)", chance["any_signal"], chance["expected_signal"]),
              ("generated", summary["generated_any_signal"], summary["generated_expected_signal"]),
              ("replicated", summary["replicated_any_signal"] or 0, summary["replicated_expected_signal"] or 0)]
    x = np.arange(len(groups))
    ax.bar(x - 0.2, [g[1] * 100 for g in groups], 0.38, color=VIOLET, label="matches a curated signal")
    ax.bar(x + 0.2, [g[2] * 100 for g in groups], 0.38, color=BLUE, label="matches the signal expected in that cell")
    for i, g in enumerate(groups):
        ax.text(i - 0.2, g[1] * 100 + 1, f"{g[1] * 100:.0f}", ha="center", fontsize=6, color=MUTED)
        ax.text(i + 0.2, g[2] * 100 + 1, f"{g[2] * 100:.1f}" if g[2] < 0.01 else f"{g[2] * 100:.0f}", ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylim(0, max(g[1] for g in groups) * 100 * 1.45)
    ax.set_ylabel("% of motif matrices", fontsize=6.5)
    ax.set_title(f"b  Literature check ({summary['expected_replicated']} of {summary['expected_pairs']} expected signals found)",
                 fontsize=7.5)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(axis="y")
    replicated = sorted((h for h in motifs if h["verdict"] == "replicated"),
                        key=lambda h: -h["tests"]["test"]["necessity_relative"])
    known = [h for h in replicated if h["status"] == "recovers known signal"][:n_logos // 2]
    other = [h for h in replicated if h["status"] != "recovers known signal"][:n_logos - len(known)]
    for i, h in enumerate((known + other)[:n_logos]):
        ax = fig.add_subplot(grid[1 + i // 4, i % 4])
        draw_logo(ax, np.asarray(h["pfm"]))
        ax.set_ylim(0, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        match = h["literature"]["name"] or "no curated match"
        ax.set_title(f"{'c  ' if i == 0 else ''}{hypothesis_label(h)}\n{h['seed']} · {match} · {100 * h['tests']['test']['necessity_relative']:.0f}%",
                     fontsize=6, fontweight="normal", color=STATUS_COLOUR[h["status"]])
    save(fig, out, name)


def stagger(positions: np.ndarray, gap: float) -> np.ndarray:
    """Label row per position (0, 1, ...) so that labels closer than `gap` go on different rows."""
    rows, last = np.zeros(len(positions), int), {}
    for k in np.argsort(positions):
        rows[k] = next(r for r in range(len(positions)) if positions[k] - last.get(r, -np.inf) >= gap)
        last[rows[k]] = positions[k]
    return rows


def asthma(data: dict, out: Path) -> None:
    """Fig 13: exploratory asthma mechanisms starting from the CDHR3 x GSDMB interaction (Eliasen et al. 2022)."""
    variants, pairs = data["variants"], data["pairs"]
    fig = plt.figure(figsize=(7.1, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], width_ratios=[0.95, 1.1, 1.05])
    ax = fig.add_subplot(grid[0, :])
    locus = data["loci"]["17q12-21 (IKZF3-ZPBP2-GSDMB-ORMDL3)"]
    x = np.asarray(locus["centers"]) / 1e6
    tracks = np.apply_along_axis(lambda t: np.convolve(t, np.ones(5) / 5, "same"), 0, np.asarray(locus["tracks"]))  # 500 bp smoothing
    colours = {"CDS": BLUE, "promoter": ORANGE, "five_prime_UTR": GREEN, "three_prime_UTR": YELLOW, "open_chromatin": AQUA,
               "enhancer": VIOLET}
    for j, biotype in enumerate(locus["classes"]):
        if biotype in colours:
            ax.plot(x, tracks[:, j], color=colours[biotype], lw=0.6, label=class_name(biotype))
    for biotype, tiles in locus["annotation"].items():
        if biotype in colours and tiles:
            ax.scatter(x[tiles], np.full(len(tiles), -0.06 - 0.04 * list(colours).index(biotype)), marker="|", s=6, lw=0.6,
                       color=colours[biotype])
    positions = np.array([v["pos"] for v in locus["variants"]]) / 1e6
    for v, pos, row in zip(locus["variants"], positions, stagger(positions, 0.1 * np.ptp(x))):
        ax.axvline(pos, color=RED, lw=0.6, ls="--")
        ax.text(pos, 0.98 - 0.09 * row, f" {v['name']}", fontsize=5.5, color=RED, ha="left", va="top", transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.32, 0.75)
    ax.set_xlim(x.min(), x.max())
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.tick_params(labelsize=6)
    ax.set_ylabel("P(class | human)", fontsize=6.5)
    ax.set_xlabel("chr17 position (Mb, GRCh38); ticks below: annotated tiles; training chromosome (in-sample readout)", fontsize=6.5)
    ax.legend(fontsize=5.5, ncols=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1)
    ax.set_title("a  17q12-21 class tracks", fontsize=7.5, pad=14)
    ax = fig.add_subplot(grid[1, 0])
    columns = ["shift", "attr.", "motif", "leads"]
    matrix = np.full((len(variants), len(columns)), np.nan)
    labels = np.full(matrix.shape, "", dtype=object)
    for i, v in enumerate(variants):
        matrix[i, 0], labels[i, 0] = v["class_shift_percentile"], f"{v['class_shift_percentile']:.0f}"
        matrix[i, 1], labels[i, 1] = v["attribution_percentile"], f"{v['attribution_percentile']:.0f}"
        if v["literature_check"]:
            matrix[i, 2], labels[i, 2] = v["literature_check"]["percentile_nearby"], f"{v['literature_check']['percentile_nearby']:.0f}"
        ratio = v["motifs_present_above_95th"] / max(v["motifs_present_above_95th_expected"], 1e-9)
        matrix[i, 3], labels[i, 3] = min(ratio, 3) / 3 * 100, f"{v['motifs_present_above_95th']}/{v['motifs_present_above_95th_expected']:.1f}"
    ax.imshow(np.ma.masked_invalid(matrix), cmap=SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
    for (i, j), text in np.ndenumerate(labels):
        ax.text(j, i, text or "-", ha="center", va="center", fontsize=5.5, color=INK if text else MUTED)
    ax.set_xticks(range(len(columns)), columns, fontsize=5.5)
    ax.set_yticks(range(len(variants)), [f"{v['name']} {v['gene']}" for v in variants], fontsize=5.5)
    ax.tick_params(length=0)
    ax.set_title("b  Percentile among nearby\nsubstitutions", fontsize=7.5)
    ax.spines[:].set_visible(False)
    ax = fig.add_subplot(grid[1, 1])
    checks = [v for v in variants if v["literature_check"]]
    for i, v in enumerate(checks):
        c = v["literature_check"]
        on_risk, on_other = (c["relative_alt"], c["relative_ref"]) if v["risk"] == v["alt"] else (c["relative_ref"], c["relative_alt"])
        ax.annotate("", xy=(i, on_risk), xytext=(i, on_other), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
        ax.scatter([i], [on_other], color=GREY, s=12, zorder=3)
        ax.text(i, 0.42, f"{c['verdict'].replace(' ', '\n')}\n{c['percentile_nearby']:.0f}th pct", ha="center", va="bottom",
                fontsize=5, color=MUTED)
    ax.axhline(0.85, color=GRID, ls=":", lw=1)
    ax.set_xticks(range(len(checks)), [f"{v['name']}\n{v['literature_check']['motif'].split(' (')[0]}" for v in checks], fontsize=5)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(-0.5, len(checks) - 0.5)
    ax.set_ylabel("Relative motif score\n(grey: other allele; arrow: risk allele)", fontsize=6)
    ax.set_title("c  Published mechanisms", fontsize=7.5)
    ax = fig.add_subplot(grid[1, 2])
    for i, pr in enumerate(pairs):
        q = pr["embedding"]["null_quantiles"]
        ax.plot([q[0], q[4]], [i, i], color=GRID, lw=1.5)
        ax.plot([q[1], q[3]], [i, i], color=GREY, lw=4, solid_capstyle="butt")
        ax.scatter([pr["embedding"]["cosine"]], [i], color=ORANGE if pr["evidence"] == "replicated interaction" else BLUE, s=18, zorder=3)
        ax.text(1.0, i, f" {pr['embedding']['percentile']:.0f}th", va="center", fontsize=5.5, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(pairs)), [f"{p['genes'].replace(' x ', ' × ')}\n{p['evidence']}" for p in pairs], fontsize=5.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color=GRID, lw=1)
    ax.set_xlabel("Cosine of risk-allele embedding shifts\n(grey: random substitution pairs)", fontsize=6)
    ax.set_title("d  Interacting pairs", fontsize=7.5)
    save(fig, out, "fig13_asthma")


FAMILIES = {"protein-coding": (("CDS",), BLUE), "UTRs": (("five_prime_UTR", "three_prime_UTR"), AQUA),
            "introns and intergenic": (("intron", "intergenic"), "#b9b7b0"),
            "regulatory": (("promoter", "enhancer", "open_chromatin", "replication_origin"), ORANGE),
            "RNA genes": (("tRNA", "snRNA", "snoRNA", "miRNA", "lncRNA"), VIOLET),
            "repeats and pseudogenes": (("transposable_element", "pseudogene"), MAGENTA)}
FAMILY_OF = {b: (name, colour) for name, (members, colour) in FAMILIES.items() for b in members}
FRIENDLY = {"CDS": "genes (coding)", "five_prime_UTR": "5' UTR", "three_prime_UTR": "3' UTR", "open_chromatin": "open chromatin",
            "transposable_element": "transposons", "replication_origin": "replication origins"}


def friendly(cell: str) -> str:
    species, biotype = cell.split("|")
    return f"{SPECIES_SHORT[species]} {FRIENDLY.get(biotype, class_name(biotype))}"


def consensus_logo_matrix(consensus: str) -> np.ndarray:
    from .literature import consensus_matrix
    return consensus_matrix(consensus)


def oriented(pfm: np.ndarray, consensus: str) -> np.ndarray:
    """The strand of a discovered matrix that reads like the textbook consensus."""
    target, flipped = consensus_logo_matrix(consensus), pfm[::-1, ::-1]
    return flipped if best_offset_score(flipped, target) > best_offset_score(pfm, target) else pfm


def best_offset_score(query: np.ndarray, target: np.ndarray) -> float:
    """Best background-subtracted cosine over offsets, same strand only."""
    best, need = -1.0, min(6, len(query), len(target))
    for offset in range(-(len(target) - need), len(query) - need + 1):
        q0, t0 = max(offset, 0), max(-offset, 0)
        width = min(len(query) - q0, len(target) - t0)
        a, b = query[q0:q0 + width] - 0.25, target[t0:t0 + width] - 0.25
        best = max(best, float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)))
    return best


CONCEPT_LAYOUTS = {  # name: (rows, figure size in inches, font size, step title size)
    "poster": (1, (16, 5.2), 8.5, 12), "paper": (2, (7.1, 5.6), 6.5, 8)}


def concept(data: dict, geometry: dict, out: Path, layout: str = "poster") -> None:
    """Fig 0: how Zoeva works, on one real example: read a window, place it in the embedding, propose and test a
    hypothesis, check it against the literature. One row of four steps (poster) or a 2 x 2 grid (paper column width)."""
    e = data["example"]
    rows, size, fs, title_fs = CONCEPT_LAYOUTS[layout]
    cols = 4 // rows
    fig = plt.figure(figsize=size)
    grid = fig.add_gridspec(2 * rows, cols, height_ratios=[0.13, 1] * rows, wspace=0.32 if rows == 1 else 0.28,
                            hspace=0.08 if rows == 1 else 0.5, width_ratios=[1.25, 1.15, 0.9, 1] if rows == 1 else [1, 1])
    cell = [(2 * (k // cols), k % cols) for k in range(4)]
    steps = ["1  Read DNA", "2  Place it on the map", "3  Propose and test", "4  Check the literature"]
    for (r, c), title in zip(cell, steps):
        ax = fig.add_subplot(grid[r, c])
        ax.axis("off")
        ax.text(0, 0.5, title, fontsize=title_fs, fontweight="bold", va="center", color=INK)
    body = [grid[r + 1, c] for r, c in cell]
    # 1 read: letters sized by in-silico mutagenesis importance
    ax = fig.add_subplot(body[0])
    importance = np.clip(np.asarray(e["importance"]), 0, None)
    importance = importance / importance.max()
    seq, per_row = e["sequence"], 40
    font = FontProperties(family="DejaVu Sans Mono", weight="bold")
    for i, base in enumerate(seq):
        row, col = divmod(i, per_row)
        path = TextPath((0, 0), base, size=1, prop=font)
        box = path.get_extents()
        transform = (Affine2D().translate(-box.x0, -box.y0).scale(0.85 / box.width, 0.9 * (0.2 + 0.8 * importance[i]) / box.height)
                     .translate(col, -row * 1.25))
        ax.add_patch(PathPatch(transform.transform_path(path), color=BASE_COLOUR[base], lw=0, alpha=0.3 + 0.7 * importance[i]))
    first, last = min(e["seed_positions"]), max(e["seed_positions"]) + 6
    for row in range(first // per_row, (last - 1) // per_row + 1):
        start, stop = max(first, row * per_row) - row * per_row, min(last, (row + 1) * per_row) - row * per_row
        ax.add_patch(plt.Rectangle((start - 0.15, -row * 1.25 - 0.12), stop - start + 0.15, 1.15, fill=False, ec=INK, lw=1.2))
    n_rows = int(np.ceil(len(seq) / per_row))
    ax.set_xlim(-0.5, per_row + 0.5)
    ax.set_ylim(-n_rows * 1.25 - 2.6, 1.2)
    ax.axis("off")
    ax.text(0, -n_rows * 1.25 - 0.3, f"{len(seq)} of 500 bases of a held-out Arabidopsis tRNA gene.\nLetter height: tRNA similarity lost "
            "when\nthe base is mutated. Box: GTTCGA (both strands),\nthe motif it proposes.", fontsize=fs, color=MUTED, va="top")
    # 2 place
    ax = fig.add_subplot(body[1])
    species = np.array([c.split("|")[0] for c in geometry["cell"]])
    x, y = np.asarray(geometry["x"]), np.asarray(geometry["y"])
    for sp in SPECIES_KEYS:
        mask = species == sp
        ax.scatter(x[mask], y[mask], s=2.5 if rows == 1 else 1.2, color=SPECIES_COLOUR[sp], alpha=0.35, lw=0)
    for n in e["neighbours"]:
        ax.plot([e["x"], n["x"]], [e["y"], n["y"]], color=INK, lw=0.6, alpha=0.6)
    ax.scatter([e["x"]], [e["y"]], marker="*", s=260 if rows == 1 else 110, color=INK, zorder=5)
    shown = {"arabidopsis|tRNA", "human|promoter", "ecoli|CDS", "worm|three_prime_UTR", "yeast|transposable_element", "human|snRNA"}
    for label in data["labels"]:
        if label["cell"] in shown:
            ax.annotate(friendly(label["cell"]), (label["x"], label["y"]), xytext=(0, 9), textcoords="offset points",
                        fontsize=fs - 1, color=INK, ha="center", bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=GRID, lw=0.6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    top_species, top_class = e["species_scores"][0], e["class_scores"][0]
    ax.set_xlabel(f"Dots: held-out 500 bp windows, coloured by organism.\nStar: the query; lines: its nearest neighbours.\n"
                  f"Call: {SPECIES[top_species['label']][0]} ({top_species['score']:.2f}), {class_name(top_class['label'])} ({top_class['score']:.2f})",
                  fontsize=fs, color=MUTED)
    # 3 propose and test
    ax = fig.add_subplot(body[2])
    scrambled, controls = np.asarray(e["scrambled_probabilities"]), np.asarray(e["control_probabilities"])
    values = [e["intact_probability"], scrambled.mean(), controls.mean()]
    ax.bar(range(3), values, color=[GREY, ORANGE, "#c9c7c0"], width=0.62)
    rng = np.random.default_rng(0)
    for i, sample in ((1, scrambled), (2, controls[:60])):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(sample)), sample, s=4, color=INK, alpha=0.35, lw=0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=fs + 0.5)
    ax.set_xticks(range(3), ["original", "GTTCGA\nscrambled", "random\nblocks"], fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("P(tRNA | A. thaliana)", fontsize=fs)
    ax.set_xlabel("Hypothesis: the tRNA call depends on GTTCGA.\nShown on this window; tested on\nheld-out chromosomes.",
                  fontsize=fs, color=MUTED)
    # 4 check
    sub = body[3].subgridspec(2, 1, hspace=0.9)
    ax = fig.add_subplot(sub[0])
    draw_logo(ax, np.asarray(e["pfm"]))
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_title("found by Zoeva", fontsize=fs + 0.5, color=ORANGE, loc="left")
    ax = fig.add_subplot(sub[1])
    draw_logo(ax, consensus_logo_matrix(e["literature"]["consensus"]), information=False)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"textbook: {e['literature']['name']} (Galli et al. 1981)", fontsize=fs + 0.5, color=BLUE, loc="left")
    ax.text(0, -0.25, "Internal promoter of tRNA genes,\nrecovered without being told.", transform=ax.transAxes, fontsize=fs,
            color=MUTED, va="top")
    # arrows between consecutive steps on the same row, level with the step titles
    for k in range(3):
        (r, c), (r2, c2) = cell[k], cell[k + 1]
        if r == r2:
            left, right = grid[r, c].get_position(fig), grid[r2, c2].get_position(fig)
            middle, level = (left.x1 + right.x0) / 2, (left.y0 + left.y1) / 2
            fig.add_artist(matplotlib.patches.FancyArrowPatch((middle - 0.012, level), (middle + 0.012, level), transform=fig.transFigure,
                                                              arrowstyle="-|>", mutation_scale=title_fs * 1.8, color=MUTED))
    save(fig, out, "fig0_concept" + ("" if layout == "poster" else f"_{layout}"))


def embedding_map(data: dict, geometry: dict, out: Path, clearance: tuple = (32.0, 7.0)) -> None:
    """Fig: the embedding map, coloured by organism and by kind of sequence, with annotation labels placed where
    they point (purest labels first; a label is skipped when another is within `clearance` (dx, dy) map units)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), layout="constrained")
    x, y = np.asarray(geometry["x"]), np.asarray(geometry["y"])
    species = np.array([c.split("|")[0] for c in geometry["cell"]])
    biotype = np.array([c.split("|")[1] for c in geometry["cell"]])
    for sp in SPECIES_KEYS:
        m = species == sp
        axes[0].scatter(x[m], y[m], s=1.5, color=SPECIES_COLOUR[sp], alpha=0.8, lw=0, label=SPECIES_SHORT[sp])
    for name, (members, colour) in FAMILIES.items():
        m = np.isin(biotype, members)
        axes[1].scatter(x[m], y[m], s=1.5, color=colour, alpha=0.8, lw=0, label=name)
    placed = np.empty((0, 2))
    for label in sorted((l for l in data["labels"] if l["purity"] >= 0.6), key=lambda l: -l["purity"]):
        point = np.array([label["x"], label["y"]])
        if not (np.abs(placed - point) < clearance).all(1).any():
            axes[1].annotate(friendly(label["cell"]), point, fontsize=5.5, ha="center", va="center", color=INK,
                             bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec=GRID, lw=0.4, alpha=0.9))
            placed = np.vstack([placed, point])
    axes[0].set_title("a  Coloured by organism", fontsize=8)
    axes[1].set_title("b  Coloured by kind of sequence; boxes: label prototypes", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        legend = ax.legend(fontsize=6, markerscale=4, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=3, handletextpad=0.2,
                           columnspacing=0.8)
    for text in axes[0].get_legend().get_texts():
        text.set_fontstyle("italic")
    save(fig, out, "fig0b_embedding_map")


POSTER_LABELS = ("ecoli|CDS", "human|promoter", "human|snRNA", "worm|three_prime_UTR", "yeast|transposable_element",
                 "arabidopsis|tRNA", "fly|CDS", "arabidopsis|transposable_element")


def embedding_map_poster(data: dict, geometry: dict, out: Path) -> None:
    """Poster version of the map: one panel, organisms as colours, a few large labels."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    x, y = np.asarray(geometry["x"]), np.asarray(geometry["y"])
    species = np.array([c.split("|")[0] for c in geometry["cell"]])
    for sp in SPECIES_KEYS:
        m = species == sp
        ax.scatter(x[m], y[m], s=5, color=SPECIES_COLOUR[sp], alpha=0.8, lw=0, label=SPECIES_SHORT[sp])
    for label in data["labels"]:
        if label["cell"] in POSTER_LABELS:
            ax.annotate(friendly(label["cell"]), (label["x"], label["y"]), xytext=(0, 10), textcoords="offset points", fontsize=10.5,
                        ha="center", color=INK, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=GRID, lw=0.6, alpha=0.92))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    legend = ax.legend(fontsize=10.5, markerscale=4, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    for text in legend.get_texts():
        text.set_fontstyle("italic")
    save(fig, out, "fig0b_embedding_map_poster")


def rediscovered(data: dict, out: Path, scale: float = 1.0, suffix: str = "") -> None:
    """Fig: motifs Zoeva proposed and replicated, side by side with the textbook signal they match."""
    rows = [h for h in data["hypotheses"] if h["family"] == "motif" and h["status"] == "recovers known signal"
            and h["verdict"] != "not supported"]
    rows.sort(key=lambda h: (h["verdict"] != "replicated", -h["tests"]["test"].get("necessity_relative", 0)))
    rows = rows[:6]
    fig, axes = plt.subplots(len(rows), 3, figsize=((10) * scale, (1.15 * len(rows) * scale)), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    for r, h in enumerate(rows):
        ax = axes[r, 0]
        ax.axis("off")
        species, biotype = h["cell"].split("|")
        verdict = "replicated on both held-out sets" if h["verdict"] == "replicated" else "held up on one held-out set"
        ax.text(0, 0.62, f"{h['literature']['name']}", fontsize=15, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0, 0.2, f"{SPECIES_SHORT[species]} {FRIENDLY.get(biotype, class_name(biotype))}", fontsize=12.5, style="italic",
                color=MUTED, transform=ax.transAxes)
        ax.text(0, -0.18, verdict, fontsize=11, color=MUTED, transform=ax.transAxes)
        draw_logo(axes[r, 1], oriented(np.asarray(h["pfm"]), h["literature"]["consensus"]))
        axes[r, 1].set_ylim(0, 2)
        draw_logo(axes[r, 2], consensus_logo_matrix(h["literature"]["consensus"]), information=False)
        axes[r, 2].set_ylim(0, 1)
        for ax in axes[r, 1:]:
            ax.axis("off")
    axes[0, 1].set_title("found by Zoeva", fontsize=14, color=ORANGE)
    axes[0, 2].set_title("textbook", fontsize=14, color=BLUE)
    fig.tight_layout()
    save(fig, out, "fig0c_rediscovered" + suffix)


def asthma_simple(data: dict, out: Path, scale: float = 1.0, suffix: str = "") -> None:
    """Fig: the asthma results in plain terms, stacked for a narrow poster column."""
    variants = {v["name"]: v for v in data["variants"]}
    fig, axes = plt.subplots(2, 1, figsize=(6.2 * scale, 7.4 * scale), gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    for i, name in enumerate(("rs12936231", "rs4065275")):
        v, c = variants[name], variants[name]["literature_check"]
        on_risk, on_other = (c["relative_alt"], c["relative_ref"]) if v["risk"] == v["alt"] else (c["relative_ref"], c["relative_alt"])
        ax.bar([i * 2.6, i * 2.6 + 1], [on_other, on_risk], color=[GREY, ORANGE], width=0.85)
        ax.text(i * 2.6 + 0.5, 0.93, f"{name}\n({v['gene']})\nrisk allele\n{'loses' if on_risk < on_other else 'gains'} a site",
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks([0, 1, 2.6, 3.6], ["other", "risk"] * 2, fontsize=11)
    ax.set_ylim(0.5, 1.18)
    ax.set_yticks([0.5, 0.7, 0.9])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("CTCF motif match", fontsize=11)
    ax.set_title("Published: risk alleles switch CTCF binding\nZoeva: same direction, modest size", fontsize=12.5, loc="left")
    ax = axes[1]
    pairs = data["pairs"]
    for i, pr in enumerate(pairs):
        q = pr["embedding"]["null_quantiles"]
        ax.plot([q[0], q[4]], [i, i], color=GRID, lw=4, solid_capstyle="round")
        ax.scatter([pr["embedding"]["cosine"]], [i], color=ORANGE if pr["evidence"] == "replicated interaction" else BLUE, s=90, zorder=3)
    ax.axvline(0, color=GRID)
    ax.set_yticks(range(len(pairs)), [f"{p['genes'].replace(' x ', ' × ')}\n{p['evidence']}" for p in pairs], fontsize=10.5)
    ax.tick_params(axis="x", labelsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("do both risk alleles push DNA the same way?\n(grey bar: random changes)", fontsize=11)
    ax.set_title("Interacting variants: no shared sequence signal", fontsize=12.5, loc="left")
    fig.tight_layout(h_pad=2)
    save(fig, out, "fig0d_asthma_simple" + suffix)


def render(results: str) -> None:
    root, out = Path(results), Path(results) / "figures"
    mech = root / "mechanisms"
    corpus_atlas(read_json(root / "preparation.json"), out)
    test = read_json(root / "test_metrics.json")
    benchmark(test, out)
    confusions(test, out)
    species_by_class(test, out)
    transfer(read_json(root / "compositional.json"), read_json(root / "leave_one_species_out.json"), out)
    shuffles(read_json(mech / "shuffle_ladder.json"), read_json(mech / "coding_frame.json"), out)
    marks = read_json(mech / "landmarks.json")
    landmarks(marks, out)
    landmarks(marks, out, "fig7c_landmarks_poster", ("tRNA 5' end", "exon start (3' splice site)", "transcription start site"))
    landmark_logos(marks, out)
    filters(read_json(mech / "filters.json"), out)
    geometry(read_json(mech / "geometry.json"), out)
    signatures(read_json(mech / "signatures.json"), out)
    loci(read_json(mech / "loci.json"), out)
    tested = read_json(root / "hypotheses" / "tested.json")
    hypotheses(tested, out)
    hypotheses(tested, out, 0, "fig12b_hypotheses_poster")
    asthma(read_json(root / "asthma" / "asthma.json"), out)
    concept_data, geometry_data = read_json(mech / "concept.json"), read_json(mech / "geometry.json")["map"]
    concept(concept_data, geometry_data, out)
    concept(concept_data, geometry_data, out, "paper")
    embedding_map(concept_data, geometry_data, out)
    rediscovered(tested, out)
    asthma_simple(read_json(root / "asthma" / "asthma.json"), out)
    embedding_map_poster(concept_data, geometry_data, out)
