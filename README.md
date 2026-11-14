# Zoeva

Species and Sequence Ontology class recognition from fixed 500 bp windows, with held-out chromosomes, transfer tests and mechanism analyses. `clop2/` is the contrastive DNA–text encoder; `atlas/` builds the corpus, trains, benchmarks, probes and draws. Read `HANDOFF.md` first. Exploratory; not a disease result.

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-tested.txt
python -m pytest tests            # 13 synthetic checks, a few seconds
```

LaTeX outputs need XeLaTeX with the Inter and TeX Gyre Pagella fonts (`latexmk -xelatex`). `pyjaspar` (bundles JASPAR 2026) is used only to annotate hypotheses.

## Read a sequence with the trained model

```
python -m atlas.cli hypothesize --fasta queries.fa --out hypotheses     # writes hypotheses.json and .md
```

```python
from atlas.pipeline import Zoeva
zoeva = Zoeva("results")                       # main model, seed 0, full training index
reports = zoeva.hypothesize({"q1": "ACGT..."})  # species, class, class within top species, neighbours
centres, scores = zoeva.scan(codes)            # cell scores along a long sequence, 500 bp tiles every 100 bp
```

## Automatic hypotheses

```
python -m atlas.cli hypotheses                                   # generate on train, register, test on validation + test, literature check
python -m atlas.cli explain --fasta queries.fa --out explain.json # test the supported hypotheses of each query's top cell on that query
```

`results/hypotheses/registered.json` is written once (delete it to regenerate); `tested.json` holds tests, verdicts, literature matches and the chance baseline; `report.md` is a readable table. Re-running `hypotheses` with an unchanged registered file reuses the tests and only redoes the literature check, so editing `data/literature_signals.tsv` is cheap (record that you did).

## Asthma variants

```
python -m atlas.cli --cache genomes asthma     # readouts for data/asthma_variants.tsv; add rows to extend
```

The browser reader (`webapp/index.html`) runs the same encoder client-side on a 2,640-window index.

## Rebuild

`sources.example.json` lists every input path. Where the inputs came from (all public mirrors; see HANDOFF for the official replacements):

| Genome | Sequence | Annotation |
|---|---|---|
| Human | GRCh38 autosomes, Ensembl r91 GenBank dumps (`maximebodereau/Homo_sapiens.GRCh38.91.chromosome`) | GENCODE intervals (`saketkc/gencode_regions`, hg38), ncRNA from the GenBank dumps, Regulatory Build via Genomic Benchmarks |
| Fly | dm6 2bit (`vivekbhr/dm6_bsgenome`) | FlyBase r6.17 GTF (`bmc-CompBio-user/Annotations`), BDGP6 introns (`gencode_regions`), Stark enhancers via Genomic Benchmarks |
| Worm | ce10 BSgenome `.rda` (`Przemol/BSgenome.Celegans.UCSC.ce10`) | WBcel235 / Ensembl 96 intervals (`gencode_regions`), lifted to ce10 in code |
| Arabidopsis | TAIR10 2bit (`lileiting/BSgenome.Athaliana10.TAIR.TAIR10`) | TAIR10 exon, intron and TE tables (`ajtock/TAIR10`) |
| Yeast | S288C GenBank (`BjornFJohansson/pygenome`) | same records |
| E. coli | NC_000913.3 GenBank (`SMRUCC/GCA_000005845.2_ASM584v2_genomic`) | same record |

```
SOURCES=sources.json CACHE=genomes sh scripts/reproduce.sh
```

| Step | Command | Output |
|---|---|---|
| Corpus | `atlas prepare --sources sources.json` | `results/manifest.tsv`, `features/*.tsv.gz`, `preparation.json` |
| Models | `atlas train --variant main --seed-index N` (also `compositional`, `without_<species>`) | `results/models/*.pt`, `*.embeddings.npz` |
| Benchmark | `atlas evaluate --split validation`; `atlas freeze`; `atlas evaluate --split test` | `validation_metrics.json`, `protocol.json`, `test_metrics.json` |
| Transfer | `atlas transfer` | `compositional.json`, `leave_one_species_out.json` |
| Mechanisms | `atlas mechanisms` | `results/mechanisms/*.json` |
| Hypotheses | `atlas hypotheses` | `results/hypotheses/{registered,tested}.json`, `report.md` |
| Asthma | `atlas asthma` | `results/asthma/asthma.json`, `report.md` |
| Explanatory figure data | `atlas concept` | `results/mechanisms/concept.json` |
| Figures, numbers, paper | `atlas figures`; `python scripts/make_numbers.py`; `latexmk -xelatex supplement.tex paper.tex abstract.tex poster.tex` in `paper/` (supplement first) | `results/figures/`, `paper/numbers.tex`, PDFs |
| Browser app | `atlas export-web` | `webapp/index.html` |

`atlas` is `python -m atlas.cli` (add `--cache DIR` for the genome cache). The test evaluation refuses to run twice.

## Arms in the result files

| Arm | What it does |
|---|---|
| `majority` | Constant ranking by training frequency |
| `composition_logistic` | Strand-symmetric dinucleotide odds ratios and GC |
| `kmer4_logistic`, `kmer6_logistic` | Canonical k-mer profile |
| `kmer5_core_window_logistic` | 5-mer profile of the central 200 bp plus of the window (anchor) |
| `kmer4_knn` | Cosine retrieval on 4-mer profiles |
| `clop2_main_sN_knn`, `_text`, `_zoeva` | CLOP2 retrieval, text prototypes, their mean |
| `clop2_main_s0_knn_shuffled_labels` | Retrieval with permuted training labels |
