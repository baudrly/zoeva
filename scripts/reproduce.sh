#!/bin/sh
# Full rebuild from the inputs listed in sources.json (about 3 hours on two CPU cores).
# Usage: SOURCES=sources.json CACHE=genomes sh scripts/reproduce.sh
set -e
SOURCES=${SOURCES:-sources.json}; CACHE=${CACHE:-genomes}
run() { python -m atlas.cli --cache "$CACHE" "$@"; }
run prepare --sources "$SOURCES"
for seed in 0 1; do run train --variant main --seed-index $seed; done
run evaluate --split validation
run freeze
run evaluate --split test                       # refuses to run a second time
run train --variant compositional
for s in ecoli yeast arabidopsis worm fly human; do run train --variant without_$s; done
run transfer
run mechanisms
run hypotheses
run asthma
run concept
run figures
python scripts/make_numbers.py
(cd paper && for f in supplement paper abstract poster; do latexmk -xelatex -interaction=nonstopmode -quiet $f.tex; done)
run export-web
