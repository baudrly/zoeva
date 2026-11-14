#!/bin/sh
# usage: scripts_train_chain.sh "variant:seed variant:seed ..."
for job in $1; do
  variant=${job%%:*}; seed=${job##*:}
  python3 -c "from atlas.models import fit; fit('results/manifest.tsv','results/models','$variant',$seed)" > logs/${variant}_s${seed}.log 2>&1
done
