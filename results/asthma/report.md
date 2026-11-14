# Exploratory asthma mechanisms (starting from Eliasen et al. 2022)

8 variants; 1045 motif matrices (JASPAR 2026 vertebrates + curated signals). Published mechanisms reproduced at sequence level: 0 of 3. Model readouts are in-sample for rs6967330, rs2305480, rs12936231, rs4065275, rs11078928, rs7758229, rs2230641 (training chromosomes).

| Variant | Gene | Class shift (pct) | Attribution pct | Seeds changed | Strongest motif change (risk - other) | Literature |
|---|---|---|---|---|---|---|
| rs6967330 | CDHR3 | 0.030 intergenic (56) | 33 | none | TEAD3 (MA0808.1) -6.98 (95th) | no sequence-level prediction |
| rs2305480 | GSDMB | 0.056 CDS (82) | 36 | none | ZNF354C (MA0130.1) +7.18 (96th) | no sequence-level prediction |
| rs12936231 | ZPBP2 | 0.023 intergenic (45) | 19 | none | DMRTA2 (MA1478.2) +6.66 (93th) | direction consistent, weak |
| rs4065275 | ORMDL3 | 0.041 three_prime_UTR (84) | 81 | none | ZNF335 (MA2511.1) +7.18 (97th) | direction consistent, weak |
| rs11078928 | GSDMB | 0.030 snoRNA (44) | 85 | none | TATA box +5.15 (55th) | not recovered |
| rs7758229 | SLC22A3 | 0.061 intron (92) | 34 | none | SP140L (MA2540.1) -7.18 (97th) | no sequence-level prediction |
| rs2230641 | CCNH | 0.059 snoRNA (89) | 94 | none | Foxn1 (MA1684.1) -7.18 (93th) | no sequence-level prediction |
| rs2280134 | ART1 | 0.016 CDS (23) | 90 | none | ZNF395 (MA2566.1) -6.48 (96th) | no sequence-level prediction |

## Motif leads (JASPAR, >= 95th percentile of nearby substitutions, motif present on one allele)

- The rs6967330 risk allele (A) weakens a TEAD3 site (relative score 0.87 on the other allele, 0.73 on the risk allele). 95th percentile
- The rs6967330 risk allele (A) strengthens a HOXC8 site (relative score 0.69 on the other allele, 0.87 on the risk allele). 97th percentile
- The rs6967330 risk allele (A) strengthens a VAX2 site (relative score 0.68 on the other allele, 0.87 on the risk allele). 96th percentile
- The rs6967330 risk allele (A) weakens a TEAD4 site (relative score 0.88 on the other allele, 0.74 on the risk allele). 97th percentile
- The rs2305480 risk allele (G) strengthens a ZNF354C site (relative score 0.67 on the other allele, 0.85 on the risk allele). 96th percentile
- The rs2305480 risk allele (G) strengthens a NR2C1 site (relative score 0.67 on the other allele, 0.85 on the risk allele). 96th percentile
- The rs2305480 risk allele (G) strengthens a NR2C2 site (relative score 0.68 on the other allele, 0.86 on the risk allele). 100th percentile
- The rs2305480 risk allele (G) strengthens a Hand1::Tcf3 site (relative score 0.74 on the other allele, 0.88 on the risk allele). 95th percentile
- The rs12936231 risk allele (C) weakens a NKX2-5 site (relative score 0.88 on the other allele, 0.74 on the risk allele). 97th percentile
- The rs4065275 risk allele (G) strengthens a ZNF335 site (relative score 0.73 on the other allele, 0.87 on the risk allele). 97th percentile
- The rs4065275 risk allele (G) strengthens a NR2C1 site (relative score 0.77 on the other allele, 0.95 on the risk allele). 95th percentile
- The rs4065275 risk allele (G) strengthens a Zic2 site (relative score 0.73 on the other allele, 0.86 on the risk allele). 95th percentile
- The rs7758229 risk allele (T) weakens a SP140L site (relative score 0.87 on the other allele, 0.66 on the risk allele). 97th percentile
- The rs2230641 risk allele (A) weakens a MSC site (relative score 0.86 on the other allele, 0.75 on the risk allele). 97th percentile
- The rs2230641 risk allele (A) weakens a MYF6 site (relative score 0.86 on the other allele, 0.75 on the risk allele). 95th percentile
- The rs2230641 risk allele (A) weakens a Zic1::Zic2 site (relative score 0.88 on the other allele, 0.72 on the risk allele). 99th percentile
- The rs2280134 risk allele (T) weakens a ZNF395 site (relative score 0.95 on the other allele, 0.77 on the risk allele). 96th percentile
- rs11078928: the alternative allele destroys a AG dinucleotide on the - strand (sequence fact, not a model readout).

## Interacting pairs

| Pair | Evidence | Embedding cosine (pct) | Class-shift cosine (pct) |
|---|---|---|---|
| rs6967330 x rs2305480 (CDHR3 x GSDMB) | replicated interaction | -0.01 (49) | -0.27 (26) |
| rs6967330 x rs12936231 (CDHR3 x ZPBP2) | similar interaction reported | -0.30 (18) | -0.47 (3) |
| rs7758229 x rs2305480 (SLC22A3 x GSDMB) | not replicated | -0.57 (2) | -0.76 (0) |
| rs2230641 x rs2280134 (CCNH x ART1) | not replicated | -0.46 (8) | +0.01 (52) |
