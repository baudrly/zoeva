# Minimal orchestrator using the CLI
configfile: "config.yaml"

rule all:
    input:
        "run/report/report.md"

rule embed:
    output:
        "run/embeddings/final_embeddings.csv"
    shell:
        "zoeva embed --clop_repo {config[clop_repo]} --input {config[input_file]} --input-type {config[input_type]} --reference_fasta {config.get('reference_fasta','')} --out run/embeddings"

rule hypothesize:
    input:
        "run/embeddings/final_embeddings.csv"
    output:
        "run/hypotheses/hypotheses.parquet"
    shell:
        "zoeva hypothesize --embeddings {input} --out run/hypotheses --zero_shot_prompts {config.get('prompts','')} --text_encoder_onnx {config.get('text_encoder_onnx','')} --text_vocab_json {config.get('text_vocab_json','')}"

rule graph:
    input:
        "run/hypotheses/hypotheses.parquet"
    output:
        "run/graph/graph_scores.parquet"
    shell:
        "zoeva graph --hypotheses {input} --out run/graph --kw_to_label {config.get('kw_to_label','ref/kw_to_label.tsv')} --label_edges {config.get('label_edges','ref/label_edges.tsv')}"

rule validate:
    input:
        fasta = lambda wildcards: config['input_file']
    output:
        "run/validation/summary.parquet"
    shell:
        "zoeva validate --fasta {input.fasta} --out run/validation --blast_db {config.get('blast_db','')} --diamond_db {config.get('diamond_db','')} --pfam_hmm {config.get('pfam_hmm','')} --kofam_db {config.get('kofam_db','')} --eggnog_db {config.get('eggnog_db','')}"

rule report:
    input:
        hypo = "run/hypotheses/hypotheses.parquet",
        val  = "run/validation/summary.parquet",
        g    = "run/graph/graph_scores.parquet"
    output:
        "run/report/report.md"
    shell:
        "zoeva report --hypotheses {input.hypo} --validation {input.val} --graph {input.g} --out run/report"
