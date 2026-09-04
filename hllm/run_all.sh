#!/bin/sh
# Reproduce everything. E4 dominates the runtime (~15 min for both settings).
set -e
PY=../.venv/bin/python
$PY e1_solver_latency.py   | tee results/e1_solver_latency.txt
$PY e1b_solver_impls.py    | tee results/e1b_solver_impls.txt
$PY e2_hungarian_vs_argsort.py | tee results/e2_hungarian_vs_argsort.txt
$PY e3_decoder_quality.py  | tee results/e3_decoder_quality.txt
$PY e5_rectangular.py      | tee results/e5_rectangular.txt
REL_NOISE=0.1 $PY e4_end_to_end.py | tee results/e4_easy.txt
REL_NOISE=1.5 $PY e4_end_to_end.py | tee results/e4_hard.txt

# LLM experiment: needs a CUDA GPU. ~70 min on an RTX 4070.
#   $PY prep_data.py           # build slates from data/*.parquet
#   ./run_llm.sh
