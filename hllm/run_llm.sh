#!/bin/sh
# Same backbone, same LoRA, same supervision -- vary the head, loss and decoder.
PY=../.venv/bin/python
set -x
$PY llm_rerank.py --mode hllm --head attn --loss ce       --out results/llm_hllm_ce.json
$PY llm_rerank.py --mode hllm --head attn --loss sinkhorn --out results/llm_hllm_sinkhorn.json
$PY llm_rerank.py --mode ar                               --out results/llm_ar.json
