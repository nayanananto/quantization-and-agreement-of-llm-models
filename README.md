# Quantization and Agreement in Homogeneous LLM Panels

This branch is the minimal reproducibility artifact for the paired FP16--NF4
study. It contains no manuscript files, generated paper figures, failed runs,
notebook workers, or polling infrastructure.

## Contents

- `configs/study.json`: the eight model--precision--task conditions.
- `experiments/run_panel.py`: dataset loading, prompting, generation, parsing,
  and question-level result writing.
- `experiments/run_dual_gpu.py`: condition-level two-GPU orchestration.
- `results/final/`: the eight raw 300-question JSONL files used in the paper.
- `analysis/analyze_results.py`: aggregation, tie-aware panel metrics, and
  10,000-replicate paired question bootstrap.
- `analysis/paper_metrics.{csv,json}`: the reported statistics and intervals.
- `analysis/gsm8k_parser_sensitivity.py`: relaxed-parser GSM8K sensitivity
  analysis that preserves explicit final answers and applies a last-number
  fallback only to otherwise invalid responses.
- `analysis/gsm8k_parser_sensitivity.{csv,json}`: the sensitivity estimates
  and 10,000-replicate paired bootstrap intervals reported in the appendix.
- `provenance/environments.json`: package and CUDA versions for the final runs.

The study uses Qwen2.5-3B-Instruct and Phi-3.5-mini-instruct, five independently
sampled panel members per question, temperature 0.7, top-p 0.9, and base seed
20260809. The fixed evaluation sets are the first 300 TruthfulQA MC1 examples
and the first 300 GSM8K test examples. Phi remote model code is pinned in the
runner to revision `2fe192450127e6a83f7441aef6e3ca586c338b77`.

## Reproduce the analysis

```bash
python -m pip install -r requirements.txt
python analysis/analyze_results.py
python analysis/gsm8k_parser_sensitivity.py
```

These commands overwrite the corresponding CSV and JSON files in `analysis/`.
The default analyses use 10,000 bootstrap replicates and may take a few
minutes. Quick integrity checks can use:

```bash
python analysis/analyze_results.py --bootstrap-replicates 100
python analysis/gsm8k_parser_sensitivity.py --bootstrap-replicates 100
```

## Rerun inference

On a CUDA machine, run:

```bash
python experiments/run_dual_gpu.py \
  --config configs/study.json \
  --output-dir results/reproduction
```

The runner uses up to two GPUs and writes one JSONL record per completed
question. Existing question IDs are skipped, so an interrupted run can resume
from the same output directory.
