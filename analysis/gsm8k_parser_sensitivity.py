"""Re-score invalid GSM8K generations with a last-number fallback.

This is a sensitivity analysis only. The last number in a truncated chain of
thought is not necessarily the model's intended final answer, so these results
must be interpreted alongside the primary strict FINAL-marker evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path
from typing import Any

from analyze_results import aggregate, discover, json_safe, paired_bootstrap, question_arrays


MODELS = ("Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct")
PRECISIONS = ("fp16", "nf4")
REPORTED_METRICS = (
    "individual_accuracy",
    "panel_accuracy",
    "beta_all_agents_wrong",
    "kappa_same_wrong_given_all_wrong",
    "same_wrong_all_rate",
    "agreement_brier",
    "agreement_aurc",
    "invalid_answer_rate",
)


def normalize_last_number(text: str) -> str | None:
    """Match the conventional GSM8K evaluation heuristic: use the last number."""
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() else format(number, ".10g")


def reparse_last_number(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reparsed = copy.deepcopy(rows)
    for row in reparsed:
        for sample in row["samples"]:
            # Preserve every explicit final answer from the primary parser.
            answer = sample["answer"]
            if answer is None:
                answer = normalize_last_number(sample["text"])
            sample["answer"] = answer
            sample["correct"] = answer == row["gold"]
    return reparsed


def recovery_counts(
    strict_rows: list[dict[str, Any]], reparsed_rows: list[dict[str, Any]]
) -> dict[str, int | float]:
    strict_samples = [sample for row in strict_rows for sample in row["samples"]]
    reparsed_samples = [sample for row in reparsed_rows for sample in row["samples"]]
    invalid_indices = [
        index for index, sample in enumerate(strict_samples) if sample["answer"] is None
    ]
    numeric_recovered = sum(
        reparsed_samples[index]["answer"] is not None for index in invalid_indices
    )
    correct_recovered = sum(
        reparsed_samples[index]["correct"] for index in invalid_indices
    )
    denominator = len(invalid_indices)
    return {
        "strict_invalid_samples": denominator,
        "strict_total_samples": len(strict_samples),
        "numeric_answers_recovered": numeric_recovered,
        "correct_answers_recovered": correct_recovered,
        "numeric_recovery_rate_given_strict_invalid": (
            numeric_recovered / denominator if denominator else 0.0
        ),
        "correct_recovery_rate_given_strict_invalid": (
            correct_recovered / denominator if denominator else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    conditions, sources = discover(root)

    report: dict[str, Any] = {
        "description": (
            "Sensitivity analysis preserving explicit final answers and using the last numeric "
            "string only as a fallback for strict-invalid GSM8K generations. This fallback may "
            "select an intermediate value when a derivation was truncated."
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "conditions": [],
        "paired_comparisons": [],
    }
    reparsed: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for model in MODELS:
        for precision in PRECISIONS:
            key = (model, "gsm8k", precision)
            strict_rows = conditions[key]
            last_rows = reparse_last_number(strict_rows)
            reparsed[(model, precision)] = last_rows
            strict_metrics = aggregate(question_arrays(strict_rows))
            last_metrics = aggregate(question_arrays(last_rows))
            report["conditions"].append(
                {
                    "model": model,
                    "precision": precision,
                    "source_file": sources[key],
                    "recovery": recovery_counts(strict_rows, last_rows),
                    "strict_final_marker": strict_metrics,
                    "last_number": last_metrics,
                    "last_minus_strict": {
                        metric: last_metrics[metric] - strict_metrics[metric]
                        for metric in REPORTED_METRICS
                    },
                }
            )

    for model_index, model in enumerate(MODELS):
        report["paired_comparisons"].append(
            {
                "model": model,
                "parser": "explicit_final_then_last_number_fallback",
                "metrics": paired_bootstrap(
                    reparsed[(model, "fp16")],
                    reparsed[(model, "nf4")],
                    args.bootstrap_replicates,
                    args.seed + model_index,
                ),
            }
        )

    output_json = root / "analysis" / "gsm8k_parser_sensitivity.json"
    output_csv = root / "analysis" / "gsm8k_parser_sensitivity.csv"
    output_json.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    condition_lookup = {
        (item["model"], item["precision"]): item for item in report["conditions"]
    }
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model",
            "metric",
            "fp16_strict",
            "nf4_strict",
            "fp16_last_number",
            "nf4_last_number",
            "last_number_delta_nf4_minus_fp16",
            "ci95_low",
            "ci95_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in report["paired_comparisons"]:
            model = comparison["model"]
            for metric in REPORTED_METRICS:
                interval = comparison["metrics"][metric]
                writer.writerow(
                    {
                        "model": model,
                        "metric": metric,
                        "fp16_strict": condition_lookup[(model, "fp16")][
                            "strict_final_marker"
                        ][metric],
                        "nf4_strict": condition_lookup[(model, "nf4")][
                            "strict_final_marker"
                        ][metric],
                        "fp16_last_number": condition_lookup[(model, "fp16")][
                            "last_number"
                        ][metric],
                        "nf4_last_number": condition_lookup[(model, "nf4")][
                            "last_number"
                        ][metric],
                        "last_number_delta_nf4_minus_fp16": interval[
                            "delta_nf4_minus_fp16"
                        ],
                        "ci95_low": interval["ci95_low"],
                        "ci95_high": interval["ci95_high"],
                    }
                )

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
