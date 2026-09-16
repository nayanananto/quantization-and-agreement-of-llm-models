"""Consolidate condition files and compute paired question-bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAMES = {
    "Qwen_Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "microsoft_Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
}
METRICS = (
    "individual_accuracy",
    "panel_accuracy",
    "beta_all_agents_wrong",
    "kappa_same_wrong_given_valid_all_wrong",
    "kappa_same_wrong_given_all_wrong",
    "same_wrong_all_rate",
    "agreement_brier",
    "agreement_aurc",
    "invalid_answer_rate",
)


def json_safe(value: Any) -> Any:
    """Replace non-finite floats, which are invalid JSON, with explicit nulls."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def discover(
    root: Path,
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    dict[tuple[str, str, str], str],
]:
    candidates: dict[
        tuple[str, str, str],
        list[tuple[Path, list[dict[str, Any]], bool, str]],
    ] = defaultdict(list)
    for path in root.glob("results/*/*.jsonl"):
        parts = path.stem.split("__")
        if len(parts) != 3 or parts[0] not in MODEL_NAMES:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if parts[2] == "truthful_qa_mc1":
            rows = reparse_explicit_mc_answers(rows)
        key = (MODEL_NAMES[parts[0]], parts[2], parts[1])
        metadata = json.loads((path.parent / "worker_metadata.json").read_text(encoding="utf-8"))
        succeeded = metadata.get("status") == "succeeded"
        finished_at = str(metadata.get("finished_at", ""))
        candidates[key].append((path, rows, succeeded, finished_at))

    selected: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    sources: dict[tuple[str, str, str], str] = {}
    for key, versions in candidates.items():
        path, rows, _, _ = max(
            versions,
            key=lambda item: (len(item[1]), item[2], item[3], str(item[0])),
        )
        if len(rows) != 300:
            raise ValueError(f"Incomplete selected condition {path}: {len(rows)}/300")
        selected[key] = rows
        sources[key] = path.relative_to(root).as_posix()
    return selected, sources


def reparse_explicit_mc_answers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover only unambiguous explicit labels missed by the original parser."""
    patterns = (
        r"(?i)\bfinal\s*(?:answer)?\s*:\s*(?:option\s*)?[\[(]?([A-Z])\b",
        r"(?i)\b(?:the\s+)?correct\s+answer\s*(?:is)?\s*:?\s*(?:option\s*)?[\[(]?([A-Z])\b",
    )
    for row in rows:
        for sample in row["samples"]:
            matches = [match for pattern in patterns for match in re.finditer(pattern, sample["text"])]
            if matches:
                answer = max(matches, key=lambda match: match.start()).group(1).upper()
                sample["answer"] = answer
                sample["correct"] = answer == row["gold"]
    return rows


def question_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        samples = row["samples"]
        correct = np.array([sample["correct"] for sample in samples], dtype=float)
        answers = [
            sample["answer"] if sample["answer"] is not None else f"__invalid_{sample['member']}"
            for sample in samples
        ]
        counts = Counter(answers)
        votes = max(counts.values())
        tied_answers = [answer for answer, count in counts.items() if count == votes]
        # Report the expected accuracy of uniform random tie-breaking instead
        # of letting Counter order silently favor the first panel member.
        panel_correct = float(row["gold"] in tied_answers) / len(tied_answers)
        all_wrong = float(not correct.any())
        raw_answers = [sample["answer"] for sample in samples]
        all_valid = None not in raw_answers
        valid_all_wrong = float(bool(all_wrong) and all_valid)
        same_wrong = float(bool(valid_all_wrong) and len(set(raw_answers)) == 1)
        confidence = votes / len(samples)
        values["individual_accuracy"].append(float(correct.mean()))
        values["panel_accuracy"].append(panel_correct)
        values["beta_all_agents_wrong"].append(all_wrong)
        values["valid_all_wrong"].append(valid_all_wrong)
        values["same_wrong_all_rate"].append(same_wrong)
        expected_brier = panel_correct * (confidence - 1.0) ** 2 + (1.0 - panel_correct) * confidence**2
        values["agreement_brier"].append(expected_brier)
        values["agreement_confidence"].append(confidence)
        values["invalid_answer_rate"].append(sum(answer is None for answer in raw_answers) / len(samples))
    arrays = {name: np.asarray(value) for name, value in values.items()}
    arrays["kappa_same_wrong_given_all_wrong"] = arrays["same_wrong_all_rate"]
    arrays["kappa_same_wrong_given_valid_all_wrong"] = arrays["same_wrong_all_rate"]
    return arrays


def tie_averaged_aurc(confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Compute AURC while averaging over every ordering within confidence ties."""
    risks = 1.0 - correctness
    total_risk = 0.0
    seen = 0
    area_terms: list[float] = []
    for threshold in sorted(set(confidence.tolist()), reverse=True):
        block = risks[confidence == threshold]
        block_risk = float(block.sum())
        block_size = len(block)
        for offset in range(1, block_size + 1):
            expected_prefix_risk = total_risk + offset * block_risk / block_size
            area_terms.append(expected_prefix_risk / (seen + offset))
        total_risk += block_risk
        seen += block_size
    return float(np.mean(area_terms))


def aggregate(arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, float]:
    chosen = arrays if indices is None else {key: value[indices] for key, value in arrays.items()}
    beta = chosen["beta_all_agents_wrong"].sum()
    valid_beta = chosen["valid_all_wrong"].sum()
    result = {
        key: float(chosen[key].mean())
        for key in METRICS
        if key
        not in (
            "kappa_same_wrong_given_all_wrong",
            "kappa_same_wrong_given_valid_all_wrong",
            "agreement_aurc",
        )
    }
    result["agreement_aurc"] = tie_averaged_aurc(
        chosen["agreement_confidence"], chosen["panel_accuracy"]
    )
    result["kappa_same_wrong_given_all_wrong"] = (
        float(chosen["same_wrong_all_rate"].sum() / beta) if beta else float("nan")
    )
    result["kappa_same_wrong_given_valid_all_wrong"] = (
        float(chosen["same_wrong_all_rate"].sum() / valid_beta) if valid_beta else float("nan")
    )
    result["questions"] = int(len(chosen["panel_accuracy"]))
    result["cofailure_count"] = int(beta)
    result["valid_cofailure_count"] = int(valid_beta)
    result["same_wrong_count"] = int(chosen["same_wrong_all_rate"].sum())
    return result


def paired_bootstrap(
    fp_rows: list[dict[str, Any]],
    nf_rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
    require_complete_parse: bool = False,
) -> dict[str, dict[str, float]]:
    fp_by_id = {row["id"]: row for row in fp_rows}
    nf_by_id = {row["id"]: row for row in nf_rows}
    ids = sorted(fp_by_id.keys() & nf_by_id.keys())
    if require_complete_parse:
        ids = [
            item
            for item in ids
            if all(sample["answer"] is not None for sample in fp_by_id[item]["samples"])
            and all(sample["answer"] is not None for sample in nf_by_id[item]["samples"])
        ]
    if not require_complete_parse and len(ids) != 300:
        raise ValueError(f"Expected 300 paired questions, found {len(ids)}")
    fp = question_arrays([fp_by_id[item] for item in ids])
    nf = question_arrays([nf_by_id[item] for item in ids])
    fp_point = aggregate(fp)
    nf_point = aggregate(nf)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    output: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        if metric in ("kappa_same_wrong_given_all_wrong", "kappa_same_wrong_given_valid_all_wrong"):
            denominator = "beta_all_agents_wrong" if metric == "kappa_same_wrong_given_all_wrong" else "valid_all_wrong"
            with np.errstate(divide="ignore", invalid="ignore"):
                fp_values = fp["same_wrong_all_rate"][draws].sum(axis=1) / fp[denominator][draws].sum(axis=1)
                nf_values = nf["same_wrong_all_rate"][draws].sum(axis=1) / nf[denominator][draws].sum(axis=1)
        elif metric == "agreement_aurc":
            fp_values = np.asarray(
                [
                    tie_averaged_aurc(fp["agreement_confidence"][draw], fp["panel_accuracy"][draw])
                    for draw in draws
                ]
            )
            nf_values = np.asarray(
                [
                    tie_averaged_aurc(nf["agreement_confidence"][draw], nf["panel_accuracy"][draw])
                    for draw in draws
                ]
            )
        else:
            fp_values = fp[metric][draws].mean(axis=1)
            nf_values = nf[metric][draws].mean(axis=1)
        delta = nf_values - fp_values
        finite_delta = delta[np.isfinite(delta)]
        point_delta = nf_point[metric] - fp_point[metric]
        estimable = math.isfinite(point_delta) and len(finite_delta) > 0
        output[metric] = {
            "delta_nf4_minus_fp16": float(point_delta) if estimable else None,
            "ci95_low": float(np.quantile(finite_delta, 0.025)) if estimable else None,
            "ci95_high": float(np.quantile(finite_delta, 0.975)) if estimable else None,
            "valid_bootstrap_replicates": int(len(finite_delta)),
        }
    return output


def stratified_independence_null(rows: list[dict[str, Any]], indices: np.ndarray | None = None) -> float:
    selected = rows if indices is None else [rows[int(index)] for index in indices]
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for row in selected:
        if all(not sample["correct"] for sample in row["samples"]):
            answers = [sample["answer"] for sample in row["samples"]]
            if None not in answers:
                groups[str(row["gold"])].append(answers)
    total = sum(len(group) for group in groups.values())
    if not total:
        return float("nan")
    expected = 0.0
    for group in groups.values():
        answer_matrix = np.asarray(group, dtype=object)
        categories = set(answer_matrix.ravel())
        group_probability = sum(
            float(np.prod([(answer_matrix[:, member] == answer).mean() for member in range(answer_matrix.shape[1])]))
            for answer in categories
        )
        expected += len(group) / total * group_probability
    return expected


def chance_adjusted_kappa_bootstrap(
    fp_rows: list[dict[str, Any]],
    nf_rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the paired change in kappa minus an answer-frequency null."""
    fp_by_id = {row["id"]: row for row in fp_rows}
    nf_by_id = {row["id"]: row for row in nf_rows}
    ids = sorted(fp_by_id.keys() & nf_by_id.keys())
    fp = [fp_by_id[item] for item in ids]
    nf = [nf_by_id[item] for item in ids]

    def prepare(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cofailure = np.asarray([all(not sample["correct"] for sample in row["samples"]) for row in rows])
        answers = np.asarray([[sample["answer"] for sample in row["samples"]] for row in rows], dtype=object)
        valid = np.asarray([None not in answer_row for answer_row in answers])
        same_wrong = np.asarray(
            [bool(cofailure[index] and valid[index] and len(set(answer_row)) == 1) for index, answer_row in enumerate(answers)]
        )
        gold = np.asarray([str(row["gold"]) for row in rows], dtype=object)
        return cofailure & valid, same_wrong, gold, answers

    def excess(
        prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        indices: np.ndarray,
    ) -> float:
        valid_cofailure, same_wrong, gold, answers = (value[indices] for value in prepared)
        denominator = int(valid_cofailure.sum())
        if not denominator:
            return float("nan")
        point = float(same_wrong.sum() / denominator)
        null = 0.0
        for gold_value in set(gold[valid_cofailure].tolist()):
            block = answers[valid_cofailure & (gold == gold_value)]
            probability = sum(
                float(np.prod([(block[:, member] == answer).mean() for member in range(block.shape[1])]))
                for answer in set(block.ravel().tolist())
            )
            null += len(block) / denominator * probability
        return point - null

    fp_prepared = prepare(fp)
    nf_prepared = prepare(nf)
    full_indices = np.arange(len(ids))
    fp_point = excess(fp_prepared, full_indices)
    nf_point = excess(nf_prepared, full_indices)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    delta = np.asarray([excess(nf_prepared, draw) - excess(fp_prepared, draw) for draw in draws])
    finite = delta[np.isfinite(delta)]
    return {
        "fp16_excess": fp_point,
        "nf4_excess": nf_point,
        "delta_excess_nf4_minus_fp16": nf_point - fp_point,
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
        "valid_bootstrap_replicates": int(len(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    conditions, sources = discover(root)
    expected = {
        (model, dataset, precision)
        for model in MODEL_NAMES.values()
        for dataset in ("truthful_qa_mc1", "gsm8k")
        for precision in ("fp16", "nf4")
    }
    missing = expected - conditions.keys()
    if missing:
        raise ValueError(f"Missing complete conditions: {sorted(missing)}")

    report: dict[str, Any] = {"conditions": [], "paired_comparisons": []}
    for (model, dataset, precision), rows in sorted(conditions.items()):
        report["conditions"].append(
            {
                "model": model,
                "dataset": dataset,
                "precision": precision,
                "source_file": sources[(model, dataset, precision)],
                **aggregate(question_arrays(rows)),
            }
        )

    for pair_index, model in enumerate(MODEL_NAMES.values()):
        for dataset_index, dataset in enumerate(("truthful_qa_mc1", "gsm8k")):
            fp_rows = conditions[(model, dataset, "fp16")]
            nf_rows = conditions[(model, dataset, "nf4")]
            comparison: dict[str, Any] = {
                "model": model,
                "dataset": dataset,
                "bootstrap_replicates": args.bootstrap_replicates,
                "metrics": paired_bootstrap(
                    fp_rows,
                    nf_rows,
                    args.bootstrap_replicates,
                    args.seed + pair_index * 10 + dataset_index,
                ),
            }
            complete_ids = [
                item
                for item in {row["id"] for row in fp_rows} & {row["id"] for row in nf_rows}
                if all(sample["answer"] is not None for sample in next(row for row in fp_rows if row["id"] == item)["samples"])
                and all(sample["answer"] is not None for sample in next(row for row in nf_rows if row["id"] == item)["samples"])
            ]
            comparison["paired_complete_parse"] = {
                "questions": len(complete_ids),
                "metrics": paired_bootstrap(
                    fp_rows,
                    nf_rows,
                    args.bootstrap_replicates,
                    args.seed + 100 + pair_index * 10 + dataset_index,
                    require_complete_parse=True,
                ),
            }
            if dataset == "truthful_qa_mc1":
                comparison["chance_adjusted_kappa"] = chance_adjusted_kappa_bootstrap(
                    fp_rows,
                    nf_rows,
                    args.bootstrap_replicates,
                    args.seed + 200 + pair_index,
                )
            report["paired_comparisons"].append(comparison)

    analysis_dir = root / "analysis"
    atomic_json(analysis_dir / "paper_metrics.json", report)
    with (analysis_dir / "paper_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["model", "dataset", "metric", "fp16", "nf4", "delta", "ci95_low", "ci95_high"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        condition_lookup = {
            (item["model"], item["dataset"], item["precision"]): item for item in report["conditions"]
        }
        for comparison in report["paired_comparisons"]:
            for metric, interval in comparison["metrics"].items():
                writer.writerow(
                    {
                        "model": comparison["model"],
                        "dataset": comparison["dataset"],
                        "metric": metric,
                        "fp16": json_safe(condition_lookup[(comparison["model"], comparison["dataset"], "fp16")][metric]),
                        "nf4": json_safe(condition_lookup[(comparison["model"], comparison["dataset"], "nf4")][metric]),
                        "delta": interval["delta_nf4_minus_fp16"],
                        "ci95_low": interval["ci95_low"],
                        "ci95_high": interval["ci95_high"],
                    }
                )
    print(f"Wrote {analysis_dir / 'paper_metrics.json'} and {analysis_dir / 'paper_metrics.csv'}")


if __name__ == "__main__":
    main()
