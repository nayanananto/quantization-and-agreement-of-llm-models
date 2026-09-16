"""Run paired FP16/NF4 homogeneous-panel experiments on MC and free response."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache


def atomic_json(path: Path, value: Any) -> None:
    # Two GPU shards may write identical shared metadata concurrently. A
    # process-specific temporary path keeps their atomic replaces independent.
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".10g")
    except ValueError:
        return None


def load_examples(name: str, limit: int) -> list[dict[str, Any]]:
    if name == "truthful_qa_mc1":
        data = load_dataset("truthful_qa", "multiple_choice", split="validation")
        examples = []
        for index, row in enumerate(data.select(range(min(limit, len(data))))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            answer = chr(65 + labels.index(1))
            options = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
            examples.append(
                {
                    "id": f"truthful_qa_mc1-{index}",
                    "prompt": (
                        "Answer the multiple-choice question by returning only the single option letter. "
                        "Do not explain your answer.\n\n"
                        f"{row['question']}\n{options}"
                    ),
                    "gold": answer,
                    "kind": "mc",
                    "num_choices": len(choices),
                }
            )
        return examples
    if name == "gsm8k":
        data = load_dataset("openai/gsm8k", "main", split="test")
        examples = []
        for index, row in enumerate(data.select(range(min(limit, len(data))))):
            gold = normalize_number(row["answer"].split("####")[-1])
            examples.append(
                {
                    "id": f"gsm8k-{index}",
                    "prompt": (
                        "Solve the problem step by step. End with exactly FINAL: <number>. "
                        "The final marker is required.\n\n"
                        f"{row['question']}"
                    ),
                    "gold": gold,
                    "kind": "number",
                }
            )
        return examples
    raise ValueError(f"Unknown dataset: {name}")


def extract_answer(text: str, example: dict[str, Any]) -> str | None:
    if example["kind"] == "mc":
        # Some models emit a valid "Final Answer: B" and then append an empty
        # "FINAL:" token. Select the last explicit answer-bearing marker rather
        # than blindly taking the text after the final marker.
        explicit_patterns = (
            r"(?i)\bfinal\s*(?:answer)?\s*:\s*(?:option\s*)?[\[(]?([A-Z])\b",
            r"(?i)\b(?:the\s+)?correct\s+answer\s*(?:is)?\s*:?\s*(?:option\s*)?[\[(]?([A-Z])\b",
        )
        explicit = [match for pattern in explicit_patterns for match in re.finditer(pattern, text)]
        if explicit:
            answer = max(explicit, key=lambda match: match.start()).group(1).upper()
            return answer if ord(answer) - 65 < example["num_choices"] else None
        final = text.rsplit("FINAL:", 1)[-1].strip() if "FINAL:" in text else text
        match = re.search(r"\b([A-Z])\b", final.upper())
        if not match:
            return None
        answer = match.group(1)
        return answer if ord(answer) - 65 < example["num_choices"] else None
    # Never score the last intermediate number from a truncated derivation as
    # the final answer. A free-response vote is valid only with an explicit
    # final-answer marker.
    final_matches = list(
        re.finditer(
            r"(?i)\bfinal\s*(?:answer)?\s*(?::|is)\s*([-+]?\d[\d,]*(?:\.\d+)?)",
            text,
        )
    )
    return normalize_number(final_matches[-1].group(1)) if final_matches else None


def load_model(model_name: str, precision: str):
    # Phi's compact remote implementation avoids importing optional vision
    # components on text-only Kaggle images. Restore its deprecated cache alias
    # explicitly when using a modern Transformers release.
    use_remote_code = model_name.startswith("microsoft/Phi-3.5")
    phi_revision = "2fe192450127e6a83f7441aef6e3ca586c338b77" if use_remote_code else None
    if use_remote_code and not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda cache: cache.get_seq_length())
    if use_remote_code and not hasattr(DynamicCache, "get_max_length"):
        # The legacy Phi implementation treats None as an unbounded dynamic
        # cache, matching the behavior of the former Transformers API.
        DynamicCache.get_max_length = lambda cache: None
    if use_remote_code and not hasattr(DynamicCache, "get_usable_length"):
        # Phi-3.5's remote modeling code predates the modern Cache API. Mirror
        # the removed method instead of discovering missing aliases one run at
        # a time. DynamicCache is unbounded, so this normally returns the
        # sequence length already stored for the requested layer.
        def legacy_get_usable_length(cache, new_seq_length: int, layer_idx: int = 0) -> int:
            previous_seq_length = cache.get_seq_length(layer_idx)
            max_length = cache.get_max_length()
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length

        DynamicCache.get_usable_length = legacy_get_usable_length
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": use_remote_code}
    if phi_revision is not None:
        tokenizer_kwargs["revision"] = phi_revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": use_remote_code}
    if use_remote_code:
        # Keep Phi on the path that uses the audited Cache methods above; the
        # optional FlashAttention path also indexes legacy cache objects.
        kwargs["attn_implementation"] = "eager"
        kwargs["revision"] = phi_revision
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def generate_one(tokenizer, model, prompt: str, seed: int, config: dict[str, Any]) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_new_tokens=config["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def summarize(rows: list[dict[str, Any]], panel_size: int) -> dict[str, Any]:
    individual = [sample["correct"] for row in rows for sample in row["samples"]]
    cofailures = 0
    same_wrong = 0
    panel_correct = 0
    confidences = []
    confidence_correct = []
    for row in rows:
        # Invalid parses are not evidence that agents selected the same answer.
        answers = [
            sample["answer"] if sample["answer"] is not None else f"__invalid_{sample['member']}"
            for sample in row["samples"]
        ]
        counts = Counter(answers)
        panel_answer, votes = counts.most_common(1)[0]
        correct = panel_answer == row["gold"]
        panel_correct += int(correct)
        confidences.append(votes / panel_size)
        confidence_correct.append(int(correct))
        all_wrong = all(not sample["correct"] for sample in row["samples"])
        cofailures += int(all_wrong)
        valid_answers = [sample["answer"] for sample in row["samples"]]
        same_wrong += int(all_wrong and None not in valid_answers and len(set(valid_answers)) == 1)
    n = len(rows)
    brier = float(np.mean([(c - y) ** 2 for c, y in zip(confidences, confidence_correct)]))
    return {
        "questions": n,
        "individual_accuracy": sum(individual) / len(individual),
        "panel_accuracy": panel_correct / n,
        "beta_all_agents_wrong": cofailures / n,
        "same_wrong_all_rate": same_wrong / n,
        "kappa_same_wrong_given_all_wrong": same_wrong / cofailures if cofailures else None,
        "agreement_brier": brier,
        "cofailure_count": cofailures,
        "same_wrong_count": same_wrong,
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def run_condition(model_name: str, precision: str, dataset_spec: dict[str, Any], config: dict[str, Any], output: Path) -> dict[str, Any]:
    condition = f"{safe_name(model_name)}__{precision}__{dataset_spec['name']}"
    path = output / f"{condition}.jsonl"
    done_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        done_ids = {row["id"] for row in rows}

    examples = load_examples(dataset_spec["name"], dataset_spec["limit"])
    tokenizer, model = load_model(model_name, precision)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        for question_index, example in enumerate(examples):
            if example["id"] in done_ids:
                continue
            samples = []
            for member in range(config["panel_size"]):
                seed = config["base_seed"] + question_index * config["panel_size"] + member
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                generation_config = {**config, **dataset_spec}
                text = generate_one(tokenizer, model, example["prompt"], seed, generation_config)
                answer = extract_answer(text, example)
                samples.append({"member": member, "seed": seed, "answer": answer, "correct": answer == example["gold"], "text": text})
            row = {"id": example["id"], "gold": example["gold"], "samples": samples}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)

    del model
    torch.cuda.empty_cache()
    return {"condition": condition, "model": model_name, "precision": precision, "dataset": dataset_spec["name"], **summarize(rows, config["panel_size"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "resolved_config.json", config)
    atomic_json(
        args.output_dir / "environment.json",
        {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "cuda": torch.version.cuda,
        },
    )
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    summary_name = "summary.json" if args.num_shards == 1 else f"summary_shard_{args.shard_index}.json"
    summary_path = args.output_dir / summary_name
    summaries = load_json(summary_path, []) if summary_path.exists() else []
    finished = {item["condition"] for item in summaries}
    if "conditions" in config:
        conditions = config["conditions"]
    else:
        conditions = [
            {"model": model, "precision": precision, "dataset": dataset}
            for model in config["models"]
            for precision in config["precisions"]
            for dataset in config["datasets"]
        ]
    for condition_index, specification in enumerate(conditions):
        if condition_index % args.num_shards != args.shard_index:
            continue
        model_name = specification["model"]
        precision = specification["precision"]
        dataset_spec = specification["dataset"]
        condition = f"{safe_name(model_name)}__{precision}__{dataset_spec['name']}"
        if condition in finished:
            continue
        result = run_condition(model_name, precision, dataset_spec, config, args.output_dir)
        summaries.append(result)
        atomic_json(summary_path, summaries)


def load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


if __name__ == "__main__":
    main()
