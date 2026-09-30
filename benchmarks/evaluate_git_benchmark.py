"""Evaluate file-level retrieval rankings against Git-history targets.

This tool consumes prepared parent-state worktrees with smart-context indexes in
each worktree's local cache/. It does not create worktrees or indexes. Retrieval
uses the configured embedding provider when the evaluator is explicitly run.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR / "scripts"))
from retrieve import CONFIG, retrieve  # noqa: E402


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BENCHMARK_DIR / "generated" / "benchmark_clean.jsonl"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "generated" / "evaluation"


def deduplicate_ranked_files(chunks):
    """Convert ranked chunks to ranked files, keeping each file's first rank."""
    files = []
    seen = set()
    for item in chunks:
        chunk = item.get("chunk", item) if isinstance(item, dict) else {}
        path = str(chunk.get("file", "")).replace("\\", "/")
        if path and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def score_ranked_files(ranked_files, targets):
    target_set = {str(path).replace("\\", "/") for path in targets}
    target_ranks = [index for index, path in enumerate(ranked_files, 1) if path in target_set]
    first_rank = min(target_ranks) if target_ranks else None
    hits_at = {f"hit@{k}": int(first_rank is not None and first_rank <= k) for k in (1, 3, 5)}
    recall_at_5 = sum(1 for path in ranked_files[:5] if path in target_set) / len(target_set) if target_set else 0.0
    return {
        "first_relevant_rank": first_rank,
        **hits_at,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "recall@5": recall_at_5,
        "returned_unique_files": ranked_files,
        "target_files": sorted(target_set),
    }


def mean_metrics(rows):
    metrics = ("hit@1", "hit@3", "hit@5", "reciprocal_rank", "recall@5")
    if not rows:
        return {metric: 0.0 for metric in metrics}
    return {metric: statistics.mean(row[metric] for row in rows) for metric in metrics}


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_parent_worktree(worktree, expected_parent):
    actual = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if actual != expected_parent:
        raise ValueError(f"Worktree {worktree} is at {actual}, expected parent {expected_parent}")


def retrieve_with_mode(query, root, *, disable_git_recency, top_k=None, max_tokens=None):
    """Override recency only during this call, then restore in-memory config."""
    original_weights = CONFIG["weights"]
    if disable_git_recency:
        CONFIG["weights"] = {**original_weights, "git_recency": 0.0}
    try:
        return retrieve(query, top_k=top_k, max_tokens=max_tokens, root=root)
    finally:
        CONFIG["weights"] = original_weights


def evaluate_mode(records, worktrees_dir, *, disable_git_recency, top_k=None, max_tokens=None):
    rows = []
    label = "No Git Recency" if disable_git_recency else "Full"
    for index, record in enumerate(records, 1):
        parent = record.get("parent")
        if not parent:
            raise ValueError(f"Commit {record.get('commit')} has no parent-state to evaluate")
        worktree = worktrees_dir / parent
        if not worktree.is_dir():
            raise FileNotFoundError(f"Missing prepared parent worktree: {worktree}")
        verify_parent_worktree(worktree, parent)
        print(f"[{label}] {index}/{len(records)} {record['commit'][:10]}", flush=True)
        result = retrieve_with_mode(
            record["query"], worktree,
            disable_git_recency=disable_git_recency,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        ranked_files = deduplicate_ranked_files(result.get("selected", []))
        row = {
            "commit": record["commit"],
            "parent": parent,
            "query": record["query"],
            "mode": label,
            **score_ranked_files(ranked_files, record["targets"]),
            "context_tokens": result.get("estimated_tokens"),
        }
        rows.append(row)
    return rows


def comparison_table(full, no_recency):
    fields = [
        ("Hit@1", "hit@1"), ("Hit@3", "hit@3"), ("Hit@5", "hit@5"),
        ("MRR", "reciprocal_rank"), ("Recall@5", "recall@5"),
    ]
    summary = {"Full": mean_metrics(full), "No Git Recency": mean_metrics(no_recency)}
    print(f"{'Mode':<20}" + "".join(f"{title:>11}" for title, _ in fields), flush=True)
    for mode in ("Full", "No Git Recency"):
        values = summary[mode]
        print(f"{mode:<20}" + "".join(f"{values[key]:>11.3f}" for _, key in fields), flush=True)
    delta = {key: summary["No Git Recency"][key] - summary["Full"][key] for _, key in fields}
    print(f"{'Delta (no-recency - full)':<20}" + "".join(f"{delta[key]:>11.3f}" for _, key in fields), flush=True)
    return summary, delta


def write_results(output_dir, full, no_recency, summary, delta):
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {"summary": summary, "delta_no_recency_minus_full": delta, "per_query": full + no_recency}
    (output_dir / "evaluation_summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ["commit", "parent", "mode", "query", "first_relevant_rank", "hit@1", "hit@3", "hit@5", "reciprocal_rank", "recall@5", "returned_unique_files", "target_files", "context_tokens"]
    with (output_dir / "evaluation_per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in full + no_recency:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], list) else row.get(key) for key in columns})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--worktrees-dir", type=Path, required=True, help="Directory containing one prepared parent worktree named by full parent SHA")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    records = [r for r in load_jsonl(args.dataset) if r.get("category") == "clean" and (args.split == "all" or r.get("split") == args.split)]
    if not records:
        raise SystemExit(f"No clean examples found for split={args.split}")
    worktrees_dir = args.worktrees_dir.resolve()
    print(f"[START] evaluating {len(records)} {args.split} examples in parent-state worktrees", flush=True)
    full = evaluate_mode(records, worktrees_dir, disable_git_recency=False, top_k=args.top_k, max_tokens=args.max_tokens)
    no_recency = evaluate_mode(records, worktrees_dir, disable_git_recency=True, top_k=args.top_k, max_tokens=args.max_tokens)
    summary, delta = comparison_table(full, no_recency)
    write_results(args.output_dir.resolve(), full, no_recency, summary, delta)
    print(f"[DONE] results saved to {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"evaluation: {exc}", file=sys.stderr)
        sys.exit(1)
