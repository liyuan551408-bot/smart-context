"""Build independent Git-history datasets and a combined multi-repository benchmark."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from build_git_benchmark import (
    classify_example,
    dedup_summary,
    deduplicate_clean_records,
    find_query_duplicates,
    git,
    first_meaningful_line,
    jsonl_write,
    noisy_reason_distribution,
    normalized_exact_query,
    parse_name_status_z,
    query_signature,
    summarize_changes,
    write_outputs,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORIES = {
    "CourseCompass": {
        "path": Path(os.environ.get("SMART_CONTEXT_COURSECOMPASS_PATH",
                                   BENCHMARK_DIR / "repos" / "coursecompass")),
        "url": None,
        "stack": "Vue / Node.js / JavaScript",
        "output": "generated_coursecompass",
    },
    "Pinia": {
        "path": BENCHMARK_DIR / "repos" / "pinia",
        "url": "https://github.com/vuejs/pinia.git",
        "stack": "Vue / TypeScript",
        "output": "generated_pinia",
    },
    "Express": {
        "path": BENCHMARK_DIR / "repos" / "express",
        "url": "https://github.com/expressjs/express.git",
        "stack": "Node.js / JavaScript",
        "output": "generated_express",
    },
    "Flask": {
        "path": BENCHMARK_DIR / "repos" / "flask",
        "url": "https://github.com/pallets/flask.git",
        "stack": "Python",
        "output": "generated_flask",
    },
}
COMBINED_OUTPUT_DIR = BENCHMARK_DIR / "generated_multi_repo"
CACHE_DIR = BENCHMARK_DIR / "cache" / "git_history"
CACHE_SCHEMA_VERSION = 2


def repository_url(name, config):
    if config["url"]:
        return config["url"]
    try:
        result = subprocess.run(
            ["git", "-C", str(config["path"]), "remote", "get-url", "origin"],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "local:" + str(config["path"])


def collect_history_batched(repo, limit=None):
    """Read all reachable non-merge commit metadata and path diffs in one Git process."""
    commits = list(dict.fromkeys(git(repo, ["rev-list", "--reverse", "--no-merges", "HEAD"]).splitlines()))
    if limit is not None:
        commits = commits[:max(0, limit)]
    print(f"[START] scanning {len(commits)} all-non-merge commits", flush=True)
    if not commits:
        return commits, [], 0
    args = ["log", "--root", "-M", "--name-status", "-z", "--format=%x1e%H%x00%P%x00%cI%x00%cN%x00%B"]
    if limit is None:
        args.extend(["--reverse", "--no-merges", "HEAD"])
    else:
        # These SHAs were selected from the HEAD-only rev-list above; no parent walk occurs.
        args.extend(["--no-walk=unsorted", *commits])
    output = git(repo, args, text=False)
    entries = output.split(b"\x1e")
    if entries and entries[0] == b"":
        entries = entries[1:]
    if len(entries) != len(commits):
        raise ValueError(f"Batched Git log returned {len(entries)} commit records; expected {len(commits)}")
    records = []
    parsed_shas = []
    for position, (expected_sha, entry) in enumerate(zip(commits, entries), 1):
        fields = entry.split(b"\x00", 4)
        if len(fields) != 5:
            raise ValueError("Malformed batched Git commit record")
        sha = fields[0].decode("ascii")
        if sha != expected_sha:
            raise ValueError("Batched git log did not match the deduplicated HEAD history walk")
        parents = fields[1].decode("ascii").split()
        timestamp = fields[2].decode("utf-8", errors="replace")
        author = fields[3].decode("utf-8", errors="replace")
        message, separator, status_data = fields[4].partition(b"\x00")
        # `git log --name-status -z` inserts a line break between the pretty record and first status.
        status_data = status_data.lstrip(b"\r\n")
        try:
            changes = parse_name_status_z(status_data) if separator and status_data else []
        except ValueError as error:
            raise ValueError(f"{error}; commit={sha}; status_data={status_data[:240]!r}") from error
        query = first_meaningful_line(message.decode("utf-8", errors="replace"))
        change_summary = summarize_changes(changes)
        category, reason, word_count, meaningful_count = classify_example(query, change_summary)
        records.append({
            "commit": sha,
            "parent": parents[0] if parents else None,
            "timestamp": timestamp,
            "author": author,
            "query": query,
            **change_summary,
            "target_count": len(change_summary["targets"]),
            "changed_file_count": len(change_summary["all_changed_files"]),
            "category": category,
            "split": "excluded",
            "clean_reason": reason,
            "message_word_count": word_count,
            "meaningful_word_count": meaningful_count,
            "evaluation_type": "new_file" if category == "new_file" else "retrieval",
        })
        parsed_shas.append(sha)
        if position % 500 == 0 or position == len(commits):
            print(f"[PROGRESS] {position}/{len(commits)}", flush=True)
    if parsed_shas != commits:
        raise ValueError("Batched Git log did not match the deduplicated HEAD history walk")
    return commits, records, len(set(git(repo, ["rev-list", "--merges", "HEAD"]).splitlines()))


def repository_head(repo):
    return git(repo, ["rev-parse", "HEAD"]).strip()


def history_cache_path(name, head_sha, history_mode="all-non-merge"):
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")
    return CACHE_DIR / f"{safe_name}_{head_sha}_{history_mode}.json"


def load_or_collect_history(repo, name, sample_limit=None):
    """Cache commit messages and name-status paths, never repository source contents."""
    head_sha = repository_head(repo)
    cache_path = history_cache_path(name, head_sha)
    if sample_limit is None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (cached.get("schema_version") == CACHE_SCHEMA_VERSION
                and cached.get("repository") == str(Path(repo).resolve())
                and cached.get("head_sha") == head_sha
                and cached.get("history_mode") == "all-non-merge"):
            print(f"[CACHE HIT] {name} {head_sha}", flush=True)
            return cached["commits"], cached["records"], cached["merge_commits_excluded"], True
    started = time.perf_counter()
    commits, records, merge_count = collect_history_batched(repo, limit=sample_limit)
    elapsed = time.perf_counter() - started
    if sample_limit is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "schema_version": CACHE_SCHEMA_VERSION,
            "repository": str(Path(repo).resolve()),
            "repository_name": name,
            "head_sha": head_sha,
            "history_mode": "all-non-merge",
            "commits": commits,
            "merge_commits_excluded": merge_count,
            "records": records,
        }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"[CACHE WRITE] {name} {cache_path} ({elapsed:.2f}s)", flush=True)
    return commits, records, merge_count, False


def add_repository_metadata(records, name, url, stack):
    annotated = []
    for record in records:
        row = dict(record)
        row.update({"repository": name, "repository_url": url, "language_or_stack": stack})
        annotated.append(row)
    return annotated


def combine_records(repository_results, key="records"):
    """Concatenate repository records; SHAs are only unique within a repository."""
    return [row for result in repository_results for row in result[key]]


def deduplicate_per_repository(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row["repository"], []).append(row)
    result = []
    for name, rows in grouped.items():
        result.extend(deduplicate_clean_records(rows))
    return result


def duplicate_group_split_violations(records):
    splits = {}
    for row in records:
        if not row.get("benchmark_included"):
            continue
        key = (row.get("repository", ""), row.get("duplicate_group", row["commit"]))
        splits.setdefault(key, set()).add(row.get("split", "excluded"))
    return {key: sorted(values) for key, values in splits.items() if len(values) > 1}


def unique_target_keys(records):
    return {(row["repository"], path) for row in records for path in row.get("targets", [])}


def _directory_key(repository, path):
    normalized = path.replace("\\", "/").strip("/")
    parent = str(Path(normalized).parent).replace("\\", "/")
    return (repository, "" if parent == "." else parent)


def test_diversity(records):
    test_rows = [row for row in records if row.get("split") == "test" and row.get("benchmark_included", True)]
    query_keys = [normalized_exact_query(row["query"]) for row in test_rows]
    target_counts = Counter((row["repository"], target) for row in test_rows for target in set(row["targets"]))
    directory_counts = Counter(
        directory for row in test_rows
        for directory in {_directory_key(row["repository"], target) for target in set(row["targets"])}
    )
    patterns = Counter(" ".join(query_signature(row["query"])) for row in test_rows)
    duplicate_flags = find_query_duplicates(test_rows)
    one_file = sum(len(set(row["targets"])) == 1 for row in test_rows)
    multiple_files = sum(len(set(row["targets"])) > 1 for row in test_rows)
    count = len(test_rows)
    dominant_repo = Counter(row["repository"] for row in test_rows).most_common(1)
    top_target = target_counts.most_common(1)
    top_directory = directory_counts.most_common(1)
    top_pattern = patterns.most_common(1)
    return {
        "test_count": count,
        "unique_test_queries": len(set(query_keys)),
        "unique_target_files": len(unique_target_keys(test_rows)),
        "unique_target_sets": len({(row["repository"], tuple(sorted(set(row["targets"])))) for row in test_rows}),
        "duplicate_or_near_duplicate_query_flags": duplicate_flags,
        "single_target_count": one_file,
        "single_target_percentage": round(100 * one_file / count, 2) if count else 0.0,
        "multiple_target_count": multiple_files,
        "multiple_target_percentage": round(100 * multiple_files / count, 2) if count else 0.0,
        "dominance": {
            "repository": _top_share(dominant_repo, count),
            "target_file": _top_share(top_target, count),
            "directory": _top_share(top_directory, count),
            "commit_message_pattern": _top_share(top_pattern, count),
        },
    }


def _top_share(items, denominator):
    if not items or not denominator:
        return {"value": None, "count": 0, "percentage": 0.0}
    value, count = items[0]
    if isinstance(value, tuple):
        value = "/".join(str(part) for part in value if part)
    return {"value": value, "count": count, "percentage": round(100 * count / denominator, 2)}


def summarize_repository(name, records, clean_dedup, commits_discovered):
    clean_raw = [row for row in records if row["category"] == "clean"]
    included = [row for row in clean_dedup if row["benchmark_included"]]
    stats = dedup_summary(clean_dedup)
    target_counts = [len(set(row["targets"])) for row in included]
    split_counts = Counter(row["split"] for row in included)
    return {
        "repository": name,
        "commits_discovered": commits_discovered,
        "examples_generated": len(records),
        "clean_raw": len(clean_raw),
        "duplicate_groups": stats["duplicate_groups"],
        "clean_deduplicated": len(included),
        "examples_removed_from_evaluation": stats["examples_removed_from_evaluation"],
        "noisy": sum(row["category"] == "noisy" for row in records),
        "new_file": sum(row["category"] == "new_file" for row in records),
        "train": split_counts["train"],
        "validation": split_counts["validation"],
        "test": split_counts["test"],
        "unique_target_files": len(unique_target_keys(included)),
        "average_target_count": round(statistics.mean(target_counts), 3) if target_counts else 0,
        "median_target_count": statistics.median(target_counts) if target_counts else 0,
        "noisy_reason_distribution": noisy_reason_distribution(records),
        "test_diversity": test_diversity(clean_dedup),
    }


def write_repository_outputs(output_dir, records, clean_dedup, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(records, summary, output_dir)
    jsonl_write(output_dir / "benchmark_clean_raw.jsonl", [row for row in records if row["category"] == "clean"])
    jsonl_write(output_dir / "benchmark_clean_dedup.jsonl", clean_dedup)
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_combined_outputs(records, clean_raw, clean_dedup, summaries):
    COMBINED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    included = [row for row in clean_dedup if row["benchmark_included"]]
    jsonl_write(COMBINED_OUTPUT_DIR / "benchmark_all.jsonl", records)
    jsonl_write(COMBINED_OUTPUT_DIR / "benchmark_clean_raw.jsonl", clean_raw)
    jsonl_write(COMBINED_OUTPUT_DIR / "benchmark_clean_dedup.jsonl", clean_dedup)
    for split in ("train", "validation", "test"):
        jsonl_write(COMBINED_OUTPUT_DIR / f"benchmark_{split}.jsonl", [row for row in included if row["split"] == split])
    columns = [
        "repository", "repository_url", "language_or_stack", "commit", "parent", "timestamp", "author",
        "query", "targets", "target_count", "added_files", "modified_files", "deleted_files", "renamed_files",
        "all_changed_files", "changed_file_count", "category", "split", "clean_reason", "message_word_count", "evaluation_type",
    ]
    with (COMBINED_OUTPUT_DIR / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), list) else row.get(key, "") for key in columns})

    totals = combined_totals(records, clean_raw, clean_dedup, summaries)
    summary = {
        "history_mode": "git rev-list --reverse --no-merges HEAD",
        "history_scope": "commits reachable from each repository HEAD; repository-local chronological splits",
        "repositories": summaries,
        "combined": totals,
        "test_diversity": test_diversity(included),
        "duplicate_group_split_violations": [
            {"repository": key[0], "duplicate_group": key[1], "splits": values}
            for key, values in duplicate_group_split_violations(clean_dedup).items()
        ],
        "split_policy": "Within each repository, deduplicate clean examples first, then chronological 70/15/15 split; combined without shuffle.",
    }
    (COMBINED_OUTPUT_DIR / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def combined_totals(records, clean_raw, clean_dedup, summaries):
    included = [row for row in clean_dedup if row["benchmark_included"]]
    return {
        "repositories": len(summaries),
        "commits_discovered": sum(item["commits_discovered"] for item in summaries.values()),
        "examples_generated": len(records),
        "clean_raw": len(clean_raw),
        "duplicate_groups": sum(item["duplicate_groups"] for item in summaries.values()),
        "examples_removed_from_evaluation": sum(item["examples_removed_from_evaluation"] for item in summaries.values()),
        "clean_deduplicated": len(included),
        "noisy": sum(item["noisy"] for item in summaries.values()),
        "new_file": sum(item["new_file"] for item in summaries.values()),
        "train": sum(item["train"] for item in summaries.values()),
        "validation": sum(item["validation"] for item in summaries.values()),
        "test": sum(item["test"] for item in summaries.values()),
        "unique_target_files": len(unique_target_keys(included)),
        "average_target_count": round(statistics.mean(len(set(row["targets"])) for row in included), 3) if included else 0,
        "median_target_count": statistics.median(len(set(row["targets"])) for row in included) if included else 0,
    }


def print_report(summary):
    columns = ["repository", "commits_discovered", "clean_raw", "duplicate_groups", "clean_deduplicated", "noisy", "new_file", "train", "validation", "test", "unique_target_files", "average_target_count", "median_target_count"]
    print("[REPOSITORY SUMMARY]", flush=True)
    print(" | ".join(columns), flush=True)
    for name, row in summary["repositories"].items():
        print(" | ".join(str(row.get(key, name if key == "repository" else "")) for key in columns), flush=True)
        print(f"[NOISY REASONS] {name}", flush=True)
        for reason in row["noisy_reason_distribution"]:
            print(f"{reason['reason']} | {reason['count']} | {reason['percentage']:.2f}%", flush=True)
        print(f"[TEST DIVERSITY] {name} {json.dumps(row['test_diversity'], ensure_ascii=False)}", flush=True)
    print("[COMBINED TOTALS] " + json.dumps(summary["combined"], ensure_ascii=False), flush=True)
    print("[GLOBAL TEST DIVERSITY] " + json.dumps(summary["test_diversity"], ensure_ascii=False), flush=True)
    print(f"[OUTPUT] {COMBINED_OUTPUT_DIR}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", choices=tuple(REPOSITORIES), help="Limit to a repository; repeat to select several.")
    parser.add_argument("--sample-limit", type=int, help="Read only the oldest N reachable non-merge commits and report parser throughput; do not write datasets or cache.")
    args = parser.parse_args(argv)
    selected = args.repo or list(REPOSITORIES)
    if args.sample_limit is not None and len(selected) != 1:
        parser.error("--sample-limit requires exactly one --repo selection")
    results = []
    summaries = {}
    for name in selected:
        config = REPOSITORIES[name]
        repo = config["path"].resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"Repository {name} is not present at {repo}")
        url = repository_url(name, config)
        repository_started = time.perf_counter()
        if args.sample_limit is not None:
            started = time.perf_counter()
            commits, _, _ = collect_history_batched(repo, limit=args.sample_limit)
            elapsed = time.perf_counter() - started
            rate = len(commits) / elapsed if elapsed else 0
            pinia_total = len(git(REPOSITORIES["Pinia"]["path"], ["rev-list", "--reverse", "--no-merges", "HEAD"]).splitlines())
            all_total = sum(len(git(item["path"], ["rev-list", "--reverse", "--no-merges", "HEAD"]).splitlines()) for item in REPOSITORIES.values())
            print(f"[PERFORMANCE] repository={name} commits_processed={len(commits)} elapsed_seconds={elapsed:.3f} commits_per_second={rate:.3f}", flush=True)
            print(f"[ESTIMATE] Pinia {pinia_total} commits: {pinia_total / rate / 60:.2f} minutes; all {all_total} commits: {all_total / rate / 60:.2f} minutes", flush=True)
            return 0
        commits, raw_records, merge_count, cache_hit = load_or_collect_history(repo, name)
        raw_records = add_repository_metadata(raw_records, name, url, config["stack"])
        clean_dedup = deduplicate_per_repository(raw_records)
        per_summary = summarize_repository(name, raw_records, clean_dedup, len(commits))
        per_summary.update({
            "repository_url": url,
            "language_or_stack": config["stack"],
            "repository_path": str(repo),
            "merge_commits_excluded": merge_count,
            "elapsed_build_seconds": round(time.perf_counter() - repository_started, 3),
            "metadata_cache_hit": cache_hit,
        })
        write_repository_outputs(BENCHMARK_DIR / config["output"], raw_records, clean_dedup, per_summary)
        results.append({"name": name, "records": raw_records, "clean_dedup": clean_dedup})
        summaries[name] = per_summary
        print(f"[REPOSITORY DONE] {name}: commits={len(commits)} clean={per_summary['clean_raw']} dedup={per_summary['clean_deduplicated']} noisy={per_summary['noisy']} new_file={per_summary['new_file']}", flush=True)

    all_records = combine_records(results, "records")
    raw_clean = [row for row in all_records if row["category"] == "clean"]
    all_dedup = combine_records(results, "clean_dedup")
    summary = write_combined_outputs(all_records, raw_clean, all_dedup, summaries)
    print_report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
