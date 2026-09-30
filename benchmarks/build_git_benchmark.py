"""Build a weakly supervised, parent-state file retrieval dataset from Git history."""
import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


DEFAULT_REPO = Path.cwd()
OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
ALL_NONMERGE_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_all_nonmerge"
SOURCE_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx", ".vue", ".py", ".java", ".kt", ".go",
    ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".html", ".css", ".scss",
    ".sql", ".prisma",
}
IGNORED_DIRS = {
    "node_modules", "dist", "build", "coverage", "cache", "vendor", "target",
    ".venv", "venv", "env", "virtualenv", "__pycache__", ".tox",
}
LOCKFILES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml",
    "bun.lock", "bun.lockb", "cargo.lock", "poetry.lock", "uv.lock",
    "composer.lock", "gemfile.lock", "pipfile.lock", "go.sum",
}
GENERIC_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "change", "changes",
    "cleanup", "clean", "final", "fix", "fixed", "fixes", "for", "from", "in",
    "minor", "misc", "of", "on", "or", "small", "some", "test", "tests", "the",
    "to", "update", "updated", "updates", "wip", "work", "bug", "bugs", "stuff",
    "refactor", "version", "various",
}
QUERY_ACTION_WORDS = GENERIC_WORDS | {
    "add", "added", "adding", "adjust", "adjusted", "build", "built", "change",
    "create", "created", "enable", "enabled", "expand", "fixing", "improve",
    "improved", "implement", "implemented", "introduce", "make", "made", "remove",
    "removed", "revise", "revised", "support", "update", "upgrade", "again",
}


def git(repo, args, *, text=True):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        text=text, encoding="utf-8" if text else None, errors="replace" if text else None,
    )
    return result.stdout


def is_merge_commit(parents):
    return len(parents) > 1


def first_meaningful_line(message):
    for line in message.splitlines():
        normalized = " ".join(line.strip().split())
        if normalized:
            return normalized
    return ""


def message_quality(query):
    """Return (is_weak, reason, lexical_count, meaningful_count)."""
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    meaningful = [word for word in words if word not in GENERIC_WORDS and len(word) > 1]
    if not query:
        return True, "empty first-line commit message", 0, 0
    if not meaningful:
        return True, "generic commit message", len(words), 0
    if len(meaningful) < 3:
        return True, f"only {len(meaningful)} meaningful message words (minimum 3)", len(words), len(meaningful)
    return False, "", len(words), len(meaningful)


def is_source_path(path):
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = normalized.split("/")
    if any(part.lower() in IGNORED_DIRS for part in parts[:-1]):
        return False
    basename = parts[-1].lower()
    if basename in LOCKFILES or basename.endswith(".lock"):
        return False
    return Path(basename).suffix.lower() in SOURCE_EXTENSIONS


def parse_name_status_z(data):
    """Parse `git diff --name-status -z` output, retaining both rename paths."""
    fields = data.decode("utf-8", errors="surrogateescape").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    changes = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status.startswith(("R", "C")):
            if index + 1 >= len(fields):
                raise ValueError("Malformed rename/copy entry in Git name-status output")
            old_path, new_path = fields[index], fields[index + 1]
            index += 2
            changes.append({"status": status, "old_path": old_path, "path": new_path})
        else:
            if index >= len(fields):
                raise ValueError("Malformed path entry in Git name-status output")
            path = fields[index]
            index += 1
            changes.append({"status": status, "old_path": path if status == "D" else "", "path": path})
    return changes


def summarize_changes(changes):
    added, modified, deleted, renamed, targets, all_paths, statuses = [], [], [], [], [], [], []
    for change in changes:
        status = change["status"]
        path = change["path"].replace("\\", "/")
        old_path = change.get("old_path", "").replace("\\", "/")
        relevant = is_source_path(path) or (old_path and is_source_path(old_path))
        if not relevant:
            continue
        if status.startswith("R"):
            if is_source_path(old_path):
                renamed.append(old_path)
                targets.append(old_path)
            for item in (old_path, path):
                if is_source_path(item):
                    all_paths.append(item)
            statuses.append({"status": status, "old_path": old_path, "path": path})
        elif status.startswith("C") or status == "A":
            if is_source_path(path):
                added.append(path)
                all_paths.append(path)
                statuses.append({"status": status, "old_path": old_path, "path": path})
        elif status == "D":
            if is_source_path(path):
                deleted.append(path)
                targets.append(path)
                all_paths.append(path)
                statuses.append({"status": status, "old_path": path, "path": path})
        elif status in {"M", "T", "U"}:
            if is_source_path(path):
                modified.append(path)
                targets.append(path)
                all_paths.append(path)
                statuses.append({"status": status, "old_path": "", "path": path})
    def unique(values):
        return sorted(set(values))
    return {
        "added_files": unique(added), "modified_files": unique(modified),
        "deleted_files": unique(deleted), "renamed_files": unique(renamed),
        "targets": unique(targets), "all_changed_files": unique(all_paths),
        "file_statuses": statuses,
    }


def classify_example(query, summary):
    weak, weak_reason, word_count, meaningful_count = message_quality(query)
    targets = summary["targets"]
    if weak:
        return "noisy", "noisy: " + weak_reason, word_count, meaningful_count
    if summary["added_files"] and not targets:
        return "new_file", "added source paths did not exist in the parent; excluded from retrieval splits", word_count, meaningful_count
    if not targets:
        return "noisy", "no eligible parent-state source targets", word_count, meaningful_count
    if len(targets) > 5:
        return "noisy", f"too broad: {len(targets)} parent-state source targets (maximum 5)", word_count, meaningful_count
    return "clean", "meaningful query with 1-5 parent-state source targets", word_count, meaningful_count


def assign_chronological_splits(records):
    clean_indices = [
        i for i, record in enumerate(records)
        if record["category"] == "clean" and record.get("benchmark_included", True)
    ]
    train_end = int(len(clean_indices) * 0.70)
    validation_end = int(len(clean_indices) * 0.85)
    for record in records:
        record["split"] = "excluded"
    for rank, index in enumerate(clean_indices):
        records[index]["split"] = "train" if rank < train_end else ("validation" if rank < validation_end else "test")
    return records


def normalized_exact_query(query):
    """Normalize punctuation and case while retaining the query's lexical words."""
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.lower()))


def deduplicate_clean_records(records, overlap_threshold=0.25):
    """Mark iterative clean examples and assign splits only to group representatives."""
    clean = [dict(record) for record in records if record["category"] == "clean"]
    parent = list(range(len(clean)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    exact_queries = {}
    for index, record in enumerate(clean):
        normalized = normalized_exact_query(record["query"])
        if normalized:
            repo_key = record.get("repository", "")
            key = (repo_key, normalized)
            if key in exact_queries:
                union(index, exact_queries[key])
            else:
                exact_queries[key] = index

    signatures = [set(query_signature(record["query"])) for record in clean]
    for left in range(len(clean)):
        for right in range(left + 1, len(clean)):
            if clean[left].get("repository", "") != clean[right].get("repository", ""):
                continue
            shared_targets = set(clean[left]["targets"]) & set(clean[right]["targets"])
            union_terms = signatures[left] | signatures[right]
            overlap = len(signatures[left] & signatures[right]) / len(union_terms) if union_terms else 0.0
            if shared_targets and overlap > overlap_threshold:
                union(left, right)

    groups = {}
    for index in range(len(clean)):
        groups.setdefault(find(index), []).append(index)

    def representative_key(index):
        record = clean[index]
        signature = query_signature(record["query"])
        meaningful_count = record.get("meaningful_word_count", message_quality(record["query"])[3])
        generic_count = sum(word in GENERIC_WORDS for word in re.findall(r"[a-z0-9]+", record["query"].lower()))
        # Higher meaningful and distinctive lexical content wins; oldest breaks ties.
        return (-meaningful_count, -len(set(signature)), generic_count, record.get("timestamp", ""), record["commit"])

    for members in groups.values():
        representative_index = min(members, key=representative_key)
        representative = clean[representative_index]
        group_id = representative["commit"]
        for index in members:
            record = clean[index]
            record["benchmark_included"] = index == representative_index
            record["duplicate_group"] = group_id
            if index == representative_index:
                record["duplicate_of"] = None
                record["duplicate_reason"] = "representative"
            else:
                exact = normalized_exact_query(record["query"]) == normalized_exact_query(representative["query"])
                record["duplicate_of"] = group_id
                record["duplicate_reason"] = "exact normalized query duplicate" if exact else "near-duplicate query with shared target file"
                record["split"] = "excluded"

    assign_chronological_splits(clean)
    return clean


def dedup_summary(records):
    included = [record for record in records if record["benchmark_included"]]
    excluded = [record for record in records if not record["benchmark_included"]]
    grouped = {}
    for record in records:
        grouped.setdefault(record["duplicate_group"], []).append(record)
    group_count = sum(len(members) > 1 for members in grouped.values())
    split_summary = {}
    for split in ("train", "validation", "test"):
        split_rows = [record for record in included if record["split"] == split]
        split_summary[split] = {
            "count": len(split_rows),
            "unique_target_files": len({target for record in split_rows for target in record["targets"]}),
        }
    test_rows = [record for record in included if record["split"] == "test"]
    target_sets = {tuple(sorted(record["targets"])) for record in test_rows}
    return {
        "raw_clean_count": len(records),
        "duplicate_groups": group_count,
        "examples_removed_from_evaluation": len(excluded),
        "deduplicated_clean_count": len(included),
        "train_count": split_summary["train"]["count"],
        "validation_count": split_summary["validation"]["count"],
        "test_count": split_summary["test"]["count"],
        "unique_target_files_by_split": {name: values["unique_target_files"] for name, values in split_summary.items()},
        "test_target_sets_distinct": len(target_sets) == len(test_rows),
        "test_unique_target_set_count": len(target_sets),
    }


def print_deduplicated_dataset(records, summary):
    print("[DEDUP SUMMARY] " + json.dumps(summary, ensure_ascii=False), flush=True)
    for record in records:
        if record["benchmark_included"] and record["split"] == "test":
            print("[DEDUP TEST] " + json.dumps({key: record[key] for key in ("query", "commit", "parent", "targets")}, ensure_ascii=False), flush=True)


def history_commits(repo, history_mode):
    if history_mode == "first-parent":
        args = ["rev-list", "--first-parent", "--reverse", "HEAD"]
    elif history_mode == "all-non-merge":
        args = ["rev-list", "--reverse", "--no-merges", "HEAD"]
    else:
        raise ValueError(f"Unsupported history mode: {history_mode}")
    # Preserve Git's chronological ordering while ensuring each object is emitted once.
    return list(dict.fromkeys(git(repo, args).splitlines()))


def collect_history(repo, history_mode="first-parent"):
    commits = history_commits(repo, history_mode)
    records = []
    skipped_merges = 0
    print(f"[START] scanning {len(commits)} {history_mode} commits", flush=True)
    for position, sha in enumerate(commits, 1):
        fields = git(repo, ["show", "-s", "--format=%P%x00%cI%x00%cN%x00%B", sha]).split("\0", 3)
        parents = fields[0].split() if fields[0].strip() else []
        if is_merge_commit(parents):
            skipped_merges += 1
        else:
            parent = parents[0] if parents else ""
            diff_args = ["diff-tree", "--root", "--no-commit-id", "-r", "-M", "--name-status", "-z", sha] if not parent else ["diff-tree", "--no-commit-id", "-r", "-M", "--name-status", "-z", parent, sha]
            changes = parse_name_status_z(git(repo, diff_args, text=False))
            query = first_meaningful_line(fields[3] if len(fields) > 3 else "")
            summary = summarize_changes(changes)
            category, reason, word_count, meaningful_count = classify_example(query, summary)
            records.append({
                "commit": sha,
                "parent": parent or None,
                "timestamp": fields[1],
                "author": fields[2],
                "query": query,
                **summary,
                "target_count": len(summary["targets"]),
                "changed_file_count": len(summary["all_changed_files"]),
                "category": category,
                "split": "excluded",
                "clean_reason": reason,
                "message_word_count": word_count,
                "meaningful_word_count": meaningful_count,
                "evaluation_type": "new_file" if category == "new_file" else "retrieval",
            })
        if position % 25 == 0 or position == len(commits):
            print(f"[PROGRESS] {position}/{len(commits)}", flush=True)
    assign_chronological_splits(records)
    return commits, records, skipped_merges


def jsonl_write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def noisy_reason_distribution(records):
    noisy = [record for record in records if record["category"] == "noisy"]
    counts = Counter(record["clean_reason"] for record in noisy)
    total = len(noisy)
    return [
        {"reason": reason, "count": count, "percentage": round(100.0 * count / total, 2) if total else 0.0}
        for reason, count in counts.most_common()
    ]


def query_signature(query):
    words = [word for word in re.findall(r"[a-z0-9]+", query.lower()) if word not in QUERY_ACTION_WORDS]
    return tuple(word[:-1] if len(word) > 4 and word.endswith("s") else word for word in words)


def find_query_duplicates(records):
    """Flag action-word variants and similar queries on shared targets; never remove records."""
    signatures = [set(query_signature(record["query"])) for record in records]
    candidates = []
    exact_groups = {}
    for index, signature in enumerate(signatures):
        if signature:
            exact_groups.setdefault(tuple(sorted(signature)), []).append(index)
    grouped_pairs = set()
    for signature, members in exact_groups.items():
        if len(members) < 2:
            continue
        grouped_pairs.update((min(a, b), max(a, b)) for offset, a in enumerate(members) for b in members[offset + 1:])
        member_records = [records[i] for i in members]
        candidates.append({
            "normalized_subject": " ".join(signature),
            "count": len(members),
            "review_reason": "flagged for manual review: action-word-normalized duplicate query subject",
            "examples": [{"commit": m["commit"], "query": m["query"], "category": m["category"], "split": m["split"], "targets": m["targets"]} for m in member_records],
        })
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if (left, right) in grouped_pairs or not signatures[left] or not signatures[right]:
                continue
            shared_targets = sorted(set(records[left]["targets"]) & set(records[right]["targets"]))
            target_union = set(records[left]["targets"]) | set(records[right]["targets"])
            target_overlap = len(shared_targets) / len(target_union) if target_union else 0.0
            union_terms = signatures[left] | signatures[right]
            overlap = len(signatures[left] & signatures[right]) / len(union_terms) if union_terms else 0.0
            if shared_targets and overlap >= 0.25 and target_overlap >= 0.5:
                shared_terms = " ".join(sorted(signatures[left] & signatures[right]))
                pair = [records[left], records[right]]
                candidates.append({
                    "normalized_subject": shared_terms or "similar query terms",
                    "count": 2,
                    "review_reason": "flagged for manual review: similar query terms and shared target paths",
                    "shared_targets": shared_targets,
                    "target_overlap": round(target_overlap, 3),
                    "examples": [{"commit": m["commit"], "query": m["query"], "category": m["category"], "split": m["split"], "targets": m["targets"]} for m in pair],
                })
    return sorted(candidates, key=lambda item: (-item["count"], item["normalized_subject"]))


def build_summary(total_scanned, records, skipped_merges, repo, history_mode):
    category_counts = Counter(record["category"] for record in records)
    split_counts = Counter(record["split"] for record in records if record["split"] != "excluded")
    clean = [record for record in records if record["category"] == "clean"]
    targets = [record["target_count"] for record in clean]
    changed_distribution = Counter(str(record["changed_file_count"]) for record in records)
    extensions = Counter(Path(path).suffix.lower() for record in records for path in record["all_changed_files"])
    exclusions = Counter(record["clean_reason"] for record in records if record["category"] != "clean")
    return {
        "repository": str(Path(repo).resolve()),
        "history_mode": history_mode,
        "history_order": "oldest to newest",
        "total_commits_scanned": total_scanned,
        "merge_commits_excluded": skipped_merges,
        "examples_generated": len(records),
        "clean_count": category_counts["clean"],
        "noisy_count": category_counts["noisy"],
        "new_file_count": category_counts["new_file"],
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "average_target_count": round(statistics.mean(targets), 3) if targets else 0,
        "median_target_count": statistics.median(targets) if targets else 0,
        "changed_file_count_distribution": dict(sorted(changed_distribution.items(), key=lambda item: int(item[0]))),
        "common_source_extensions": extensions.most_common(20),
        "common_exclusion_reasons": exclusions.most_common(20),
        "noisy_reason_distribution": noisy_reason_distribution(records),
        "potential_query_duplicates": find_query_duplicates(records),
    }


def write_outputs(records, summary, output_dir):
    jsonl_write(output_dir / "benchmark_all.jsonl", records)
    jsonl_write(output_dir / "benchmark_clean.jsonl", [r for r in records if r["category"] == "clean"])
    jsonl_write(output_dir / "benchmark_noisy.jsonl", [r for r in records if r["category"] == "noisy"])
    jsonl_write(output_dir / "benchmark_new_file.jsonl", [r for r in records if r["category"] == "new_file"])
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["commit", "parent", "timestamp", "author", "query", "targets", "target_count", "added_files", "modified_files", "deleted_files", "renamed_files", "all_changed_files", "changed_file_count", "category", "split", "clean_reason", "message_word_count", "evaluation_type"]
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record[key], ensure_ascii=False) if isinstance(record[key], list) else record[key] for key in columns})
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_examples_for_display(records):
    clean = [r for r in records if r["category"] == "clean"]
    if len(clean) <= 20:
        return clean
    if len(clean) <= 10:
        return clean
    return [clean[round(i * (len(clean) - 1) / 9)] for i in range(10)]


def noisy_examples_for_display(records, limit=10):
    noisy = [r for r in records if r["category"] == "noisy"]
    grouped = {}
    for record in noisy:
        grouped.setdefault(record["clean_reason"], []).append(record)
    reasons = sorted(grouped, key=lambda reason: (-len(grouped[reason]), reason))
    selected = []
    depth = 0
    while len(selected) < min(limit, len(noisy)):
        changed = False
        for reason in reasons:
            if depth < len(grouped[reason]):
                selected.append(grouped[reason][depth])
                changed = True
                if len(selected) == min(limit, len(noisy)):
                    break
        if not changed:
            break
        depth += 1
    return selected


def print_examples(records):
    clean = clean_examples_for_display(records)
    print(f"[CLEAN EXAMPLES] showing {len(clean)} of {sum(r['category'] == 'clean' for r in records)}", flush=True)
    for record in clean:
        print(json.dumps({key: record[key] for key in ("query", "commit", "parent", "targets", "split")}, ensure_ascii=False), flush=True)
    noisy = noisy_examples_for_display(records)
    print(f"[NOISY EXAMPLES] showing {len(noisy)}", flush=True)
    for record in noisy:
        print(json.dumps({key: record[key] for key in ("query", "commit", "clean_reason")}, ensure_ascii=False), flush=True)


def comparison_stats(summary):
    keys = (
        ("Commits scanned", "total_commits_scanned"),
        ("Examples generated", "examples_generated"),
        ("Clean", "clean_count"), ("Noisy", "noisy_count"), ("New-file", "new_file_count"),
        ("Train", "train_count"), ("Validation", "validation_count"), ("Test", "test_count"),
        ("Average target count", "average_target_count"), ("Median target count", "median_target_count"),
    )
    return [{"metric": label, "first_parent": summary.get(key, 0), "all_non_merge": 0} for label, key in keys]


def print_comparison(first_parent, all_nonmerge):
    labels = (
        ("Commits scanned", "total_commits_scanned"), ("Examples generated", "examples_generated"),
        ("Clean", "clean_count"), ("Noisy", "noisy_count"), ("New-file", "new_file_count"),
        ("Train", "train_count"), ("Validation", "validation_count"), ("Test", "test_count"),
        ("Average target count", "average_target_count"), ("Median target count", "median_target_count"),
    )
    print(f"{'Metric':<25}{'First-parent':>16}{'All non-merge HEAD':>22}", flush=True)
    for label, key in labels:
        print(f"{label:<25}{str(first_parent.get(key, 0)):>16}{str(all_nonmerge.get(key, 0)):>22}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--history-mode", choices=("first-parent", "all-non-merge"), default="first-parent",
                        help="History traversal; default is first-parent. all-non-merge walks commits reachable from HEAD only.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to generated/ for first-parent, generated_all_nonmerge/ for all-non-merge.")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    output_dir = args.output_dir or (OUTPUT_DIR if args.history_mode == "first-parent" else ALL_NONMERGE_OUTPUT_DIR)
    commits, records, skipped_merges = collect_history(repo, args.history_mode)
    if args.history_mode == "all-non-merge":
        # The selected revision walk excludes merge commits; count reachable merges for the report.
        reachable_merges = git(repo, ["rev-list", "--merges", "HEAD"]).splitlines()
        skipped_merges = len(set(reachable_merges))
    summary = build_summary(len(commits), records, skipped_merges, repo, args.history_mode)
    write_outputs(records, summary, output_dir.resolve())
    if args.history_mode == "all-non-merge":
        deduplicated = deduplicate_clean_records(records)
        dedup_stats = dedup_summary(deduplicated)
        jsonl_write(output_dir.resolve() / "benchmark_clean_dedup.jsonl", deduplicated)
        summary["deduplicated_clean"] = dedup_stats
        (output_dir.resolve() / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_deduplicated_dataset(deduplicated, dedup_stats)
        first_path = OUTPUT_DIR / "benchmark_summary.json"
        first_summary = json.loads(first_path.read_text(encoding="utf-8"))
        print_comparison(first_summary, summary)
        comparison = comparison_stats(first_summary)
        for row in comparison:
            row["all_non_merge"] = summary.get({
                "Commits scanned": "total_commits_scanned", "Examples generated": "examples_generated",
                "Clean": "clean_count", "Noisy": "noisy_count", "New-file": "new_file_count",
                "Train": "train_count", "Validation": "validation_count", "Test": "test_count",
                "Average target count": "average_target_count", "Median target count": "median_target_count",
            }[row["metric"]], 0)
        summary["first_parent_comparison"] = comparison
        (output_dir.resolve() / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[NOISY REASONS] reason | count | percentage", flush=True)
        for row in summary["noisy_reason_distribution"]:
            print(f"{row['reason']} | {row['count']} | {row['percentage']:.2f}%", flush=True)
        print_examples(records)
        print(f"[QUERY REVIEW] {len(summary['potential_query_duplicates'])} possible duplicate/iterative query groups", flush=True)
        for group in summary["potential_query_duplicates"][:20]:
            print(json.dumps(group, ensure_ascii=False), flush=True)
    print(f"[DONE] clean={summary['clean_count']} noisy={summary['noisy_count']} new_file={summary['new_file_count']}", flush=True)
    print(f"[SPLITS] train={summary['train_count']} validation={summary['validation_count']} test={summary['test_count']}", flush=True)
    print(f"[OUTPUT] {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
