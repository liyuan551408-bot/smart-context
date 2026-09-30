"""Calibrate historical parent-index cost and cross-parent embedding reuse."""
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
SKILL_DIR = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(SKILL_DIR / "scripts"))

import run_historical_pilot as pilot  # noqa: E402
import utils  # noqa: E402
from run_historical_pilot import record_chronology_key  # noqa: E402


DATASET = BENCHMARK_DIR / "generated_multi_repo" / "pilot_test_63.jsonl"
CALIBRATION_ROOT = BENCHMARK_DIR / "cache" / "historical_pilot" / "calibration"
CALIBRATION_REPOSITORIES = ("Pinia", "Express", "Flask")


def read_jsonl(path=DATASET):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_calibration_records(records, repositories=CALIBRATION_REPOSITORIES):
    """Choose early, middle, and late distinct parent states deterministically."""
    selected = {}
    for repository in repositories:
        chronological = sorted(
            (row for row in records if row.get("repository") == repository and row.get("parent")),
            key=record_chronology_key,
        )
        unique = []
        seen = set()
        for row in chronological:
            if row["parent"] not in seen:
                unique.append(row)
                seen.add(row["parent"])
        if len(unique) < 3:
            raise ValueError(f"{repository} has only {len(unique)} distinct parent states; 3 are required")
        positions = (0, (len(unique) - 1) // 2, len(unique) - 1)
        selected[repository] = [unique[position] for position in positions]
    return selected


def _repo_metrics(repository_rows):
    chronological = sorted(repository_rows, key=lambda row: row["calibration_order"])
    first, warm = chronological[0], chronological[1:]
    return {
        "cold_first_parent_seconds": first["indexing_seconds"],
        "average_warm_parent_seconds": statistics.mean(row["indexing_seconds"] for row in warm),
        "median_warm_parent_seconds": statistics.median(row["indexing_seconds"] for row in warm),
        "average_cross_parent_reuse_percentage": statistics.mean(row["cross_parent_reuse_percentage"] for row in repository_rows),
        "total_new_embeddings": sum(row["embeddings_requested"] for row in repository_rows),
        "total_reused_embeddings": sum(row["cross_parent_cache_hits"] for row in repository_rows),
    }


def estimate_pilot_runtime(calibration_rows, unique_parent_counts):
    """Project fresh-cache times from each repo's cold sample and two warm samples."""
    by_repo = defaultdict(list)
    for row in calibration_rows:
        by_repo[row["repository"]].append(row)
    estimates = {}
    for repository in CALIBRATION_REPOSITORIES:
        rows = sorted(by_repo[repository], key=lambda row: row["calibration_order"])
        count = int(unique_parent_counts.get(repository, 0))
        cold = rows[0]["indexing_seconds"]
        warm_times = [row["indexing_seconds"] for row in rows[1:]]
        estimates[repository] = {
            "unique_parent_states": count,
            "optimistic_seconds": cold + max(0, count - 1) * min(warm_times),
            "observed_seconds": cold + max(0, count - 1) * statistics.mean(warm_times),
            "conservative_seconds": cold + max(0, count - 1) * max(warm_times),
            "method": "measured cold first parent + min/mean/max of two warm-parent timings",
        }

    # CourseCompass is intentionally not indexed during calibration. Use its known
    # smoke-parent chunk count when present, and measured per-chunk rates elsewhere.
    course_count = int(unique_parent_counts.get("CourseCompass", 0))
    if calibration_rows:
        per_chunk = [row["indexing_seconds"] / max(1, row["total_chunks"]) for row in calibration_rows]
        normal = statistics.median(per_chunk)
        low, high = min(per_chunk), max(per_chunk)
    else:
        low = normal = high = 0.0
    known_course_chunks = 253
    estimates["CourseCompass"] = {
        "unique_parent_states": course_count,
        "estimated_chunks_per_parent": known_course_chunks,
        "optimistic_seconds": course_count * known_course_chunks * low,
        "observed_seconds": course_count * known_course_chunks * normal,
        "conservative_seconds": course_count * known_course_chunks * high,
        "method": "253-chunk prior smoke index multiplied by min/median/max calibrated seconds per chunk",
    }
    estimates["combined"] = {
        key: sum(item[key] for name, item in estimates.items() if name != "combined")
        for key in ("optimistic_seconds", "observed_seconds", "conservative_seconds")
    }
    return estimates


def _counter_wrapper(counters):
    original = utils.requests.post
    def counted(url, *args, **kwargs):
        if str(url).rstrip("/").endswith("/embeddings"):
            counters["embedding_api_calls"] += 1
            counters["embedding_inputs_sent"] += len(kwargs.get("json", {}).get("input", []))
        response = original(url, *args, **kwargs)
        if str(url).rstrip("/").endswith("/embeddings") and response.ok:
            try:
                counters["embeddings_generated"] += len(response.json().get("data", []))
            except (ValueError, AttributeError):
                pass
        return response
    return original, counted


def run_calibration(records=None):
    records = records if records is not None else read_jsonl()
    selected = select_calibration_records(records)
    if CALIBRATION_ROOT.exists() and any(CALIBRATION_ROOT.iterdir()):
        raise RuntimeError(f"Calibration namespace is not fresh: {CALIBRATION_ROOT}")

    print("[SELECTED PARENTS] No indexes/worktrees will be created until this list is printed.", flush=True)
    for repository in CALIBRATION_REPOSITORIES:
        for order, record in enumerate(selected[repository], 1):
            print(f"{repository} {order}/3 commit={record['commit']} parent={record['parent']} timestamp={record['timestamp']}", flush=True)

    previous_workspace = pilot.WORKSPACE
    pilot.WORKSPACE = CALIBRATION_ROOT
    counters = Counter({"embedding_api_calls": 0, "embedding_inputs_sent": 0, "embeddings_generated": 0,
                        "index_cache_hits": 0, "index_cache_misses": 0, "indexing_seconds": 0.0})
    original_post, counted_post = _counter_wrapper(counters)
    utils.requests.post = counted_post
    rows = []
    wall_started = time.perf_counter()
    try:
        CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
        for repository in CALIBRATION_REPOSITORIES:
            # The selected records are already chronological; preserve warm-up order.
            for order, record in enumerate(selected[repository], 1):
                print(f"[INDEX] {repository} {order}/3 {record['parent'][:12]}", flush=True)
                worktree = pilot.prepare_parent_worktree(repository, record["parent"])
                state = pilot.ensure_historical_index(repository, record["parent"], worktree, counters)
                rows.append({
                    "repository": repository,
                    "commit": record["commit"],
                    "parent": record["parent"],
                    "timestamp": record["timestamp"],
                    "calibration_order": order,
                    "indexed_files": state["indexed_files"],
                    "total_chunks": state["total_chunks"],
                    "cross_parent_cache_hits": state["embedding_cache_hits"],
                    "cache_misses": state["embedding_cache_misses"],
                    "same_index_duplicate_reuse": state.get("intra_index_dedup_hits", 0),
                    "embeddings_requested": state["embeddings_requested"],
                    "embedding_api_calls": state["embedding_api_calls"],
                    "cross_parent_reuse_percentage": state["embedding_reuse_percentage"],
                    "indexing_seconds": state["indexing_seconds"],
                })
    finally:
        utils.requests.post = original_post
        pilot.WORKSPACE = previous_workspace

    unique_counts = {repo: len({row["parent"] for row in records if row.get("repository") == repo})
                     for repo in ("CourseCompass", *CALIBRATION_REPOSITORIES)}
    repo_summaries = {repo: _repo_metrics([row for row in rows if row["repository"] == repo])
                      for repo in CALIBRATION_REPOSITORIES}
    estimates = estimate_pilot_runtime(rows, unique_counts)
    shared_entries = sum(1 for repo in CALIBRATION_REPOSITORIES
                         for path in (CALIBRATION_ROOT / "shared_embeddings" / repo.lower()).glob("*.npy"))
    total_chunks = sum(row["total_chunks"] for row in rows)
    total_reused = sum(row["cross_parent_cache_hits"] for row in rows)
    summary = {
        "per_repository": repo_summaries,
        "combined": {
            "total_elapsed_indexing_seconds": sum(row["indexing_seconds"] for row in rows),
            "wall_elapsed_seconds_including_worktrees": time.perf_counter() - wall_started,
            "mean_parent_indexing_seconds": statistics.mean(row["indexing_seconds"] for row in rows),
            "median_parent_indexing_seconds": statistics.median(row["indexing_seconds"] for row in rows),
            "total_embeddings_requested": sum(row["embeddings_requested"] for row in rows),
            "total_reused_embeddings": total_reused,
            "total_chunks": total_chunks,
            "overall_cross_parent_reuse_percentage": 100.0 * total_reused / max(1, total_chunks),
            "embedding_api_calls": counters["embedding_api_calls"],
            "embeddings_generated": counters["embeddings_generated"],
            "shared_embedding_cache_entries": shared_entries,
        },
        "pilot_unique_parent_counts": unique_counts,
        "pilot_runtime_estimates_seconds": estimates,
        "parents": rows,
    }
    output = BENCHMARK_DIR / "generated_multi_repo" / "historical_runtime_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[PARENT RESULTS]", flush=True)
    for row in rows:
        print(f"{row['repository']} parent={row['parent']} files={row['indexed_files']} chunks={row['total_chunks']} cross-parent hits={row['cross_parent_cache_hits']} misses={row['cache_misses']} intra-index duplicates={row['same_index_duplicate_reuse']} requested={row['embeddings_requested']} API calls={row['embedding_api_calls']} reuse={row['cross_parent_reuse_percentage']:.1f}% time={row['indexing_seconds']:.2f}s", flush=True)
    print("\n[REPOSITORY SUMMARY] " + json.dumps(repo_summaries, sort_keys=True), flush=True)
    print("[COMBINED] " + json.dumps(summary["combined"], sort_keys=True), flush=True)
    print("[63-QUERY INDEX RUNTIME ESTIMATE: seconds] " + json.dumps(estimates, sort_keys=True), flush=True)
    print(f"[DONE] saved {output}", flush=True)
    return summary


if __name__ == "__main__":
    try:
        run_calibration()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"runtime calibration: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
