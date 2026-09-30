import sys
import unittest
import copy
import os
import time
from pathlib import Path
from unittest.mock import patch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "benchmarks"))

import numpy as np
import run_historical_pilot as pilot


def row(repository, date, sha):
    return {"repository": repository, "timestamp": date, "commit": sha, "parent": "parent-" + sha,
            "query": "find behavior", "targets": ["src/example.js"]}


def stub_result(stage="pilot"):
    return {"stage": stage, "query_count": 0, "summary": {}, "per_query": [], "index_states": [], "counters": {}}


class QueuedFuture:
    def __init__(self, executor, function, args):
        self.executor, self.function, self.args = executor, function, args

    def result(self):
        self.executor.completed_submissions.append(len(self.executor.submissions))
        return self.function(*self.args)


class QueuedExecutor:
    instances = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submissions = []
        self.completed_submissions = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, function, *args):
        self.submissions.append((function, args))
        return QueuedFuture(self, function, args)


def fake_repository_worker(repository, records, stage):
    return {"stage": stage, "query_count": len(records), "summary": {}, "per_query": [],
            "index_states": [], "counters": {}}


def concurrent_probe_worker(repository, records, stage):
    started = time.perf_counter()
    time.sleep(0.35)
    ended = time.perf_counter()
    return {"stage": stage, "query_count": len(records), "summary": {}, "per_query": [],
            "index_states": [{"repository": repository, "worker_pid": os.getpid(),
                              "started": started, "ended": ended}], "counters": {}}


class ParallelHistoricalPilotTests(unittest.TestCase):
    def setUp(self):
        QueuedExecutor.instances.clear()

    def test_cli_accepts_parallel_repository_flag_and_default_is_unchanged(self):
        parser = pilot.build_arg_parser()
        for requested in ("1", "2", "4"):
            args = parser.parse_args(["--stage", "pilot", "--parallel-repositories", requested])
            self.assertEqual(args.parallel_repositories, int(requested))
        args_default = parser.parse_args(["--stage", "pilot"])
        self.assertIsNone(args_default.parallel_repositories)

    def test_worker_count_is_capped_by_request_repositories_and_cpu(self):
        self.assertEqual(pilot.repository_worker_count(4, 4, cpu_count=2), 2)
        self.assertEqual(pilot.repository_worker_count(4, 3, cpu_count=16), 3)
        self.assertEqual(pilot.repository_worker_count(1, 4, cpu_count=16), 1)
        with self.assertRaises(ValueError):
            pilot.repository_worker_count(0, 4, cpu_count=16)

    def test_same_repository_parent_records_are_sorted_and_given_to_one_sequential_worker(self):
        records = [row("Pinia", "2025-03-01", "late"), row("Pinia", "2024-01-01", "early"),
                   row("Pinia", "2025-01-01", "middle")]
        observed = []
        def worker(repository, group, stage):
            observed.append((repository, [record["commit"] for record in group]))
            return stub_result(stage)
        result = pilot.evaluate_repositories_parallel(records, 4, worker_fn=worker, executor_cls=QueuedExecutor, cpu_count=8)
        self.assertEqual(observed, [("Pinia", ["early", "middle", "late"])])
        self.assertEqual(result["counters"]["parallel_worker_count"], 1)

    def test_different_repositories_are_submitted_to_pool_before_results_are_consumed(self):
        records = [row("Pinia", "2025-01-01", "p"), row("Express", "2025-01-01", "e"),
                   row("Flask", "2025-01-01", "f"), row("CourseCompass", "2025-01-01", "c")]
        pilot.evaluate_repositories_parallel(records, 4, worker_fn=fake_repository_worker,
            executor_cls=QueuedExecutor, cpu_count=3)
        executor = QueuedExecutor.instances[-1]
        self.assertEqual(executor.max_workers, 3)
        self.assertEqual(len(executor.submissions), 4)
        self.assertEqual(executor.completed_submissions, [4, 4, 4, 4])

    def test_different_repositories_execute_concurrently_in_processes(self):
        records = [row("Pinia", "2025-01-01", "p"), row("Express", "2025-01-01", "e")]
        result = pilot.evaluate_repositories_parallel(records, 2, stage="pilot", cpu_count=2,
            worker_fn=concurrent_probe_worker)
        intervals = result["index_states"]
        self.assertEqual(len({item["worker_pid"] for item in intervals}), 2)
        self.assertLess(max(item["started"] for item in intervals), min(item["ended"] for item in intervals))

    def test_parallel_metrics_match_serial_and_workers_do_not_write_outputs(self):
        records = [row("Pinia", "2025-01-01", "p"), row("Express", "2025-01-01", "e")]
        state = {"cache_hit": True, "indexed_files": 1, "chunk_count": 1, "total_chunks": 1,
                 "embedding_count": 1, "embedding_cache_hits": 0, "embedding_cache_misses": 1,
                 "embeddings_requested": 1, "embedding_api_calls": 1}
        selected = [{"chunk": {"file": "src/example.js"}, "score": .9, "base_score": .9,
                     "modifier_total": 0., "tiny_adjustment": 0., "path_role_adjustment": 0.,
                     "source_type_adjustment": 0.}]
        config_before = copy.deepcopy(pilot.utils.CONFIG)
        with patch.object(pilot, "prepare_parent_worktree", side_effect=lambda repo, _parent: Path(repo)), \
             patch.object(pilot, "ensure_historical_index", return_value=state), \
             patch.object(pilot, "query_vector", return_value=np.array([1.0], dtype=np.float32)), \
             patch.object(pilot.retrieve_module, "retrieve", return_value={"selected": selected, "estimated_tokens": 20}), \
             patch.object(pilot, "_persist") as persist:
            serial = pilot.evaluate_records(pilot.group_repository_records(records)["Pinia"] +
                pilot.group_repository_records(records)["Express"], stage="pilot")
            parallel = pilot.evaluate_repositories_parallel(records, 2, stage="pilot",
                executor_cls=QueuedExecutor, cpu_count=2)
            self.assertEqual(serial["summary"], parallel["summary"])
            self.assertEqual([(r["repository"], r["commit"], r["mode"], r["hit@5"]) for r in serial["per_query"]],
                [(r["repository"], r["commit"], r["mode"], r["hit@5"]) for r in parallel["per_query"]])
            self.assertEqual(persist.call_count, 0)
        self.assertEqual(pilot.utils.CONFIG, config_before)

    def test_parent_output_writer_writes_final_files_once(self):
        result = {"per_query": []}
        with patch.object(pilot, "OUTPUT_DIR", Path("unused")), patch.object(pilot, "_persist") as persist:
            pilot.persist_pilot_outputs(result, "pilot")
        paths = [call.args[1].name for call in persist.call_args_list]
        self.assertEqual(paths.count("pilot_results.json"), 1)
        self.assertEqual(paths.count("pilot_per_query.json"), 1)

    def test_parallel_paths_do_not_mutate_production_config(self):
        before = __import__("copy").deepcopy(pilot.utils.CONFIG)
        self.assertEqual(pilot.utils.CONFIG, before)


if __name__ == "__main__":
    unittest.main()
