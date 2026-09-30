import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from collections import Counter
from unittest.mock import patch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "benchmarks"))

import retrieve
import utils
import update_index as update_index_module
from evaluate_git_benchmark import deduplicate_ranked_files, score_ranked_files, verify_parent_worktree
from run_historical_pilot import (ABLATION_MODES, MODES, SharedEmbeddingCache, evaluate_records,
    ensure_historical_index, ranking_ablation_analysis, sample_pilot_records, summarize_results)


def record(repo, i, targets=None):
    return {"repository": repo, "query": f"improve unique behavior feature {i}", "commit": f"commit-{repo}-{i}",
            "parent": f"parent-{repo}-{i}", "timestamp": "2024-01-01T00:00:00+00:00",
            "targets": targets or [f"src/area{i}/module{i}.js"]}


class HistoricalPilotTests(unittest.TestCase):
    def test_stratified_sampling_is_deterministic_and_balances_diversity(self):
        rows = []
        counts = {"CourseCompass": 3, "Pinia": 20, "Express": 20, "Flask": 20}
        for repo, count in counts.items():
            for i in range(count if repo == "CourseCompass" else 30):
                targets = [f"src/area{i % 8}/file{i}.js"]
                if i % 3 == 0:
                    targets.append(f"test/area{i % 6}/test{i}.js")
                rows.append(record(repo, i, targets))
        a = sample_pilot_records(rows, counts)
        b = sample_pilot_records(rows, counts)
        self.assertEqual([r["commit"] for r in a], [r["commit"] for r in b])
        self.assertEqual(counts, {repo: sum(r["repository"] == repo for r in a) for repo in counts})
        self.assertEqual([r["commit"] for r in a if r["repository"] == "CourseCompass"],
                         [r["commit"] for r in rows if r["repository"] == "CourseCompass"])
        pinia_targets = {t for r in a if r["repository"] == "Pinia" for t in r["targets"]}
        self.assertGreaterEqual(len(pinia_targets), 20)

    def test_parent_state_selection_verifies_expected_head(self):
        result = type("Completed", (), {"stdout": "parent-sha\n"})()
        with patch("evaluate_git_benchmark.subprocess.run", return_value=result) as run:
            verify_parent_worktree(Path("fake-worktree"), "parent-sha")
            self.assertIn("rev-parse", run.call_args.args[0])
        with patch("evaluate_git_benchmark.subprocess.run", return_value=type("Completed", (), {"stdout": "wrong\n"})()):
            with self.assertRaisesRegex(ValueError, "expected parent"):
                verify_parent_worktree(Path("fake-worktree"), "parent-sha")

    def test_historical_git_recency_uses_supplied_reference_time(self):
        retrieve._LATEST_CHANGE_STAMPS.clear()
        with patch("retrieve.git", return_value="100"):
            score = retrieve.latest_change("repo", "src/a.js", reference_time=100)
        self.assertAlmostEqual(score, 0.25)

    def test_modes_use_requested_signals_without_mutating_config(self):
        config_before = copy.deepcopy(retrieve.CONFIG)
        chunk = {"file": "src/courseService.js", "content": "implementation " * 30, "symbol": "courseService",
                 "language": "js", "start_line": 1, "end_line": 2}
        vectors = __import__("numpy").array([[1.0, 0.0]], dtype="float32")
        with patch("retrieve.repo_root", side_effect=lambda root: Path(root)), \
             patch("retrieve.load_index", return_value=([chunk], {"files": {"src/courseService.js": "x"}}, vectors)), \
             patch("retrieve.embed", return_value=vectors[0:1]), \
             patch("retrieve.git", return_value=""), \
             patch("retrieve.dependencies", return_value=set()), \
             patch("retrieve.latest_change", return_value=0.25) as recent:
            semantic = retrieve.retrieve("course service implementation", root="repo", ranking_mode="semantic-only")
            self.assertEqual(semantic["selected"][0]["score"], 1.0)
            self.assertEqual(semantic["selected"][0]["modifier_total"], 0.0)
            no_git = retrieve.retrieve("course service implementation", root="repo", ranking_mode="no-git-recency")
            self.assertEqual(recent.call_count, 0)
            full = retrieve.retrieve("course service implementation", root="repo", ranking_mode="full", reference_time=100)
            self.assertEqual(recent.call_count, 1)
            self.assertEqual(recent.call_args.args[2], 100)
            self.assertGreater(full["selected"][0]["score"], no_git["selected"][0]["score"])
        self.assertEqual(retrieve.CONFIG, config_before)

    def test_file_level_deduplication_and_metrics(self):
        chunks = [{"chunk": {"file": "A.js"}}, {"chunk": {"file": "A.js"}}, {"chunk": {"file": "B.js"}}]
        ranked = deduplicate_ranked_files(chunks)
        self.assertEqual(ranked, ["A.js", "B.js"])
        scores = score_ranked_files(ranked, ["B.js", "C.js"])
        self.assertEqual(scores["first_relevant_rank"], 2)
        self.assertEqual(scores["reciprocal_rank"], 0.5)
        self.assertEqual(scores["recall@5"], 0.5)

    def test_micro_and_macro_repository_metrics(self):
        rows = []
        for repo, hit in (("A", 1), ("A", 1), ("B", 0)):
            rows.append({"repository": repo, "mode": "Semantic-only", "hit@1": hit, "hit@3": hit, "hit@5": hit,
                         "reciprocal_rank": float(hit), "recall@5": float(hit), "context_tokens": 100})
        for repo in ("A", "B"):
            rows.append({"repository": repo, "mode": "No Git Recency", "hit@1": 0, "hit@3": 0, "hit@5": 0,
                         "reciprocal_rank": 0.0, "recall@5": 0.0, "context_tokens": 200})
        summary = summarize_results(rows)
        self.assertAlmostEqual(summary["micro"]["Semantic-only"]["hit@1"], 2 / 3)
        self.assertAlmostEqual(summary["macro_repository_average"]["Semantic-only"]["hit@1"], 0.5)

    def test_one_index_and_query_embedding_are_reused_across_modes(self):
        row = record("Pinia", 1)
        qvec = object()
        result = {"selected": [{"chunk": {"file": "src/area1/module1.js"}, "score": 1.0}], "estimated_tokens": 50}
        with patch("run_historical_pilot.prepare_parent_worktree", return_value=Path("worktree")) as prepare, \
             patch("run_historical_pilot.ensure_historical_index", return_value={"cache_hit": True}) as index, \
             patch("run_historical_pilot.query_vector", return_value=qvec) as query_embedding, \
             patch("run_historical_pilot.retrieve_module.retrieve", return_value=result) as retrieve_call:
            data = evaluate_records([row], stage="test")
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(index.call_count, 1)
        self.assertEqual(query_embedding.call_count, 1)
        self.assertEqual(retrieve_call.call_count, len(MODES))
        self.assertEqual({call.kwargs["query_embedding"] for call in retrieve_call.call_args_list}, {qvec})
        self.assertEqual(data["query_count"], 1)

    def test_ablation_runs_all_profiles_against_one_index_and_query_embedding(self):
        row = record("Pinia", 3)
        qvec = object()
        result = {"selected": [{"chunk": {"file": "src/area3/module3.js"}, "score": 1.0}], "estimated_tokens": 50}
        with patch("run_historical_pilot.prepare_parent_worktree", return_value=Path("worktree")) as prepare, \
             patch("run_historical_pilot.ensure_historical_index", return_value={"cache_hit": True}) as index, \
             patch("run_historical_pilot.query_vector", return_value=qvec) as query_embedding, \
             patch("run_historical_pilot.retrieve_module.retrieve", return_value=result) as retrieve_call:
            data = evaluate_records([row], stage="ablation")
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(index.call_count, 1)
        self.assertEqual(query_embedding.call_count, 1)
        self.assertEqual(retrieve_call.call_count, 10)
        self.assertEqual({call.kwargs["query_embedding"] for call in retrieve_call.call_args_list}, {qvec})
        self.assertTrue(all(call.kwargs["ranking_mode"] == "custom" for call in retrieve_call.call_args_list))
        self.assertEqual(data["summary"]["modes"], [label for label, _ in ABLATION_MODES])

    def test_ablation_profiles_isolate_signals_without_mutating_config(self):
        before = copy.deepcopy(utils.CONFIG)
        profiles = dict(ABLATION_MODES)
        self.assertEqual(set(profiles), {"Semantic Only", "Semantic + Name/Path", "Semantic + Dependency",
            "Semantic + V2.1 Reranking", "Semantic + Git Recency", "Full minus Name/Path",
            "Full minus Dependency", "Full minus V2.1 Reranking", "Full minus Git Recency",
            "Current Full Smart Context"})
        self.assertEqual(profiles["Semantic Only"]["weights"]["semantic"], 1.0)
        for mode, signal in (("Semantic + Name/Path", "name_match"),
                             ("Semantic + Dependency", "dependency"),
                             ("Semantic + Git Recency", "git_recency")):
            self.assertGreater(profiles[mode]["weights"][signal], 0)
            self.assertEqual(sum(value > 0 for value in profiles[mode]["weights"].values()), 2)
            self.assertFalse(profiles[mode]["v21_enabled"])
        self.assertTrue(profiles["Semantic + V2.1 Reranking"]["v21_enabled"])
        self.assertEqual(utils.CONFIG, before)

    def test_ablation_analysis_reports_first_rank_and_top5_transitions(self):
        def row(mode, rank, files):
            target = "src/target.js"
            return {"repository": "Express", "query": "implement target", "commit": "c1", "parent": "p1",
                "targets": [target], "mode": mode, "ranked_unique_files": files,
                "first_relevant_rank": rank, "hit@1": rank == 1, "hit@3": rank is not None and rank <= 3,
                "hit@5": rank is not None and rank <= 5, "reciprocal_rank": 1 / rank if rank else 0,
                "recall@5": 1.0 if rank is not None and rank <= 5 else 0.0,
                "context_tokens": 80, "ranking_scores": []}
        baseline = row("Semantic Only", 6, ["a", "b", "c", "d", "e", "src/target.js"])
        improved = row("Semantic + Name/Path", 2, ["a", "src/target.js"])
        analysis = ranking_ablation_analysis([baseline, improved])
        counts = analysis["query_signal_changes"]["Semantic + Name/Path"]
        self.assertEqual(len(counts["improved_first_relevant_rank"]), 1)
        self.assertEqual(len(counts["relevant_file_entered_top5"]), 1)
        self.assertEqual(analysis["express_pinia_query_signal_changes"]["Express"]["Semantic + Name/Path"][
            "relevant_file_entered_top5"], counts["relevant_file_entered_top5"])

    def test_semantic_only_does_not_change_production_config(self):
        before = copy.deepcopy(utils.CONFIG)
        self.assertEqual(utils.CONFIG, before)

    def test_parent_indexes_reuse_only_identical_chunk_vectors(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            shared_root = Path(temp) / "shared_embeddings" / "pinia"
            api_batches = []
            def fake_embed(texts):
                api_batches.append(list(texts))
                return np.asarray([[float(sum(text.encode("utf-8")) % 100), 1.0] for text in texts], dtype=np.float32)
            parent_a = SharedEmbeddingCache(shared_root, "https://example.test/pinia.git", model="test-model",
                chunker_version="1", embed_fn=fake_embed)
            vectors_a = parent_a.embed_chunks(["A", "B", "C"])
            self.assertEqual(parent_a.stats["embedding_cache_hits"], 0)
            self.assertEqual(parent_a.stats["embedding_cache_misses"], 3)
            parent_a_index = Path(temp) / "indexes" / "parent-a"
            parent_a_index.mkdir(parents=True)
            chunks_a = [{"file": f"src/{name}.js", "content": name} for name in ("A", "B", "C")]
            (parent_a_index / "chunks.json").write_text(json.dumps(chunks_a), encoding="utf-8")
            np.save(parent_a_index / "embeddings.npy", vectors_a)
            (parent_a_index / "metadata.json").write_text(json.dumps({"parent": "parent-a", "files": {"A": "hash-a", "B": "hash-b", "C": "hash-c"}}), encoding="utf-8")
            before_a = (parent_a_index / "chunks.json").read_bytes()

            parent_b = SharedEmbeddingCache(shared_root, "https://example.test/pinia.git", model="test-model",
                chunker_version="1", embed_fn=fake_embed)
            vectors_b = parent_b.embed_chunks(["A", "B", "D"])
            self.assertEqual(parent_b.stats["embedding_cache_hits"], 2)
            self.assertEqual(parent_b.stats["embedding_cache_misses"], 1)
            self.assertEqual(parent_b.stats["embeddings_requested"], 1)
            self.assertEqual(len(api_batches), 2)
            parent_b_index = Path(temp) / "indexes" / "parent-b"
            parent_b_index.mkdir(parents=True)
            chunks_b = [{"file": f"src/{name}.js", "content": name} for name in ("A", "B", "D")]
            (parent_b_index / "chunks.json").write_text(json.dumps(chunks_b), encoding="utf-8")
            np.save(parent_b_index / "embeddings.npy", vectors_b)
            (parent_b_index / "metadata.json").write_text(json.dumps({"parent": "parent-b", "files": {"A": "hash-a", "B": "hash-b", "D": "hash-d"}}), encoding="utf-8")

            stored_b = json.loads((parent_b_index / "chunks.json").read_text(encoding="utf-8"))
            metadata_a = json.loads((parent_a_index / "metadata.json").read_text(encoding="utf-8"))
            metadata_b = json.loads((parent_b_index / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual([chunk["content"] for chunk in stored_b], ["A", "B", "D"])
            self.assertNotIn("C", [chunk["content"] for chunk in stored_b])
            self.assertEqual((parent_a_index / "chunks.json").read_bytes(), before_a)
            self.assertNotIn("D", metadata_a["files"])
            self.assertNotIn("C", metadata_b["files"])
            self.assertEqual(metadata_a["parent"], "parent-a")
            self.assertEqual(metadata_b["parent"], "parent-b")
            np.testing.assert_array_equal(vectors_a[:2], vectors_b[:2])

    def test_shared_embedding_cache_key_invalidates_configuration_changes(self):
        import numpy as np
        variants = (
            {"model": "other-model", "chunker_version": "1", "dimension": 2, "schema_version": 1},
            {"model": "test-model", "chunker_version": "1", "dimension": 3, "schema_version": 1},
            {"model": "test-model", "chunker_version": "2", "dimension": 2, "schema_version": 1},
            {"model": "test-model", "chunker_version": "1", "dimension": 2, "schema_version": 2},
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "pinia"
            def base_embed(texts):
                return np.ones((len(texts), 2), dtype=np.float32)
            base = SharedEmbeddingCache(root, "pinia-id", model="test-model", chunker_version="1",
                dimension=2, schema_version=1, embed_fn=base_embed)
            base.embed_chunks(["same exact chunk"])
            # A newly constructed cache instance must reuse the same exact configuration.
            base_again = SharedEmbeddingCache(root, "pinia-id", model="test-model", chunker_version="1",
                dimension=2, schema_version=1, embed_fn=base_embed)
            base_again.embed_chunks(["same exact chunk"])
            self.assertEqual(base_again.stats["embedding_cache_hits"], 1)
            for change in variants:
                calls = []
                def changed_embed(texts, dimension=change["dimension"]):
                    calls.extend(texts)
                    return np.ones((len(texts), dimension), dtype=np.float32)
                changed = SharedEmbeddingCache(root, "pinia-id", model=change["model"],
                    chunker_version=change["chunker_version"], dimension=change["dimension"],
                    schema_version=change["schema_version"], embed_fn=changed_embed)
                changed.embed_chunks(["same exact chunk"])
                self.assertEqual(changed.stats["embedding_cache_hits"], 0)
                self.assertEqual(changed.stats["embedding_cache_misses"], 1)
                self.assertEqual(calls, ["same exact chunk"])

    def test_intra_parent_duplicate_is_not_reported_as_cross_parent_cache_hit(self):
        import numpy as np
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            cache = SharedEmbeddingCache(Path(temp), "pinia-id", model="test-model", chunker_version="1",
                dimension=2, embed_fn=lambda texts: (calls.extend(texts) or np.ones((len(texts), 2), dtype=np.float32)))
            cache.embed_chunks(["same", "same"])
        self.assertEqual(cache.stats["embedding_cache_hits"], 0)
        self.assertEqual(cache.stats["embedding_cache_misses"], 2)
        self.assertEqual(cache.stats["intra_index_dedup_hits"], 1)
        self.assertEqual(cache.stats["embeddings_requested"], 1)
        self.assertEqual(calls, ["same"])

    def test_historical_index_builder_keeps_parent_chunks_and_metadata_isolated(self):
        import numpy as np
        from run_historical_pilot import shared_embedding_cache_dir
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            indexes = root / "indexes"
            shared = root / "shared" / "pinia"
            content_by_parent = {"A": ["A", "B", "C"], "B": ["A", "B", "D"]}
            def fake_embed(texts):
                return np.asarray([[float(ord(text[0])), 1.0] for text in texts], dtype=np.float32)
            def fake_update(root):
                worktree = root
                contents = content_by_parent[Path(worktree).name]
                target = utils.cache_dir(worktree)
                chunks = [{"file": f"src/{value}.js", "content": value} for value in contents]
                embeddings = update_index_module.embed([chunk["content"] for chunk in chunks])
                target.mkdir(parents=True, exist_ok=True)
                (target / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
                np.save(target / "embeddings.npy", embeddings)
                (target / "metadata.json").write_text(json.dumps({"root": str(worktree),
                    "embedding_model": utils.CONFIG["embedding_model"], "embedding_provider": utils.CONFIG["embedding_provider"],
                    "embedding_dimension": int(embeddings.shape[1]), "files": {f"{x}.js": x for x in contents},
                    "chunk_count": len(chunks), "chunker_version": "1"}), encoding="utf-8")
            with patch("run_historical_pilot._state_cache_dir", side_effect=lambda _repo, parent: indexes / parent), \
                 patch("run_historical_pilot.shared_embedding_cache_dir", return_value=shared), \
                 patch("run_historical_pilot._repository_identity", return_value="pinia-test-id"), \
                 patch("run_historical_pilot._index_is_usable", wraps=__import__("run_historical_pilot")._index_is_usable), \
                 patch("run_historical_pilot.update_index_module.update", side_effect=fake_update), \
                 patch("run_historical_pilot.utils.embed", side_effect=fake_embed):
                first = ensure_historical_index("Pinia", "parent-a", root / "A", Counter())
                parent_a_chunks_path = indexes / "parent-a" / "chunks.json"
                parent_a_before = parent_a_chunks_path.read_bytes()
                second = ensure_historical_index("Pinia", "parent-b", root / "B", Counter())
            chunks_b = json.loads((indexes / "parent-b" / "chunks.json").read_text(encoding="utf-8"))
            metadata_a = json.loads((indexes / "parent-a" / "metadata.json").read_text(encoding="utf-8"))
            metadata_b = json.loads((indexes / "parent-b" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual((first["embedding_cache_hits"], first["embedding_cache_misses"]), (0, 3))
            self.assertEqual((second["embedding_cache_hits"], second["embedding_cache_misses"]), (2, 1))
            self.assertEqual([x["content"] for x in chunks_b], ["A", "B", "D"])
            self.assertNotIn("C", [x["content"] for x in chunks_b])
            self.assertEqual(parent_a_chunks_path.read_bytes(), parent_a_before)
            self.assertEqual(metadata_a["files"], {"A.js": "A", "B.js": "B", "C.js": "C"})
            self.assertEqual(metadata_b["files"], {"A.js": "A", "B.js": "B", "D.js": "D"})
            self.assertNotEqual(metadata_a["root"], metadata_b["root"])


if __name__ == "__main__":
    unittest.main()
