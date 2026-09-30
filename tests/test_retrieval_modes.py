import copy
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import retrieve


class RetrievalModeTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"file": "src/seed.js", "content": "implementation " * 100,
             "symbol": "seed", "start_line": 1, "end_line": 20},
            {"file": "src/dep.js", "content": "implementation " * 100,
             "symbol": "dependency", "start_line": 1, "end_line": 20},
            {"file": "src/alpha.js", "content": "implementation " * 100,
             "symbol": "alpha", "start_line": 1, "end_line": 20},
            {"file": "src/zeta.js", "content": "implementation " * 100,
             "symbol": "zeta", "start_line": 1, "end_line": 20},
        ]
        self.vectors = np.asarray([[1.0, 0.0], [0.2, 0.98], [0.8, 0.6], [0.8, 0.6]], dtype=np.float32)
        self.metadata = {"files": {chunk["file"]: "hash" for chunk in self.chunks}}

    def retrieval_patches(self, dependency_files=(), candidate_k=1):
        return (
            patch("retrieve.repo_root", side_effect=lambda root: root),
            patch("retrieve.load_index", return_value=(self.chunks, self.metadata, self.vectors)),
            patch("retrieve.embed", return_value=np.asarray([[1.0, 0.0]], dtype=np.float32)),
            patch("retrieve.dependencies", return_value=set(dependency_files)),
            patch("retrieve.git", return_value=""),
            patch("retrieve.CONFIG", {**retrieve.CONFIG, "candidate_k": candidate_k, "top_k": 4,
                                       "semantic_chunking": {"enabled": False}}),
        )

    def test_api_and_cli_default_to_hybrid_and_expose_semantic_alternate(self):
        self.assertEqual(retrieve.retrieve.__kwdefaults__["ranking_mode"], "hybrid")
        for argv, expected_mode in ((["retrieve.py", "--query", "find implementation"], "hybrid"),
                                    (["retrieve.py", "--query", "find implementation", "--mode", "semantic"], "semantic"),
                                    (["retrieve.py", "--query", "find implementation", "--mode", "hybrid"], "hybrid")):
            with self.subTest(argv=argv), patch.object(retrieve, "retrieve", return_value={
                "task": "find implementation", "selected": [], "dependencies": [],
                "estimated_tokens": 3, "max_tokens": 6000}) as call, patch("sys.argv", argv), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(retrieve.main(), 0)
                self.assertEqual(call.call_args.kwargs["ranking_mode"], expected_mode)

    def test_invalid_cli_mode_fails_cleanly(self):
        with patch("sys.argv", ["retrieve.py", "--query", "find code", "--mode", "automatic"]), \
                redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                retrieve.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", error.getvalue())

    def test_semantic_mode_uses_only_semantic_candidates_and_scores(self):
        patches = self.retrieval_patches(dependency_files={"src/dep.js"})
        with patches[0], patches[1], patches[2], patches[3] as dependency, patches[4], patches[5], \
                patch("retrieve.reranking_adjustments", side_effect=AssertionError("V2.1 used")), \
                patch("retrieve.latest_change", side_effect=AssertionError("Git recency used")):
            result = retrieve.retrieve("find implementation", root="repo", ranking_mode="semantic",
                                       query_embedding=np.asarray([1.0, 0.0], dtype=np.float32))
        dependency.assert_not_called()
        self.assertEqual(result["dependencies"], [])
        self.assertNotIn("src/dep.js", {item["chunk"]["file"] for item in result["selected"]})
        self.assertTrue(all(item["score"] == item["semantic_similarity"] for item in result["selected"]))
        self.assertTrue(all(item["modifier_total"] == 0.0 for item in result["selected"]))

    def test_default_hybrid_is_semantic_plus_v21_without_legacy_rank_weights(self):
        config_before = copy.deepcopy(retrieve.CONFIG)
        patches = self.retrieval_patches(dependency_files={"src/dep.js"})
        v21 = {"tiny_adjustment": 0.0, "path_role_adjustment": 0.0,
               "source_type_adjustment": 0.0, "modifier_total": 0.0}
        with patches[0], patches[1], patches[2], patches[3] as dependency, patches[4], patches[5], \
                patch("retrieve.reranking_adjustments", return_value=v21) as rerank, \
                patch("retrieve.latest_change", side_effect=AssertionError("Git recency used")):
            default = retrieve.retrieve("find implementation", root="repo",
                                         query_embedding=np.asarray([1.0, 0.0], dtype=np.float32))
            explicit = retrieve.retrieve("find implementation", root="repo", ranking_mode="hybrid",
                                         query_embedding=np.asarray([1.0, 0.0], dtype=np.float32))
        dependency.assert_called()
        rerank.assert_called()
        self.assertEqual(default, explicit)
        by_file = {item["chunk"]["file"]: item for item in default["selected"]}
        self.assertAlmostEqual(by_file["src/seed.js"]["base_score"], 0.55)
        self.assertAlmostEqual(by_file["src/dep.js"]["base_score"], 0.55 * 0.2)
        self.assertEqual(retrieve.CONFIG, config_before)

    def test_semantic_and_hybrid_ordering_is_deterministic_for_fixed_fixture(self):
        patches = self.retrieval_patches(candidate_k=4)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patch("retrieve.reranking_adjustments", return_value={
                    "tiny_adjustment": 0.0, "path_role_adjustment": 0.0,
                    "source_type_adjustment": 0.0, "modifier_total": 0.0}):
            results = {}
            for mode in ("semantic", "hybrid"):
                first = retrieve.retrieve("neutral query", root="repo", ranking_mode=mode,
                                          query_embedding=np.asarray([1.0, 0.0], dtype=np.float32))
                second = retrieve.retrieve("neutral query", root="repo", ranking_mode=mode,
                                           query_embedding=np.asarray([1.0, 0.0], dtype=np.float32))
                self.assertEqual(first, second)
                results[mode] = [item["chunk"]["file"] for item in first["selected"]]
        self.assertEqual(results["semantic"][:3], ["src/seed.js", "src/alpha.js", "src/zeta.js"])
        self.assertEqual(results["hybrid"][:3], ["src/seed.js", "src/alpha.js", "src/zeta.js"])

    def test_ranking_profile_override_is_local_to_call(self):
        config_before = copy.deepcopy(retrieve.CONFIG)
        profile = {"weights": {"semantic": 0.2, "dependency": 0.0, "git_recency": 0.0,
                               "name_match": 0.0, "working_file": 0.0}, "v21_enabled": False}
        patches = self.retrieval_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patch("retrieve.reranking_adjustments", side_effect=AssertionError("V2.1 used")):
            result = retrieve.retrieve("find implementation", root="repo", ranking_mode="custom",
                                       ranking_profile=profile,
                                       query_embedding=np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(result["selected"][0]["score"], 0.2)
        self.assertEqual(retrieve.CONFIG, config_before)
        self.assertEqual(profile["weights"]["semantic"], 0.2)

    def test_token_budget_is_enforced(self):
        patches = self.retrieval_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
                patch("retrieve.CONFIG", {**retrieve.CONFIG, "top_k": 4,
                                           "semantic_chunking": {"enabled": False}}):
            result = retrieve.retrieve("find code", root="repo", max_tokens=80, ranking_mode="semantic",
                                       query_embedding=np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertEqual(result["max_tokens"], 80)
        self.assertLessEqual(result["estimated_tokens"], 80)
        self.assertEqual(result["selected"], [])


if __name__ == "__main__":
    unittest.main()
