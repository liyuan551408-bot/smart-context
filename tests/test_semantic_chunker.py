import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import requests
import semantic_chunker as semantic
from chunker import chunk_file
from update_index import effective_chunker_version, index_requires_rechunk


def fixture_source():
    lines = ["const alpha = () => {"]
    lines.extend(f"  const alphaValue{n} = {n}" for n in range(1, 40))
    lines.append("}")
    lines.append("const beta = () => {")
    lines.extend(f"  const betaValue{n} = {n}" for n in range(1, 40))
    lines.append("}")
    return "\n".join(lines) + "\n"


def response_for(plan_text):
    return type("Response", (), {"ok": True, "status_code": 200, "json": lambda self: {"choices": [{"message": {"content": plan_text}}]}})()


class SemanticChunkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config" / "config.json").read_text(encoding="utf-8"))
        self.config["semantic_chunking"]["max_retries"] = 0
        self.rel = "src/mixed.js"
        self.source = fixture_source()
        self.local = chunk_file(self.rel, self.source)
        self.assertEqual(len(self.local), 1)
        self.assertTrue(semantic.evaluate({**self.local[0], "chunk_method": "local_fallback"}, self.source, self.config)["needs_refinement"])
        os.environ["ZHIPU_API_KEY"] = "unit-test-only"

    def tearDown(self):
        os.environ.pop("ZHIPU_API_KEY", None)
        self.temp.cleanup()

    def valid_plan_text(self):
        middle = len(self.source.splitlines()) // 2
        return json.dumps({"chunks": [
            {"start_line": 1, "end_line": middle, "symbol": "alpha", "purpose": "alpha responsibility", "confidence": 0.9},
            {"start_line": middle + 1, "end_line": len(self.source.splitlines()), "symbol": "beta", "purpose": "beta responsibility", "confidence": 0.9},
        ]})

    def test_json_response_is_parsed_and_applied(self):
        with patch.object(semantic.requests, "post", return_value=response_for(self.valid_plan_text())) as post:
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(metrics["glm_api_calls"], 1)
        self.assertTrue(all(c["chunk_method"] == "glm_semantic" for c in chunks))
        self.assertEqual(post.call_count, 1)

    def test_invalid_json_retries_once_then_falls_back(self):
        with patch.object(semantic.requests, "post", return_value=response_for("not JSON")) as post:
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], self.local[0]["content"])
        self.assertEqual(metrics["glm_api_calls"], 2)
        self.assertEqual(post.call_count, 2)

    def test_top_level_list_is_accepted_without_repair(self):
        bare = json.dumps(json.loads(self.valid_plan_text())["chunks"])
        with patch.object(semantic.requests, "post", return_value=response_for(bare)) as post:
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(metrics["glm_api_calls"], 1)
        self.assertEqual(metrics["glm_response_top_level_type"], "list")
        self.assertEqual(metrics["glm_normalized_chunk_count"], 2)
        self.assertEqual(metrics["glm_validation_result"], "accepted")
        self.assertEqual(metrics["fallback_count"], 0)
        self.assertEqual(post.call_count, 1)

    def test_wrong_top_level_type_rejected(self):
        result, reason = semantic._validate_plan("text", self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("top-level", reason)

    def test_dict_without_chunks_rejected(self):
        result, reason = semantic._validate_plan({"result": []}, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("chunks", reason)

    def test_top_level_list_with_invalid_ranges_rejected(self):
        plan = [{"start_line": 1, "end_line": len(self.source.splitlines()) + 1, "confidence": 0.9}]
        result, reason = semantic._validate_plan(plan, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("outside", reason)

    def test_top_level_list_with_overlap_rejected(self):
        middle = len(self.source.splitlines()) // 2
        plan = [
            {"start_line": 1, "end_line": middle, "confidence": 0.9},
            {"start_line": middle, "end_line": len(self.source.splitlines()), "confidence": 0.9},
        ]
        result, reason = semantic._validate_plan(plan, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("overlap", reason)

    def test_top_level_list_with_out_of_bounds_rejected(self):
        plan = [{"start_line": 0, "end_line": len(self.source.splitlines()), "confidence": 0.9}]
        result, reason = semantic._validate_plan(plan, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("outside", reason)

    def test_overlapping_ranges_rejected(self):
        plan = {"chunks": [
            {"start_line": 1, "end_line": 12, "confidence": 0.9},
            {"start_line": 12, "end_line": len(self.source.splitlines()), "confidence": 0.9},
        ]}
        result, reason = semantic._validate_plan(plan, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("gap", reason)

    def test_out_of_bounds_ranges_rejected(self):
        plan = {"chunks": [{"start_line": 0, "end_line": len(self.source.splitlines()), "confidence": 0.9}]}
        result, _ = semantic._validate_plan(plan, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)

    def test_gaps_are_rejected(self):
        plan = {"chunks": [
            {"start_line": 1, "end_line": 10, "confidence": 0.9},
            {"start_line": 12, "end_line": len(self.source.splitlines()), "confidence": 0.9},
        ]}
        result, reason = semantic._validate_plan(plan, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("gap", reason)

    def test_confidence_validation(self):
        plan = {"chunks": [{"start_line": 1, "end_line": len(self.source.splitlines()), "confidence": 0.2}]}
        result, reason = semantic._validate_plan(plan, self.local[0], self.source.splitlines(), self.config["semantic_chunking"])
        self.assertIsNone(result)
        self.assertIn("confidence", reason)

    def test_tiny_comment_merges_into_adjacent_function(self):
        chunks = [
            {"file": "a.js", "start_line": 1, "end_line": 1, "language": "js", "symbol": "", "content": "// note", "content_hash": "a", "id": "a"},
            {"file": "a.js", "start_line": 2, "end_line": 5, "language": "js", "symbol": "work", "content": "function work() {\n return 1\n}", "content_hash": "b", "id": "b"},
        ]
        merged, count = semantic.merge_small_chunks(chunks, self.config)
        self.assertEqual(count, 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["chunk_method"], "merged_small_chunk")

    def test_semantic_plan_cache_hit_and_invalidation(self):
        plan = self.valid_plan_text()
        with patch.object(semantic.requests, "post", return_value=response_for(plan)) as post:
            first, first_metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
            second, second_metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
            changed_source = self.source.replace("const alphaValue1 = 1", "const alphaValue1 = 101")
            semantic.adaptive_chunk_file(self.rel, changed_source, self.root, self.config)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(first_metrics["glm_api_calls"], 1)
        self.assertEqual(second_metrics["glm_plans_reused"], 1)
        self.assertEqual(first, second)

    def test_api_failure_falls_back_to_local(self):
        with patch.object(semantic.requests, "post", side_effect=requests.Timeout):
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], self.local[0]["content"])
        self.assertEqual(metrics["glm_api_calls"], 1)

    def test_disabled_mode_matches_v1_local_chunking(self):
        disabled = json.loads(json.dumps(self.config))
        disabled["semantic_chunking"]["enabled"] = False
        with patch.object(semantic.requests, "post") as post:
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, disabled)
        self.assertEqual(chunks, self.local)
        self.assertEqual(metrics["glm_api_calls"], 0)
        post.assert_not_called()

    def test_missing_key_falls_back_without_request(self):
        os.environ.pop("ZHIPU_API_KEY", None)
        with patch.object(semantic.requests, "post") as post:
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], self.local[0]["content"])
        self.assertEqual(metrics["glm_api_calls"], 0)
        post.assert_not_called()

    def test_blank_line_gap_is_reconciled_to_previous_chunk(self):
        lines = ["function a() {", " return 1", "}", "", "function b() {", " return 2", "}"]
        region = {"start_line": 1, "end_line": 7}
        plan = [{"start_line": 1, "end_line": 3}, {"start_line": 5, "end_line": 7}]
        result, repaired, error = semantic.reconcile_plan_ranges(plan, region, lines)
        self.assertFalse(error)
        self.assertEqual(repaired, [4])
        self.assertEqual([[x["start_line"], x["end_line"]] for x in result], [[1, 4], [5, 7]])

    def test_comment_gap_is_reconciled_to_following_chunk(self):
        lines = ["function a() {", " return 1", "}", "// next responsibility", "function b() {", " return 2", "}"]
        region = {"start_line": 1, "end_line": 7}
        plan = [{"start_line": 1, "end_line": 3}, {"start_line": 5, "end_line": 7}]
        result, repaired, error = semantic.reconcile_plan_ranges(plan, region, lines)
        self.assertFalse(error)
        self.assertEqual(repaired, [4])
        self.assertEqual([[x["start_line"], x["end_line"]] for x in result], [[1, 3], [4, 7]])

    def test_standalone_closing_brace_gap_attaches_to_previous(self):
        lines = ["function a() {", " return 1", "}", "function b() {", " return 2", "}"]
        region = {"start_line": 1, "end_line": 6}
        plan = [{"start_line": 1, "end_line": 2}, {"start_line": 4, "end_line": 6}]
        result, repaired, error = semantic.reconcile_plan_ranges(plan, region, lines)
        self.assertFalse(error)
        self.assertEqual(repaired, [3])
        self.assertEqual([[x["start_line"], x["end_line"]] for x in result], [[1, 3], [4, 6]])

    def test_meaningful_code_gap_is_rejected(self):
        lines = ["function a() {", " return 1", "}", "const beta = () => {", " return 2", "}"]
        region = {"start_line": 1, "end_line": 6}
        plan = [{"start_line": 1, "end_line": 3}, {"start_line": 5, "end_line": 6}]
        result, _, reason = semantic.reconcile_plan_ranges(plan, region, lines)
        self.assertIsNone(result)
        self.assertIn("meaningful source omitted", reason)
        self.assertIn("4", reason)

    def test_meaningful_gap_causes_one_repair_request(self):
        lines = self.source.splitlines()
        middle = len(lines) // 2
        bad = json.dumps({"chunks": [
            {"start_line": 1, "end_line": middle, "confidence": 0.9},
            {"start_line": middle + 2, "end_line": len(lines), "confidence": 0.9},
        ]})
        valid = self.valid_plan_text()
        with patch.object(semantic.requests, "post", side_effect=[response_for(bad), response_for(valid)]) as post:
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(metrics["glm_api_calls"], 2)
        self.assertEqual(metrics["glm_validation_result"], "accepted")
        self.assertEqual(metrics["fallback_count"], 0)
        self.assertEqual(len(chunks), 2)

    def test_invalid_repair_falls_back_to_local(self):
        lines = self.source.splitlines()
        middle = len(lines) // 2
        bad = json.dumps({"chunks": [
            {"start_line": 1, "end_line": middle, "confidence": 0.9},
            {"start_line": middle + 2, "end_line": len(lines), "confidence": 0.9},
        ]})
        with patch.object(semantic.requests, "post", return_value=response_for(bad)) as post:
            chunks, metrics = semantic.adaptive_chunk_file(self.rel, self.source, self.root, self.config)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(metrics["fallback_count"], 1)
        self.assertIn("meaningful source omitted", metrics["glm_rejection_reason"])
        self.assertEqual(chunks[0]["content"], self.local[0]["content"])

    def test_leading_blank_and_comment_gap_attaches_to_first_chunk(self):
        lines = ["", "// module purpose", "function work() {", " return 1", "}"]
        region = {"start_line": 1, "end_line": 5}
        plan = [{"start_line": 3, "end_line": 5}]
        result, repaired, error = semantic.reconcile_plan_ranges(plan, region, lines)
        self.assertFalse(error)
        self.assertEqual(repaired, [1, 2])
        self.assertEqual([[x["start_line"], x["end_line"]] for x in result], [[1, 5]])

    def test_trailing_blank_and_comment_gap_attaches_to_previous(self):
        lines = ["function work() {", " return 1", "}", "// trailing note", ""]
        region = {"start_line": 1, "end_line": 5}
        plan = [{"start_line": 1, "end_line": 3}]
        result, repaired, error = semantic.reconcile_plan_ranges(plan, region, lines)
        self.assertFalse(error)
        self.assertEqual(repaired, [4, 5])
        self.assertEqual([[x["start_line"], x["end_line"]] for x in result], [[1, 5]])

    def test_non_one_absolute_line_numbers_validate(self):
        lines = ["// preamble"] * 119 + ["function work() {"] + ["  return 1"] * 59 + ["}"]
        region = {"start_line": 120, "end_line": 180}
        plan = {"chunks": [
            {"start_line": 120, "end_line": 150, "confidence": 0.9},
            {"start_line": 151, "end_line": 180, "confidence": 0.9},
        ]}
        result, reason = semantic._validate_plan(plan, region, lines, self.config["semantic_chunking"])
        self.assertFalse(reason)
        self.assertEqual(result[0]["start_line"], 120)
        self.assertEqual(result[-1]["end_line"], 180)

    def test_reconciled_ranges_cover_every_line_without_standalone_gap_chunk(self):
        lines = ["function a() {", " return 1", "}", "", "function b() {", " return 2", "}"]
        region = {"start_line": 1, "end_line": 7}
        plan = [{"start_line": 1, "end_line": 3}, {"start_line": 5, "end_line": 7}]
        reconciled, _, error = semantic.reconcile_plan_ranges(plan, region, lines)
        result, validation_error = semantic._validate_plan(reconciled, region, lines, self.config["semantic_chunking"])
        self.assertFalse(error)
        self.assertFalse(validation_error)
        covered = [n for item in result for n in range(item["start_line"], item["end_line"] + 1)]
        self.assertEqual(covered, list(range(1, 8)))
        self.assertEqual(len(result), 2)
        self.assertNotIn([4, 4], [[x["start_line"], x["end_line"]] for x in result])

    def test_v1_v2_mode_versions_and_rechunk_transitions(self):
        v1 = json.loads(json.dumps(self.config))
        v1["semantic_chunking"]["enabled"] = False
        v2 = json.loads(json.dumps(self.config))
        v2["semantic_chunking"]["enabled"] = True
        v1_version = effective_chunker_version(v1)
        v2_version = effective_chunker_version(v2)
        self.assertEqual(v1_version, "1")
        self.assertTrue(v2_version.startswith("2:"))
        self.assertTrue(index_requires_rechunk({"chunker_version": v2_version}, v1))
        self.assertTrue(index_requires_rechunk({"chunker_version": v1_version}, v2))
        self.assertFalse(index_requires_rechunk({"chunker_version": v1_version}, v1))
        self.assertFalse(index_requires_rechunk({"chunker_version": v2_version}, v2))


if __name__ == "__main__":
    unittest.main()
