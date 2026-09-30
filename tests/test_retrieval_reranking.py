import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrieve import identifier_terms, reranking_adjustments


class RetrievalRerankingTests(unittest.TestCase):
    def chunk(self, file, content, symbol=""):
        return {"file": file, "content": content, "symbol": symbol}

    def test_tiny_anonymous_chunk_loses_to_similar_complete_implementation(self):
        noisy = self.chunk("src/misc.js", " ".join(["noise"] * 20))
        implementation = self.chunk("src/searchService.js", " ".join(["implementation"] * 100), "searchService")
        with patch("retrieve.token_estimate", side_effect=lambda text: 20 if text == noisy["content"] else 100):
            noisy_adj = reranking_adjustments("search service implementation", noisy)
            impl_adj = reranking_adjustments("search service implementation", implementation)
        noisy_final = 0.52 + noisy_adj["modifier_total"]
        implementation_final = 0.49 + impl_adj["modifier_total"]
        self.assertLess(noisy_final, implementation_final)
        self.assertEqual(noisy_adj["tiny_adjustment"], -0.05)

    def test_exact_meaningful_short_symbol_is_protected(self):
        short = self.chunk("src/searchService.js", "function search() {}", "searchService")
        self.assertEqual(identifier_terms("searchService"), {"search", "service"})
        adj = reranking_adjustments("searchService implementation", short, name_score=1.0)
        self.assertEqual(adj["tiny_adjustment"], 0.0)
        self.assertGreaterEqual(adj["modifier_total"], 0.0)

    def test_generic_camel_case_filename_role_boost(self):
        chunk = self.chunk("backend/src/services/courseRetrievalService.js", " ".join(["code"] * 90))
        adj = reranking_adjustments("course retrieval service", chunk)
        self.assertEqual(adj["path_role_adjustment"], 0.05)

    def test_readme_penalized_for_implementation_query(self):
        chunk = self.chunk("README.md", " ".join(["project"] * 100))
        adj = reranking_adjustments("show the implementation code", chunk)
        self.assertEqual(adj["source_type_adjustment"], -0.035)

    def test_readme_not_penalized_for_documentation_query(self):
        chunk = self.chunk("README.md", " ".join(["project"] * 100))
        adj = reranking_adjustments("find the project documentation", chunk)
        self.assertEqual(adj["source_type_adjustment"], 0.0)

    def test_combined_modifier_is_capped(self):
        chunk = self.chunk("README.md", "short")
        adj = reranking_adjustments("implementation code", chunk)
        self.assertGreaterEqual(adj["modifier_total"], -0.06)
        self.assertLessEqual(adj["modifier_total"], 0.06)
        self.assertEqual(adj["modifier_total"], -0.06)

    def test_compare_persist_exact_filename_keeps_strong_match(self):
        compare = self.chunk("frontend/src/stores/compare.js", "export const state = {}", "compareState")
        adj = reranking_adjustments("persistent compare state", compare, name_score=1.0)
        self.assertEqual(adj["tiny_adjustment"], 0.0)
        self.assertGreater(adj["modifier_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
