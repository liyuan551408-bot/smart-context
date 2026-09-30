import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "benchmarks"))

from build_git_benchmark import (  # noqa: E402
    assign_chronological_splits,
    classify_example,
    deduplicate_clean_records,
    dedup_summary,
    find_query_duplicates,
    history_commits,
    is_merge_commit,
    is_source_path,
    message_quality,
    parse_name_status_z,
    summarize_changes,
)
from evaluate_git_benchmark import deduplicate_ranked_files, score_ranked_files  # noqa: E402
from build_multi_repo_benchmark import (  # noqa: E402
    add_repository_metadata,
    combined_totals,
    combine_records,
    collect_history_batched,
    deduplicate_per_repository,
    duplicate_group_split_violations,
    load_or_collect_history,
)


class GitBenchmarkBuilderTests(unittest.TestCase):
    @staticmethod
    def clean_row(commit, query, targets, timestamp):
        return {
            "commit": commit, "parent": "parent-" + commit, "query": query,
            "targets": targets, "category": "clean", "split": "excluded",
            "timestamp": timestamp, "meaningful_word_count": len(query.split()),
        }

    def test_merge_commit_is_excluded(self):
        self.assertTrue(is_merge_commit(["parent-a", "parent-b"]))
        self.assertFalse(is_merge_commit(["parent-a"]))

    def test_all_non_merge_history_walk_uses_head_only_and_deduplicates_shas(self):
        with patch("build_git_benchmark.git", return_value="sha-a\nsha-a\nsha-b\n") as git_call:
            commits = history_commits(Path("repo"), "all-non-merge")
        self.assertEqual(commits, ["sha-a", "sha-b"])
        args = git_call.call_args.args[1]
        self.assertEqual(args, ["rev-list", "--reverse", "--no-merges", "HEAD"])
        self.assertNotIn("--all", args)

    def test_batched_history_parses_metadata_and_parent_state_paths(self):
        log = (
            b"\x1esha-a\x00\x00time-a\x00author-a\x00Implement course search service\x00\n"
            b"M\x00src/a.js\x00"
            + "\x1esha-b\x00sha-a\x00time-b\x00author-b\x00Implement búsqueda profile service behavior\x00".encode("utf-8")
            + b"A\x00src/new.js\x00M\x00src/existing.js\x00"
            + b"\x1esha-c\x00sha-b\x00time-c\x00author-c\x00Rename legacy search service module\x00"
            + b"R100\x00src/old.js\x00src/new name.js\x00"
            + b"\x1esha-d\x00sha-c\x00time-d\x00author-d\x00Remove deprecated cache cleanup service\x00"
            + b"D\x00src/gone.js\x00"
        )
        with patch("build_multi_repo_benchmark.git", side_effect=["sha-a\nsha-a\nsha-b\nsha-c\nsha-d\n", log, "merge-sha\n"]):
            commits, rows, merge_count = collect_history_batched(Path("repo"))
        self.assertEqual(commits, ["sha-a", "sha-b", "sha-c", "sha-d"])
        self.assertEqual(merge_count, 1)
        self.assertEqual(rows[0]["targets"], ["src/a.js"])
        self.assertEqual(rows[1]["targets"], ["src/existing.js"])
        self.assertEqual(rows[1]["added_files"], ["src/new.js"])
        self.assertEqual(rows[2]["targets"], ["src/old.js"])
        self.assertEqual(rows[2]["renamed_files"], ["src/old.js"])
        self.assertEqual(rows[2]["all_changed_files"], ["src/new name.js", "src/old.js"])
        self.assertEqual(rows[3]["targets"], ["src/gone.js"])
        self.assertEqual(rows[0]["parent"], None)
        self.assertEqual(rows[1]["parent"], "sha-a")
        self.assertEqual(rows[1]["query"], "Implement búsqueda profile service behavior")

    def test_metadata_cache_hits_for_same_repository_head_and_history_mode(self):
        with tempfile.TemporaryDirectory() as cache_directory:
            cached_records = [{"commit": "sha-a", "query": "Implement search service", "targets": ["src/a.js"]}]
            with patch("build_multi_repo_benchmark.CACHE_DIR", Path(cache_directory)), \
                 patch("build_multi_repo_benchmark.repository_head", return_value="head-sha"), \
                 patch("build_multi_repo_benchmark.collect_history_batched", return_value=(["sha-a"], cached_records, 2)) as collect:
                first = load_or_collect_history(Path("repo"), "Pinia")
                self.assertFalse(first[3])
                collect.side_effect = AssertionError("cache miss unexpectedly reread Git history")
                second = load_or_collect_history(Path("repo"), "Pinia")
                self.assertTrue(second[3])
                self.assertEqual(second[:3], first[:3])

    def test_weak_commit_message_is_classified_noisy(self):
        self.assertTrue(message_quality("update")[0])
        category, reason, _, _ = classify_example("update", summarize_changes([
            {"status": "M", "old_path": "", "path": "src/search.js"},
        ]))
        self.assertEqual(category, "noisy")
        self.assertIn("generic", reason)

    def test_modified_file_becomes_target(self):
        result = summarize_changes([{"status": "M", "old_path": "", "path": "src/searchService.js"}])
        self.assertEqual(result["targets"], ["src/searchService.js"])
        self.assertEqual(result["modified_files"], ["src/searchService.js"])

    def test_deleted_file_becomes_parent_state_target(self):
        result = summarize_changes([{"status": "D", "old_path": "src/oldService.js", "path": "src/oldService.js"}])
        self.assertEqual(result["targets"], ["src/oldService.js"])
        self.assertEqual(result["deleted_files"], ["src/oldService.js"])

    def test_rename_uses_old_path_as_target(self):
        result = summarize_changes([{"status": "R095", "old_path": "src/oldName.js", "path": "src/newName.js"}])
        self.assertEqual(result["targets"], ["src/oldName.js"])
        self.assertEqual(result["renamed_files"], ["src/oldName.js"])
        self.assertEqual(result["all_changed_files"], ["src/newName.js", "src/oldName.js"])

    def test_git_name_status_parser_preserves_both_rename_paths(self):
        parsed = parse_name_status_z(b"R100\0src/old.js\0src/new.js\0M\0src/other.js\0")
        self.assertEqual(parsed, [
            {"status": "R100", "old_path": "src/old.js", "path": "src/new.js"},
            {"status": "M", "old_path": "", "path": "src/other.js"},
        ])

    def test_added_only_change_is_new_file_category(self):
        changes = summarize_changes([{"status": "A", "old_path": "", "path": "src/courseSearch.js"}])
        category, _, _, _ = classify_example("Add course search endpoint", changes)
        self.assertEqual(category, "new_file")
        self.assertEqual(changes["targets"], [])
        self.assertEqual(changes["added_files"], ["src/courseSearch.js"])

    def test_mixed_addition_and_parent_state_modification_keeps_modified_target(self):
        changes = summarize_changes([
            {"status": "A", "old_path": "", "path": "src/newSearch.js"},
            {"status": "M", "old_path": "", "path": "src/searchService.js"},
        ])
        category, _, _, _ = classify_example("Implement course search service behavior", changes)
        self.assertEqual(category, "clean")
        self.assertEqual(changes["targets"], ["src/searchService.js"])

    def test_source_extension_and_ignored_path_filtering(self):
        self.assertTrue(is_source_path("src/app.tsx"))
        self.assertTrue(is_source_path("server/model.prisma"))
        self.assertFalse(is_source_path("docs/guide.md"))
        self.assertFalse(is_source_path("node_modules/pkg/index.js"))
        self.assertFalse(is_source_path("backend/package-lock.json"))

    def test_chronological_splits_use_clean_records_only(self):
        records = [{"category": "clean", "split": "excluded"} for _ in range(10)]
        records.insert(4, {"category": "noisy", "split": "train"})
        assign_chronological_splits(records)
        self.assertEqual([r["split"] for r in records if r["category"] == "clean"],
                         ["train"] * 7 + ["validation"] + ["test"] * 2)
        self.assertEqual(records[4]["split"], "excluded")

    def test_action_word_variants_on_same_target_are_flagged_for_review(self):
        rows = [
            {"commit": "a", "query": "Fix search endpoint", "category": "clean", "split": "train", "targets": ["src/search.js"]},
            {"commit": "b", "query": "Update search endpoint", "category": "clean", "split": "test", "targets": ["src/search.js"]},
        ]
        groups = find_query_duplicates(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["count"], 2)
        self.assertIn("review", groups[0]["review_reason"])

    def test_near_duplicate_terms_on_shared_target_are_flagged(self):
        rows = [
            {"commit": "a", "query": "Update CI and add health check", "category": "clean", "split": "validation", "targets": ["backend/app.js"]},
            {"commit": "b", "query": "ci: expand backend checks and add configurable server startup", "category": "clean", "split": "test", "targets": ["backend/app.js"]},
        ]
        self.assertTrue(any(group["count"] == 2 for group in find_query_duplicates(rows)))

    def test_clean_dedup_groups_exact_queries(self):
        rows = [
            self.clean_row("a", "Add course search service", ["src/a.js"], "2024-01-01"),
            self.clean_row("b", "ADD course search service!", ["src/b.js"], "2024-01-02"),
        ]
        result = deduplicate_clean_records(rows)
        self.assertEqual(sum(row["benchmark_included"] for row in result), 1)
        excluded = next(row for row in result if not row["benchmark_included"])
        self.assertEqual(excluded["duplicate_of"], "a")
        self.assertEqual(excluded["duplicate_reason"], "exact normalized query duplicate")

    def test_clean_dedup_near_matches_require_shared_target_and_overlap(self):
        rows = [
            self.clean_row("a", "Improve course search endpoint", ["src/shared.js"], "2024-01-01"),
            self.clean_row("b", "Improve course search handler", ["src/shared.js"], "2024-01-02"),
            self.clean_row("c", "Improve course search controller", ["src/other.js"], "2024-01-03"),
        ]
        result = deduplicate_clean_records(rows)
        self.assertEqual(sum(row["benchmark_included"] for row in result), 2)
        self.assertFalse(next(row for row in result if row["commit"] == "b")["benchmark_included"])
        self.assertTrue(next(row for row in result if row["commit"] == "c")["benchmark_included"])

    def test_clean_dedup_prefers_more_lexical_content_and_splits_after_grouping(self):
        rows = [
            self.clean_row("early", "Improve course search endpoint", ["src/shared.js"], "2024-01-01"),
            self.clean_row("later", "Implement robust course search endpoint handler", ["src/shared.js"], "2024-01-02"),
        ]
        result = deduplicate_clean_records(rows)
        representative = next(row for row in result if row["benchmark_included"])
        excluded = next(row for row in result if not row["benchmark_included"])
        self.assertEqual(representative["commit"], "later")
        self.assertIn(representative["split"], {"train", "validation", "test"})
        self.assertEqual(excluded["split"], "excluded")
        self.assertEqual(dedup_summary(result)["deduplicated_clean_count"], 1)

    def test_multi_repository_metadata_and_repository_name(self):
        row = self.clean_row("same-sha", "Implement course search endpoint", ["src/search.js"], "2024-01-01")
        annotated = add_repository_metadata([row], "Pinia", "https://github.com/vuejs/pinia.git", "Vue / TypeScript")
        self.assertEqual(annotated[0]["repository"], "Pinia")
        self.assertEqual(annotated[0]["repository_url"], "https://github.com/vuejs/pinia.git")
        self.assertEqual(annotated[0]["language_or_stack"], "Vue / TypeScript")

    def test_split_is_chronological_and_independent_within_each_repository(self):
        rows = []
        for repository in ("Pinia", "Flask"):
            for index in range(10):
                row = self.clean_row(
                    f"{repository}-{index}", f"Implement feature module item{index}",
                    [f"{repository}/src/item{index}.js"], f"2024-01-{index + 1:02d}",
                )
                row["repository"] = repository
                rows.append(row)
        result = deduplicate_per_repository(rows)
        for repository in ("Pinia", "Flask"):
            splits = [row["split"] for row in result if row["repository"] == repository and row["benchmark_included"]]
            self.assertEqual(splits, ["train"] * 7 + ["validation"] + ["test"] * 2)

    def test_combined_rows_keep_same_sha_from_different_repositories(self):
        rows = [
            {"repository": "Pinia", "commit": "same-sha"},
            {"repository": "Express", "commit": "same-sha"},
        ]
        combined = combine_records([{"records": rows[:1]}, {"records": rows[1:]}])
        self.assertEqual(len(combined), 2)
        self.assertEqual({row["repository"] for row in combined}, {"Pinia", "Express"})

    def test_duplicate_grouping_is_repository_local(self):
        rows = []
        for repository, commit in (("Pinia", "sha-1"), ("Express", "sha-1")):
            row = self.clean_row(commit, "Implement course search endpoint", ["src/search.js"], "2024-01-01")
            row["repository"] = repository
            rows.append(row)
        result = deduplicate_per_repository(rows)
        self.assertEqual(sum(row["benchmark_included"] for row in result), 2)
        self.assertEqual({row["repository"] for row in result}, {"Pinia", "Express"})

    def test_combined_summary_counts_namespaced_targets_and_split_leakage(self):
        dedup = []
        for repository, commit, split in (("Pinia", "sha-1", "test"), ("Express", "sha-1", "test")):
            row = self.clean_row(commit, f"Implement {repository} feature handler", ["src/file.js"], "2024-01-01")
            row.update({"repository": repository, "benchmark_included": True, "duplicate_group": commit, "split": split})
            dedup.append(row)
        repo_summaries = {
            name: {"commits_discovered": 4, "duplicate_groups": 0, "examples_removed_from_evaluation": 0,
                   "noisy": 1, "new_file": 2, "train": 0, "validation": 0, "test": 1}
            for name in ("Pinia", "Express")
        }
        totals = combined_totals(dedup, dedup, dedup, repo_summaries)
        self.assertEqual(totals["commits_discovered"], 8)
        self.assertEqual(totals["clean_deduplicated"], 2)
        self.assertEqual(totals["test"], 2)
        self.assertEqual(totals["unique_target_files"], 2)
        self.assertEqual(duplicate_group_split_violations(dedup), {})


class GitBenchmarkMetricsTests(unittest.TestCase):
    def test_ranked_chunk_files_are_deduplicated_in_first_seen_order(self):
        chunks = [
            {"chunk": {"file": "CompareCourses.vue"}},
            {"chunk": {"file": "CompareCourses.vue"}},
            {"chunk": {"file": "compare.js"}},
        ]
        self.assertEqual(deduplicate_ranked_files(chunks), ["CompareCourses.vue", "compare.js"])

    def test_hit_at_k_uses_first_relevant_deduplicated_rank(self):
        scores = score_ranked_files(["other.js", "target.js", "last.js"], ["target.js"])
        self.assertEqual(scores["hit@1"], 0)
        self.assertEqual(scores["hit@3"], 1)
        self.assertEqual(scores["hit@5"], 1)

    def test_mrr_is_reciprocal_of_first_relevant_rank(self):
        scores = score_ranked_files(["a.js", "target.js"], ["target.js"])
        self.assertEqual(scores["first_relevant_rank"], 2)
        self.assertEqual(scores["reciprocal_rank"], 0.5)

    def test_recall_at_5_counts_unique_relevant_files(self):
        scores = score_ranked_files(["a.js", "b.js", "target-a.js", "c.js"], ["target-a.js", "target-b.js"])
        self.assertEqual(scores["recall@5"], 0.5)


if __name__ == "__main__":
    unittest.main()
