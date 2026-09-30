# Smart Context v1.0 benchmark

## Dataset construction

The dataset uses Git-history weak supervision. The query is the commit subject (the first meaningful commit-message line), the target is the eligible source file or files touched by that commit, and evaluation uses the parent commit state. Examples are categorized as `clean`, `noisy`, or `new_file`; added-file examples are kept separate because the target does not exist in the parent state. Query/target deduplication is repository-local. Clean examples are split chronologically into train, validation, and test sets; splits are fixed and are not regenerated for these results.

Git-touched files are weak labels: a commit can touch only part of the code required to understand a task.

## Earlier diagnostic pilot

The earlier 63-query pilot was diagnostic only. It is not part of the final v1 validation or held-out headline figures. Pinia experiments are also diagnostic and excluded from final v1 aggregates.

## Final validation ablation

The existing validation split was filtered to CourseCompass (3), Express (169), and Flask (180), for 352 examples. No resampling or query changes were made. Ten profiles were compared:

1. Semantic Only
2. Semantic + Name/Path
3. Semantic + Dependency
4. Semantic + V2.1 Reranking
5. Semantic + Git Recency
6. Full minus Name/Path
7. Full minus Dependency
8. Full minus V2.1 Reranking
9. Full minus Git Recency
10. Current Full Smart Context

Semantic + V2.1 was selected using micro MRR as the predeclared primary metric. It scored 0.472 MRR, compared with 0.395 for Semantic Only. Its changes versus baseline improved the first relevant rank on 85 queries and worsened it on 35 (net +50); 31 relevant files entered Top-5 and 12 left (net +19). Dependency as a standalone signal was harmful; removing it improved the full profile. Name/Path was mildly helpful, Git Recency had modest repository-sensitive effects, and Current Full Smart Context underperformed the selected V2.1 profile. The frozen production default therefore excludes Dependency, Git Recency, and Name/Path ranking weights.

| Profile | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|
| Semantic Only | 0.287 | 0.483 | 0.557 | 0.395 | 0.449 | 1722 |
| Semantic + Name/Path | 0.324 | 0.489 | 0.571 | 0.419 | 0.466 | 1688 |
| Semantic + Dependency | 0.239 | 0.446 | 0.506 | 0.345 | 0.404 | 1528 |
| Semantic + V2.1 Reranking | 0.372 | 0.568 | 0.611 | 0.472 | 0.498 | 2263 |
| Semantic + Git Recency | 0.287 | 0.509 | 0.568 | 0.405 | 0.467 | 2052 |
| Full minus Name/Path | 0.318 | 0.540 | 0.594 | 0.428 | 0.485 | 2132 |
| Full minus Dependency | 0.369 | 0.551 | 0.619 | 0.466 | 0.512 | 2362 |
| Full minus V2.1 Reranking | 0.293 | 0.523 | 0.574 | 0.406 | 0.474 | 1732 |
| Full minus Git Recency | 0.327 | 0.523 | 0.585 | 0.429 | 0.465 | 1941 |
| Current Full Smart Context | 0.344 | 0.554 | 0.602 | 0.447 | 0.492 | 2060 |

The final validation artifacts are `generated_multi_repo/ranking_ablation_validation_3repo_results.json` and `generated_multi_repo/ranking_ablation_validation_3repo_per_query.json`.

## Final held-out sanity test

The manifest was persisted before scoring or cache/index inspection. It contains a deterministic uniform sample without replacement from untouched eligible test records: 50 Express and 50 Flask, seed `20260930`. Validation examples, Pinia, and records used in the earlier 63-query diagnostic pilot were excluded. Exactly two frozen modes were evaluated; no ranking changes followed this evaluation.

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Avg tokens | Median | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Semantic Only | 0.220 | 0.430 | 0.510 | 0.335 | 0.409 | 1769 | 951 | 5927 |
| Semantic + V2.1 | 0.350 | 0.550 | 0.570 | 0.447 | 0.489 | 2516 | 1840 | 5996 |

| Repository | Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Recall@5 | Avg tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Express | Semantic Only | 0.160 | 0.320 | 0.420 | 0.251 | 0.327 | 3026 |
| Express | Semantic + V2.1 | 0.240 | 0.480 | 0.500 | 0.355 | 0.393 | 4224 |
| Flask | Semantic Only | 0.280 | 0.540 | 0.600 | 0.419 | 0.492 | 512 |
| Flask | Semantic + V2.1 | 0.460 | 0.620 | 0.640 | 0.538 | 0.584 | 808 |

V2.1 minus Semantic Only MRR delta was `+0.1117`. A paired query bootstrap with 10,000 resamples and seed `20260930` gave a 95% percentile interval of `[+0.0631, +0.1630]`. V2.1 improved ranking on this sample and increased average context tokens from 1,769 to 2,516. This is a compact sanity test, not a full test-set estimate.

The persisted artifacts are `generated_multi_repo/final_heldout_compact_manifest.json` and `generated_multi_repo/final_heldout_compact_results.json`.

## Reproduction and caches

`run_historical_pilot.py` is the canonical historical runner. Existing indexes are keyed by repository and parent commit; shared embeddings use repository-scoped exact-content identity; query vectors have a separate cache. Repository parallelism assigns one serial parent-state worker per repository, avoiding parallel parent processing inside a repository. The fixed final held-out manifest is reused when present. Example invocations (run only when an evaluation is intended):

```powershell
python benchmarks\run_historical_pilot.py --stage ablation --dataset benchmarks\generated_multi_repo\benchmark_validation.jsonl --repositories CourseCompass Express Flask --parallel-repositories 3 --output-prefix ranking_ablation_validation_3repo
python benchmarks\run_historical_pilot.py --stage heldout --manifest benchmarks\generated_multi_repo\final_heldout_compact_manifest.json --parallel-repositories 2
```

Generated repository clones, indexes, worktrees, embedding vectors, and intermediate datasets are local benchmark data and should not be committed.

## Limitations

- Touched files provide weak supervision and may omit context a task needs.
- The headline validation covers three repositories; CourseCompass contributes only three validation examples.
- The held-out sanity test samples 100 remaining test records, not the entire remaining test set.
- Results do not establish general performance across all languages, repository sizes, or task types.
- The held-out sample is a final check of a frozen choice, not an opportunity to tune that choice.
