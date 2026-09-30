# Smart Context v1.0.0

Smart Context is a codebase retrieval and context optimization tool for AI coding agents. It combines BGE-M3 semantic retrieval with a validated lightweight reranking layer and context-budget management.

## Why

Large repositories contain far more code than an LLM context window should receive. Smart Context retrieves a compact, relevant subset before the coding agent starts work.

## Architecture

Developer query → BGE-M3 query embedding → semantic candidate retrieval → optional V2.1 reranking → file-level deduplication → context construction → token budget → coding agent

## Retrieval modes

- **hybrid (default):** semantic retrieval plus the frozen V2.1 modifiers (tiny/no-symbol handling, filename/path role, and source/implementation role).
- **semantic:** semantic similarity only; this is the stable baseline.

Dependency, Git Recency, and Name/Path ranking weights are not part of either production mode. Experimental benchmark profiles remain available in the benchmark runner.

## Key features

- BGE-M3 semantic code retrieval and deterministic local chunking.
- Adaptive semantic chunking infrastructure with optional GLM-assisted boundary proposals and local fallback.
- SHA-256 content-addressed chunk identity and incremental/shared embedding reuse.
- Historical parent-state evaluation, ranking ablations, repository-level benchmark parallelism, and token-budget-aware context construction.

## Installation

Requires Python 3.10+ and the dependencies in `requirements.txt`. Retrieval and indexing use `SILICONFLOW_API_KEY`; GLM boundary proposals optionally use `ZHIPU_API_KEY`. Put keys in a private local `.env` copied from `.env.example`.

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Usage

From the Smart Context directory, index a target repository and retrieve context:

```powershell
python scripts\update_index.py --root C:\path\to\repository
python scripts\retrieve.py --root C:\path\to\repository --query "Find the request authentication flow"
python scripts\retrieve.py --root C:\path\to\repository --query "Find the request authentication flow" --mode hybrid
python scripts\retrieve.py --root C:\path\to\repository --query "Find the request authentication flow" --mode semantic
```

Use `--top-k N`, `--max-tokens N`, and `--json` to control or inspect the result. The default context budget is 6,000 tokens.

## Benchmark

Validation used 352 examples from CourseCompass (3), Express (169), and Flask (180). The frozen Semantic + V2.1 profile achieved micro MRR 0.472 versus 0.395 for Semantic Only.

The final held-out sanity test used 100 untouched examples (50 Express, 50 Flask). Semantic Only scored MRR 0.335; Semantic + V2.1 scored 0.447, a +0.1117 paired delta (95% bootstrap CI [+0.0631, +0.1630]). Hit@1/3/5 changed 0.220→0.350, 0.430→0.550, and 0.510→0.570; Recall@5 changed 0.409→0.489. Average context tokens increased from 1,769 to 2,516: V2.1 improved ranking quality but used more context.

### Fixed context-budget evaluation

The same 100 held-out examples were evaluated at six maximum context budgets. Under every shared cap, Hybrid achieved higher MRR than Semantic Only. At the 2,000-token cap, MRR improved from 0.338 to 0.448, Hit@5 from 0.510 to 0.570, and Recall@5 from 0.404 to 0.484. MRR per 1,000 actual context tokens rose from 0.319 to 0.335.

| Maximum budget | Semantic MRR / avg tokens | Hybrid MRR / avg tokens | Semantic MRR per 1k tokens | Hybrid MRR per 1k tokens |
|---:|---:|---:|---:|---:|
| 1,000 | 0.321 / 642 | 0.423 / 784 | 0.501 | 0.539 |
| 1,500 | 0.336 / 858 | 0.441 / 1,083 | 0.392 | 0.407 |
| 2,000 | 0.338 / 1,060 | 0.448 / 1,337 | 0.319 | 0.335 |
| 3,000 | 0.333 / 1,364 | 0.446 / 1,763 | 0.244 | 0.253 |
| 4,000 | 0.337 / 1,585 | 0.447 / 2,118 | 0.212 | 0.211 |
| 6,000 | 0.335 / 1,769 | 0.447 / 2,516 | 0.189 | 0.178 |

These are equal **maximum** budgets, not equal realized token counts: Hybrid used more context on average at every cap. Its MRR-per-token ratio was higher through the 3,000-token cap, approximately tied at 4,000, and lower at 6,000. In particular, the 2,000-token cap improved retrieval quality with about 277 additional average context tokens (1,337 versus 1,060); it does not demonstrate a reduction in reading tokens. Full metrics and paired budget deltas are in the [aggregate results](benchmarks/generated_multi_repo/fixed_context_budget_results.json), [per-query results](benchmarks/generated_multi_repo/fixed_context_budget_per_query.json), and [paired deltas](benchmarks/generated_multi_repo/fixed_context_budget_paired_deltas.json).

See [benchmarks/BENCHMARK.md](benchmarks/BENCHMARK.md) for methods and full results.

## Limitations

- Git-touched files are weak supervision and may not represent every file needed for a task.
- The final benchmark uses three repositories; the CourseCompass validation sample is very small.
- The held-out sanity test uses 100 examples rather than the full remaining test set.
- Results are not universal performance claims across languages or repositories.
- Adaptive routing and learned reranking are future work.

## Future work

Adaptive/query-aware routing, learned reranking, larger multi-repository evaluation, more languages, incremental historical index assembly, and optional local embedding inference.

## Tests

```powershell
python -B -m unittest discover -v -s tests
```
