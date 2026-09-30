---
name: smart-context
description: Reduce context use on medium or large codebases by incrementally indexing source files and retrieving a small, semantically ranked set of code chunks before coding. Use when a coding task depends on existing repository code.
---

# Smart Context

Use this skill before exploring a medium or large repository when a coding task depends on existing project code.

1. Run `python <skill>/scripts/update_index.py` from the target repository. On a first run it builds the index. The index is local under that repository's `cache/` directory and must not be committed.
2. Run `python <skill>/scripts/retrieve.py --query "<the user's current task>"` (optionally `--top-k N`, `--max-tokens N`, or `--json`).
3. Inspect the returned chunks before searching elsewhere. The tool performs query embedding, cosine search, reranking, limited direct dependency expansion, Git recency adjustment, and token-budget filtering.
4. Follow imports or dependencies only when needed. If retrieval is insufficient, expand progressively:
   - Stage 1: top semantic results.
   - Stage 2: direct dependencies and exact symbol matches.
   - Stage 3: targeted grep/search for the missing symbol or behavior.
   - Stage 4: broader repository inspection only if prior stages fail.
5. Respect the configured context budget. Do not include unrelated code just to fill it.

## Adaptive chunking (V2)

Index updates keep the deterministic local chunker as the source of truth. Large, mixed, or structurally ambiguous regions may be refined by GLM-4-Flash when `semantic_chunking.enabled` is true. Set `ZHIPU_API_KEY` in the current process before indexing if remote refinement is wanted; if it is unavailable or the response is invalid, indexing falls back to local chunks. Semantic plans are cached under the repository's `cache/semantic_chunk_plans.json`; the cache contains content hashes and boundary metadata, not credentials or source text.

Set `semantic_chunking.enabled` to `false` in `config/config.json` to use the V1 local boundaries and ranking behavior. BGE-M3 embeddings still contain original source code only; GLM purpose metadata is not substituted for source text.

Do not read the whole repository or run commands that recursively print all source files. Install `requirements.txt` and configure `SILICONFLOW_API_KEY` before indexing or retrieval. In Windows PowerShell, set the key for the current session with `$env:SILICONFLOW_API_KEY="YOUR_KEY"`; persistent user-level configuration can be set separately in Windows environment settings. Never put the key in skill files, repository files, index metadata, or command output. If the variable is missing, the script reports `SILICONFLOW_API_KEY is not configured.` The index is a retrieval aid, not a substitute for checking current source before edits.
