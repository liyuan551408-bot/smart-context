# Smart Context

Smart Context indexes a codebase and retrieves a small set of relevant source chunks for a coding task. It combines semantic search with dependency, filename, Git recency, and working-tree signals to help focus repository exploration.

## Requirements

- Python 3.10 or newer
- A SiliconFlow API key for BGE-M3 embeddings
- Optional: a Zhipu API key when semantic chunk refinement is enabled

Install the Python dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

Set the embedding key in the current PowerShell session. Keep credentials out of repository files and command output.

```powershell
$env:SILICONFLOW_API_KEY = "YOUR_KEY"
```

## Index a repository

Run the index updater from the repository you want to work on. The generated index is local to that repository and should not be committed.

```powershell
python C:\path\to\smart-context\scripts\update_index.py
```

For optional GLM semantic chunk refinement, set `ZHIPU_API_KEY` before indexing. If refinement is unavailable or returns an invalid plan, indexing falls back to local chunking. Set `semantic_chunking.enabled` to `false` in `config/config.json` to use local chunk boundaries only.

## Retrieve context

Run retrieval from the target repository, or pass the repository path with `--root`:

```powershell
python C:\path\to\smart-context\scripts\retrieve.py --query "Find the course retrieval implementation"
```

Useful options:

- `--top-k N` limits the number of returned chunks.
- `--max-tokens N` sets the context budget.
- `--json` emits machine-readable output, including ranking scores and reranking adjustments.
- `--root PATH` selects the repository to search.

The default context budget is 6,000 tokens. Retrieval adjusts ranking conservatively for very small chunks, generic filename/path term overlap, and documentation or package files when a query clearly asks for implementation. Semantic similarity remains the main signal, and exact meaningful short-name matches are protected from the tiny-chunk penalty.

Inspect retrieved chunks before editing source files. The index is a navigation aid and does not replace checking the current source.

## Tests

Run the smart-context unit suite from this directory:

```powershell
python -m unittest discover -v -s tests
```
