"""Run a deterministic parent-state retrieval pilot with shared historical indexes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import statistics
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
SKILL_DIR = BENCHMARK_DIR.parent
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(BENCHMARK_DIR))

import retrieve as retrieve_module  # noqa: E402
import update_index as update_index_module  # noqa: E402
import utils  # noqa: E402
from build_multi_repo_benchmark import REPOSITORIES  # noqa: E402
from evaluate_git_benchmark import deduplicate_ranked_files, score_ranked_files  # noqa: E402


DATASET = BENCHMARK_DIR / "generated_multi_repo" / "benchmark_test.jsonl"
OUTPUT_DIR = BENCHMARK_DIR / "generated_multi_repo"
WORKSPACE = BENCHMARK_DIR / "cache" / "historical_pilot"
COUNTS = {"CourseCompass": 3, "Pinia": 20, "Express": 20, "Flask": 20}
MODES = (("Semantic-only", "semantic-only"), ("No Git Recency", "no-git-recency"), ("Full Smart Context", "full"))
ABLATION_MODES = ()
SHARED_EMBEDDING_SCHEMA_VERSION = 1
EMBEDDING_FORMAT = "numpy-f32-unit-v1"
REUSE_STATS_SCHEMA_VERSION = 1


def build_ablation_modes(weights=None):
    """Return evaluation-only profiles isolating each ranking signal."""
    current = dict(weights or utils.CONFIG["weights"])
    semantic_weight = float(current.get("semantic", 1.0))

    def profile(enabled=(), *, v21=False):
        values = {key: 0.0 for key in current}
        values["semantic"] = semantic_weight
        for key in enabled:
            values[key] = float(current[key])
        return {"weights": values, "v21_enabled": v21}

    semantic_only = {"weights": {key: (1.0 if key == "semantic" else 0.0) for key in current}, "v21_enabled": False}
    full = {"weights": current, "v21_enabled": True}
    no_name, no_dependency, no_v21, no_git = (dict(full) for _ in range(4))
    no_name["weights"] = {**current, "name_match": 0.0}
    no_dependency["weights"] = {**current, "dependency": 0.0}
    no_git["weights"] = {**current, "git_recency": 0.0}
    no_v21["v21_enabled"] = False
    return (
        ("Semantic Only", semantic_only),
        ("Semantic + Name/Path", profile(("name_match",))),
        ("Semantic + Dependency", profile(("dependency",))),
        ("Semantic + V2.1 Reranking", profile((), v21=True)),
        ("Semantic + Git Recency", profile(("git_recency",))),
        ("Full minus Name/Path", no_name),
        ("Full minus Dependency", no_dependency),
        ("Full minus V2.1 Reranking", no_v21),
        ("Full minus Git Recency", no_git),
        ("Current Full Smart Context", full),
    )


ABLATION_MODES = build_ablation_modes()
HELDOUT_MODES = tuple(mode for mode in ABLATION_MODES
                      if mode[0] in {"Semantic Only", "Semantic + V2.1 Reranking"})


def modes_for_stage(stage):
    if stage == "ablation":
        return ABLATION_MODES
    if stage == "heldout":
        return HELDOUT_MODES
    return MODES


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def record_chronology_key(record):
    value = str(record.get("timestamp", ""))
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (parsed.timestamp(), record.get("commit", ""))
    except ValueError:
        return (float("inf"), value, record.get("commit", ""))


def _terms(query):
    return set(re.findall(r"[a-z0-9]+", query.lower()))


def _near_duplicate(left, right):
    if left.get("repository") != right.get("repository"):
        return False
    if not (set(left.get("targets", [])) & set(right.get("targets", []))):
        return False
    a, b = _terms(left.get("query", "")), _terms(right.get("query", ""))
    return bool(a and b and len(a & b) / len(a | b) >= 0.62)


def sample_pilot_records(records, counts=COUNTS):
    """Select per-repository examples greedily for file/directory diversity."""
    selected = []
    for repository, wanted in counts.items():
        pool = [row for row in records if row.get("repository") == repository]
        if len(pool) < wanted:
            raise ValueError(f"{repository} has {len(pool)} records, needs {wanted}")
        pool.sort(key=lambda row: (row.get("timestamp", ""), row.get("commit", "")))
        chosen = []
        # CourseCompass' complete test split is intentionally kept as-is.
        if len(pool) == wanted:
            chosen = list(pool)
        else:
            total_single = sum(len(set(x.get("targets", []))) == 1 for x in pool)
            single_quota = min(wanted - 1, max(1, round(wanted * total_single / len(pool))))
            while len(chosen) < wanted:
                remaining = [x for x in pool if x not in chosen]
                nonduplicate = [x for x in remaining if not any(_near_duplicate(x, y) for y in chosen)]
                candidates = nonduplicate or remaining
                single_chosen = sum(len(set(x.get("targets", []))) == 1 for x in chosen)
                need_single = single_chosen < single_quota
                if need_single:
                    preferred = [x for x in candidates if len(set(x.get("targets", []))) == 1]
                elif len(chosen) - single_chosen < wanted - single_quota:
                    preferred = [x for x in candidates if len(set(x.get("targets", []))) > 1]
                else:
                    preferred = candidates
                if preferred:
                    candidates = preferred
                known_targets = {t for row in chosen for t in set(row.get("targets", []))}
                known_dirs = {str(Path(t).parent).replace("\\", "/") for row in chosen for t in row.get("targets", [])}
                def diversity(row):
                    targets = set(row.get("targets", []))
                    dirs = {str(Path(t).parent).replace("\\", "/") for t in targets}
                    novelty = len(targets - known_targets) + 0.35 * len(dirs - known_dirs)
                    # Earlier ties reduce future-information bias; SHA stabilizes exact ties.
                    return (novelty, -pool.index(row), row.get("commit", ""))
                chosen.append(max(candidates, key=diversity))
        selected.extend(chosen)
    return selected


def write_manifest(records, path=OUTPUT_DIR / "pilot_test_63.jsonl"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    return Path(path)


def write_final_heldout_manifest(records, path, *, seed=20260930, dataset_path=DATASET,
                                 pilot_path=OUTPUT_DIR / "pilot_test_63.jsonl"):
    """Uniformly sample the frozen 50+50 untouched test set and persist it first."""
    try:
        pilot_commits = {row["commit"] for row in read_jsonl(pilot_path)}
    except OSError:
        pilot_commits = set()
    eligible = [row for row in records if row.get("category", "clean") == "clean"
                and row.get("benchmark_included", True) and row.get("split") == "test"
                and row.get("repository") in ("Express", "Flask")
                and row.get("commit") not in pilot_commits]
    rng = random.Random(seed)
    selected = []
    expected = {"Express": 50, "Flask": 50}
    for repository, count in expected.items():
        pool = sorted((row for row in eligible if row["repository"] == repository),
                      key=lambda row: row["commit"])
        if len(pool) < count:
            raise ValueError(f"{repository} has {len(pool)} remaining test records; needs {count}")
        selected.extend(rng.sample(pool, count))
    selected.sort(key=lambda row: (row["repository"], row.get("timestamp", ""), row["commit"]))
    manifest = {"seed": seed,
        "sampling": "uniform random sample without replacement from eligible untouched test records",
        "source_dataset": str(dataset_path), "previous_63_query_pilot_commits_excluded": True,
        "repositories": ["Express", "Flask"], "examples_per_repository": expected,
        "validation_examples_used": 0, "examples": selected}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_final_heldout_manifest(manifest):
    rows = manifest.get("examples", [])
    counts = Counter(row.get("repository") for row in rows)
    if len(rows) != 100 or counts != {"Express": 50, "Flask": 50}:
        raise ValueError(f"Final held-out manifest must contain 50 Express + 50 Flask examples; got {dict(counts)}")
    if any(row.get("split") != "test" or row.get("category", "clean") != "clean"
           or not row.get("benchmark_included", True) for row in rows):
        raise ValueError("Final held-out manifest contains an ineligible record")
    return rows


def smoke_records(records):
    first = {}
    for record in records:
        first.setdefault(record["repository"], record)
    if set(first) != set(COUNTS):
        raise ValueError("Smoke test requires one sample from each of the four repositories")
    return [first[name] for name in COUNTS]


def make_workspace_repo(repository):
    source = Path(REPOSITORIES[repository]["path"]).resolve()
    clone = WORKSPACE / "clones" / repository.lower()
    clone.parent.mkdir(parents=True, exist_ok=True)
    if not (clone / ".git").exists():
        print(f"[CLONE] {repository} into isolated benchmark cache", flush=True)
        subprocess.run(["git", "clone", "--shared", "--no-checkout", str(source), str(clone)], check=True, capture_output=True, text=True)
    return clone


def prepare_parent_worktree(repository, parent):
    clone = make_workspace_repo(repository)
    worktree = WORKSPACE / "worktrees" / repository.lower() / parent
    worktree.parent.mkdir(parents=True, exist_ok=True)
    if not worktree.is_dir():
        print(f"[WORKTREE] {repository} parent={parent[:12]}", flush=True)
        subprocess.run(["git", "-C", str(clone), "worktree", "add", "--detach", str(worktree), parent], check=True, capture_output=True, text=True)
    evaluate_git = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if evaluate_git != parent:
        raise ValueError(f"Historical worktree is at {evaluate_git}; expected parent {parent}")
    return worktree


def _state_cache_dir(repository, parent):
    return WORKSPACE / "indexes" / repository.lower() / parent


def _repository_identity(repository):
    config = REPOSITORIES[repository]
    if config.get("url"):
        return config["url"].rstrip("/").lower()
    return str(Path(config["path"]).resolve()).replace("\\", "/").lower()


class SharedEmbeddingCache:
    """Repository-scoped content-addressed cache; it stores vectors only."""

    def __init__(self, root, repository, *, model, chunker_version, dimension=None,
                 schema_version=SHARED_EMBEDDING_SCHEMA_VERSION, embedding_format=EMBEDDING_FORMAT,
                 embed_fn=None):
        self.root = Path(root)
        self.repository = str(repository)
        self.model = str(model)
        self.chunker_version = str(chunker_version)
        self.expected_dimension = int(dimension) if dimension is not None else None
        self.schema_version = int(schema_version)
        self.embedding_format = str(embedding_format)
        self.embed_fn = embed_fn or utils.embed
        self.stats = Counter()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "cache_state.json"
        self.state = {}
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            self.state = {}

    def _profile_key(self):
        return hashlib.sha256(json.dumps({
            "repository": self.repository, "embedding_model": self.model,
            "chunker_version": self.chunker_version, "embedding_format": self.embedding_format,
            "cache_schema_version": self.schema_version,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _dimension(self):
        if self.expected_dimension is not None:
            return self.expected_dimension
        value = self.state.get(self._profile_key(), {}).get("embedding_dimension")
        return int(value) if value is not None else None

    def _vector_path(self, content_hash, dimension):
        identity = {
            "repository": self.repository,
            "content_hash": content_hash,
            "embedding_model": self.model,
            "embedding_dimension": int(dimension),
            "effective_chunker_version": self.chunker_version,
            "embedding_format": self.embedding_format,
            "embedding_cache_schema_version": self.schema_version,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return self.root / f"{key}.npy"

    def embed_chunks(self, texts):
        import numpy as np
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = [None] * len(texts)
        pending = {}
        dimension = self._dimension()
        for position, content in enumerate(texts):
            content_hash = utils.content_hash(content)
            path = self._vector_path(content_hash, dimension) if dimension is not None else None
            if path is not None and path.exists():
                try:
                    vector = np.load(path)
                    if vector.ndim == 1 and vector.shape[0] == dimension:
                        vectors[position] = vector.astype(np.float32, copy=False)
                        self.stats["embedding_cache_hits"] += 1
                        continue
                except (OSError, ValueError):
                    pass
            if content_hash in pending:
                pending[content_hash]["positions"].append(position)
                # A duplicate seen again within this same index build avoids another
                # API vector, but is not a hit from the persistent shared cache.
                self.stats["intra_index_dedup_hits"] += 1
            else:
                pending[content_hash] = {"content": content, "hash": content_hash, "positions": [position]}
                self.stats["embedding_cache_misses"] += 1
            self.stats["embedding_cache_misses"] += int(content_hash in pending and len(pending[content_hash]["positions"]) > 1)
        misses = list(pending.values())
        self.stats["embeddings_requested"] += len(misses)
        if misses:
            embedded = self.embed_fn([entry["content"] for entry in misses])
            if embedded.ndim != 2 or embedded.shape[0] != len(misses):
                raise RuntimeError("Shared embedding cache received an invalid embedding batch")
            actual_dimension = int(embedded.shape[1])
            if dimension is not None and actual_dimension != dimension:
                raise RuntimeError(f"Embedding dimension changed from {dimension} to {actual_dimension}")
            dimension = actual_dimension
            if self.expected_dimension is not None and dimension != self.expected_dimension:
                raise RuntimeError(f"Embedding dimension {dimension} differs from configured {self.expected_dimension}")
            for row, entry in zip(embedded, misses):
                vector = np.asarray(row, dtype=np.float32)
                cache_path = self._vector_path(entry["hash"], dimension)
                temporary = cache_path.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, vector)
                temporary.replace(cache_path)
                for position in entry["positions"]:
                    vectors[position] = vector
            profile = self._profile_key()
            self.state[profile] = {"embedding_dimension": dimension}
            temporary_state = self.state_path.with_suffix(".json.tmp")
            temporary_state.write_text(json.dumps(self.state, sort_keys=True), encoding="utf-8")
            temporary_state.replace(self.state_path)
        if any(vector is None for vector in vectors):
            raise RuntimeError("Shared embedding cache failed to assemble every chunk vector")
        return np.stack(vectors).astype(np.float32)

    @property
    def entry_count(self):
        return sum(1 for path in self.root.glob("*.npy") if path.is_file())


def shared_embedding_cache_dir(repository):
    safe = "".join(char.lower() if char.isalnum() else "_" for char in repository).strip("_")
    return WORKSPACE / "shared_embeddings" / safe


def choose_close_parent_states(repository="Pinia"):
    """Choose adjacent first-parent evaluation states with empty parent/shared caches."""
    repo = Path(REPOSITORIES[repository]["path"]).resolve()
    output = subprocess.run(["git", "-C", str(repo), "log", "--first-parent", "--format=%H%x00%P", "HEAD", "-n", "500"],
                            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    history = []
    seen = set()
    for line in output.splitlines():
        fields = line.split("\x00", 1)
        if len(fields) != 2 or fields[0] in seen:
            continue
        seen.add(fields[0])
        parents = fields[1].split()
        if len(parents) == 1:
            history.append((fields[0], parents[0]))
    # Git log is newest first; an older commit's own parent and that older commit
    # are consecutive tree states when a newer commit points to it.
    for newer_commit, newer_parent in history:
        older = next((row for row in history if row[0] == newer_parent), None)
        if older is None:
            continue
        parent1, parent2 = older[1], older[0]
        if _index_is_usable(_state_cache_dir(repository, parent1)) or _index_is_usable(_state_cache_dir(repository, parent2)):
            continue
        shared_dir = shared_embedding_cache_dir(repository)
        if shared_dir.exists() and any(shared_dir.glob("*.npy")):
            continue
        return parent1, parent2
    raise RuntimeError(f"Could not find two adjacent, uncached parent states for {repository}")


def run_embedding_reuse_benchmark(repository="Pinia", parents=None):
    parents = parents or choose_close_parent_states(repository)
    if len(parents) != 2 or parents[0] == parents[1]:
        raise ValueError("Reuse benchmark requires two different parent SHAs")
    shared_dir = shared_embedding_cache_dir(repository)
    if shared_dir.exists() and any(shared_dir.glob("*.npy")):
        raise RuntimeError("Shared embedding cache is not cold; choose a fresh repository cache profile")
    counters = Counter({"embedding_api_calls": 0, "embedding_inputs_sent": 0, "embeddings_generated": 0,
                        "index_cache_hits": 0, "index_cache_misses": 0, "indexing_seconds": 0.0})
    original_post, counted_post = _call_counter_wrapper(counters)
    utils.requests.post = counted_post
    states = []
    try:
        for number, parent in enumerate(parents, 1):
            print(f"[REUSE BENCHMARK] Parent {number}/2 {repository} {parent}", flush=True)
            worktree = prepare_parent_worktree(repository, parent)
            state = ensure_historical_index(repository, parent, worktree, counters)
            states.append(state)
    finally:
        utils.requests.post = original_post
    total_chunks = sum(row["total_chunks"] for row in states)
    reused = sum(row["embedding_cache_hits"] for row in states)
    new_embeddings = sum(row["embeddings_requested"] for row in states)
    shared_entries = sum(1 for item in shared_dir.glob("*.npy") if item.is_file())
    result = {"repository": repository, "parents": states,
              "overall": {"shared_embedding_cache_entries": shared_entries,
                          "total_reused_embeddings": reused, "total_new_embeddings": new_embeddings,
                          "total_chunks": total_chunks,
                          "global_reuse_percentage": 100.0 * reused / max(1, total_chunks),
                          "embedding_api_calls": counters["embedding_api_calls"],
                          "embedding_inputs_sent": counters["embedding_inputs_sent"],
                          "embeddings_generated": counters["embeddings_generated"]}}
    print("\nParent-state embedding reuse:", flush=True)
    for i, state in enumerate(states, 1):
        print(f"Parent {i}: files={state['indexed_files']} chunks={state['total_chunks']} hits={state['embedding_cache_hits']} misses={state['embedding_cache_misses']} reuse={state['embedding_reuse_percentage']:.1f}% API calls={state['embedding_api_calls']} requested={state['embeddings_requested']} indexing={state['indexing_seconds']:.2f}s", flush=True)
    print("Overall: " + json.dumps(result["overall"], sort_keys=True), flush=True)
    output = BENCHMARK_DIR / "generated_multi_repo" / "pinia_embedding_reuse_benchmark.json"
    _persist(result, output)
    print(f"[DONE] saved {output}", flush=True)
    return result


def _index_is_usable(path):
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        import numpy as np
        vectors = np.load(path / "embeddings.npy", mmap_mode="r")
        return (metadata.get("chunker_version") == "1"
                and metadata.get("embedding_model") == utils.CONFIG["embedding_model"]
                and len(chunks) == len(vectors) == int(metadata.get("chunk_count", -1)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def ensure_historical_index(repository, parent, worktree, counters):
    cache = _state_cache_dir(repository, parent)
    hit = _index_is_usable(cache)
    if hit:
        counters["index_cache_hits"] += 1
    else:
        counters["index_cache_misses"] += 1
        cache.mkdir(parents=True, exist_ok=True)
        old_enabled = utils.CONFIG.get("semantic_chunking", {}).get("enabled", False)
        utils.CONFIG.setdefault("semantic_chunking", {})["enabled"] = False
        old_utils_cache = utils.cache_dir
        old_update_cache = update_index_module.cache_dir
        old_update_embed = update_index_module.embed
        utils.cache_dir = lambda _root: cache
        update_index_module.cache_dir = lambda _root: cache
        chunker_version = update_index_module.effective_chunker_version(utils.CONFIG)
        shared = SharedEmbeddingCache(shared_embedding_cache_dir(repository), _repository_identity(repository),
            model=utils.CONFIG["embedding_model"], chunker_version=chunker_version,
            dimension=utils.CONFIG.get("embedding_dimension"))
        api_calls_before = counters.get("embedding_api_calls", 0)
        state_api_calls = Counter()
        original_post = utils.requests.post
        def count_state_post(url, *args, **kwargs):
            if str(url).rstrip("/").endswith("/embeddings"):
                state_api_calls["embedding_api_calls"] += 1
            return original_post(url, *args, **kwargs)
        utils.requests.post = count_state_post
        update_index_module.embed = shared.embed_chunks
        started = time.perf_counter()
        try:
            update_index_module.update(root=worktree)
        finally:
            elapsed = time.perf_counter() - started
            counters["indexing_seconds"] += elapsed
            counters["last_index_seconds"] = elapsed
            utils.cache_dir = old_utils_cache
            update_index_module.cache_dir = old_update_cache
            update_index_module.embed = old_update_embed
            utils.requests.post = original_post
            utils.CONFIG["semantic_chunking"]["enabled"] = old_enabled
        if not _index_is_usable(cache):
            raise RuntimeError(f"Historical index was not created correctly for {repository}@{parent}")
    meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    chunks = json.loads((cache / "chunks.json").read_text(encoding="utf-8"))
    import numpy as np
    vectors = np.load(cache / "embeddings.npy", mmap_mode="r")
    if hit:
        try:
            shared_stats = json.loads((cache / "shared_embedding_stats.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            shared_stats = {}
        if shared_stats.get("reuse_stats_schema_version") != REUSE_STATS_SCHEMA_VERSION:
            shared_stats = {}
    else:
        shared_stats = dict(shared.stats)
        shared_stats.update({"total_chunks": len(chunks), "embedding_api_calls": dict(state_api_calls).get("embedding_api_calls", 0),
                             "indexing_seconds": counters.get("last_index_seconds", 0.0),
                             "reuse_stats_schema_version": REUSE_STATS_SCHEMA_VERSION})
        cache_stats_path = cache / "shared_embedding_stats.json"
        cache_stats_path.write_text(json.dumps(shared_stats, indent=2) + "\n", encoding="utf-8")
    total = int(shared_stats.get("total_chunks", len(chunks)))
    cache_hits = int(shared_stats.get("embedding_cache_hits", 0))
    cache_misses = int(shared_stats.get("embedding_cache_misses", 0))
    return {"repository": repository, "parent": parent, "indexed_files": len(meta.get("files", {})),
            "chunk_count": len(chunks), "embedding_count": len(vectors), "cache_hit": hit,
            "total_chunks": total, "embedding_cache_hits": cache_hits, "embedding_cache_misses": cache_misses,
            "intra_index_dedup_hits": int(shared_stats.get("intra_index_dedup_hits", 0)),
            "embeddings_requested": int(shared_stats.get("embeddings_requested", 0)),
            "embedding_reuse_percentage": 100.0 * cache_hits / max(1, total),
            "embedding_api_calls": int(shared_stats.get("embedding_api_calls", 0)),
            "indexing_seconds": float(shared_stats.get("indexing_seconds", 0.0))}


def query_vector(query, counters, repository=None):
    digest = hashlib.sha256((utils.CONFIG["embedding_model"] + "\0" + query).encode("utf-8")).hexdigest()
    if repository is None:
        path = WORKSPACE / "query_embeddings" / f"{digest}.npy"
    else:
        namespace = "".join(c.lower() if c.isalnum() else "_" for c in repository).strip("_")
        path = WORKSPACE / "query_embeddings" / namespace / f"{digest}.npy"
    import numpy as np
    if path.exists():
        counters["query_embedding_cache_hits"] += 1
        return np.load(path)
    vec = utils.embed([query])[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)
    return vec


def _metric_rows(rows):
    metrics = ("hit@1", "hit@3", "hit@5", "reciprocal_rank", "recall@5")
    result = {key: sum(row[key] for row in rows) / len(rows) if rows else 0.0 for key in metrics}
    tokens = [row["context_tokens"] for row in rows]
    result["average_context_tokens"] = sum(tokens) / len(tokens) if tokens else 0.0
    result["median_context_tokens"] = float(statistics.median(tokens)) if tokens else 0.0
    result["max_context_tokens"] = max(tokens, default=0)
    return result


def summarize_results(rows, modes=None):
    modes = modes or (ABLATION_MODES if any(r.get("mode") == "Semantic Only" for r in rows) else MODES)
    summaries = {mode: _metric_rows([r for r in rows if r["mode"] == mode]) for mode, _ in modes}
    repos = sorted({r["repository"] for r in rows})
    macro = {}
    per_repository = {}
    for mode, _ in modes:
        per_repository[mode] = {repo: _metric_rows([r for r in rows if r["mode"] == mode and r["repository"] == repo]) for repo in repos}
        macro[mode] = {metric: sum(per_repository[mode][repo][metric] for repo in repos) / len(repos)
                       for metric in per_repository[mode][repos[0]]} if repos else {}
    metrics = ("hit@1", "hit@3", "hit@5", "reciprocal_rank", "recall@5")
    delta = {}
    for aggregate_name, aggregate in (("micro", summaries), ("macro_repository_average", macro)):
        full_label = "Current Full Smart Context" if "Current Full Smart Context" in aggregate else "Full Smart Context"
        no_git_label = "Full minus Git Recency" if "Full minus Git Recency" in aggregate else "No Git Recency"
        full, no_git = aggregate.get(full_label, {}), aggregate.get(no_git_label, {})
        delta[aggregate_name] = {metric: {"absolute": full.get(metric, 0.0) - no_git.get(metric, 0.0),
                                           "percentage_points": 100 * (full.get(metric, 0.0) - no_git.get(metric, 0.0))}
                                 for metric in metrics}
    return {"micro": summaries, "macro_repository_average": macro, "per_repository": per_repository,
            "full_minus_no_git_recency": delta, "modes": [label for label, _ in modes]}


def _call_counter_wrapper(counters):
    original_post = utils.requests.post
    def counted_post(url, *args, **kwargs):
        if str(url).rstrip("/").endswith("/embeddings"):
            counters["embedding_api_calls"] += 1
            payload = kwargs.get("json", {})
            counters["embedding_inputs_sent"] += len(payload.get("input", []))
        response = original_post(url, *args, **kwargs)
        if str(url).rstrip("/").endswith("/embeddings") and response.ok:
            try:
                counters["embeddings_generated"] += len(response.json().get("data", []))
            except (ValueError, AttributeError):
                pass
        return response
    return original_post, counted_post


def evaluate_records(records, *, stage, repository_query_cache=False):
    counters = Counter()
    counters.update({"embedding_api_calls": 0, "embedding_inputs_sent": 0, "embeddings_generated": 0,
                     "query_embedding_cache_hits": 0, "index_cache_hits": 0, "index_cache_misses": 0,
                     "indexing_seconds": 0.0, "retrieval_seconds": 0.0})
    rows, index_records = [], []
    index_map = {}
    total_started = time.perf_counter()
    original_post, counted_post = _call_counter_wrapper(counters)
    utils.requests.post = counted_post
    try:
        per_repo_total = Counter(record["repository"] for record in records)
        per_repo_progress = Counter()
        for number, record in enumerate(records, 1):
            repository, parent = record["repository"], record["parent"]
            per_repo_progress[repository] += 1
            key = (repository, parent)
            print(f"[{repository}] {per_repo_progress[repository]}/{per_repo_total[repository]} {record['commit'][:10]} parent={parent[:10]}", flush=True)
            if key not in index_map:
                worktree = prepare_parent_worktree(repository, parent)
                state = ensure_historical_index(repository, parent, worktree, counters)
                index_map[key] = (worktree, state)
                index_records.append(state)
            worktree, state = index_map[key]
            qvec = query_vector(record["query"], counters,
                                repository=repository if repository_query_cache else None)
            modes = modes_for_stage(stage)
            for label, mode in modes:
                started = time.perf_counter()
                previous_cache_dir = utils.cache_dir
                utils.cache_dir = lambda _root, state_cache=_state_cache_dir(repository, parent): state_cache
                try:
                    if isinstance(mode, dict):
                        result = retrieve_module.retrieve(record["query"], root=worktree, query_embedding=qvec,
                            ranking_mode="custom", ranking_profile=mode, reference_time=record["timestamp"])
                    else:
                        result = retrieve_module.retrieve(record["query"], root=worktree, query_embedding=qvec,
                            ranking_mode=mode, reference_time=record["timestamp"])
                finally:
                    utils.cache_dir = previous_cache_dir
                elapsed = time.perf_counter() - started
                counters["retrieval_seconds"] += elapsed
                ranked_files = deduplicate_ranked_files(result.get("selected", []))
                scored = score_ranked_files(ranked_files, record["targets"])
                rows.append({"repository": repository, "query": record["query"], "commit": record["commit"],
                    "parent": parent, "targets": record["targets"], "ranked_unique_files": ranked_files,
                    "mode": label, **scored, "context_tokens": result.get("estimated_tokens", 0),
                    "retrieval_seconds": elapsed, "ranking_scores": [
                        {"file": item["chunk"].get("file"), "score": item.get("score"),
                         "base_score": item.get("base_score"), "modifier_total": item.get("modifier_total"),
                         "tiny_adjustment": item.get("tiny_adjustment"),
                         "path_role_adjustment": item.get("path_role_adjustment"),
                         "source_type_adjustment": item.get("source_type_adjustment")}
                        for item in result.get("selected", [])]})
        counters["elapsed_seconds"] = time.perf_counter() - total_started
    finally:
        utils.requests.post = original_post
    counters["cache_hit_rate"] = counters["index_cache_hits"] / max(1, counters["index_cache_hits"] + counters["index_cache_misses"])
    counters["total_reused_embeddings"] = sum(state.get("embedding_cache_hits", 0) for state in index_records)
    counters["total_new_embeddings"] = sum(state.get("embeddings_requested", 0) for state in index_records)
    counters["total_chunks"] = sum(state.get("total_chunks", state.get("chunk_count", 0)) for state in index_records)
    counters["global_reuse_percentage"] = 100.0 * counters["total_reused_embeddings"] / max(1, counters["total_chunks"])
    counters["shared_embedding_cache_entries"] = shared_embedding_entry_count(per_repo_total)
    modes = modes_for_stage(stage)
    return {"stage": stage, "query_count": len(records), "summary": summarize_results(rows, modes),
            "per_query": rows, "index_states": index_records, "counters": dict(counters)}


def group_repository_records(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["repository"]].append(record)
    for repository in groups:
        groups[repository].sort(key=record_chronology_key)
    return {repository: groups[repository] for repository in REPOSITORIES if repository in groups}


def repository_worker_count(requested, repository_count, cpu_count=None):
    if requested is None:
        return 1
    if int(requested) < 1:
        raise ValueError("--parallel-repositories must be at least 1")
    available = os.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(int(requested), int(repository_count), int(available or 1)))


def _evaluate_repository_worker(repository, records, stage):
    """Process one repository serially in a child process; never writes final outputs."""
    chronological = sorted(records, key=record_chronology_key)
    return evaluate_records(chronological, stage=stage, repository_query_cache=True)


def combine_repository_results(results, *, wall_seconds, requested_workers, actual_workers):
    rows = [row for result in results for row in result["per_query"]]
    states = [state for result in results for state in result["index_states"]]
    aggregate = Counter()
    for result in results:
        for key, value in result["counters"].items():
            if isinstance(value, (int, float)):
                aggregate[key] += value
    aggregate["cache_hit_rate"] = aggregate["index_cache_hits"] / max(
        1, aggregate["index_cache_hits"] + aggregate["index_cache_misses"])
    aggregate["elapsed_seconds"] = wall_seconds
    aggregate["parallel_repositories_requested"] = requested_workers
    aggregate["parallel_worker_count"] = actual_workers
    aggregate["total_reused_embeddings"] = sum(state.get("embedding_cache_hits", 0) for state in states)
    aggregate["total_new_embeddings"] = sum(state.get("embeddings_requested", 0) for state in states)
    aggregate["total_chunks"] = sum(state.get("total_chunks", state.get("chunk_count", 0)) for state in states)
    aggregate["global_reuse_percentage"] = 100.0 * aggregate["total_reused_embeddings"] / max(1, aggregate["total_chunks"])
    aggregate["shared_embedding_cache_entries"] = shared_embedding_entry_count(
        {row["repository"] for row in rows})
    stage = results[0]["stage"] if results else "pilot"
    return {"stage": stage, "query_count": len(rows),
            "summary": summarize_results(rows, modes_for_stage(stage)), "per_query": rows, "index_states": states,
            "counters": dict(aggregate)}


def evaluate_repositories_parallel(records, requested_workers, *, stage="pilot", executor_cls=ProcessPoolExecutor,
                                   worker_fn=_evaluate_repository_worker, cpu_count=None):
    groups = group_repository_records(records)
    worker_count = repository_worker_count(requested_workers, len(groups), cpu_count=cpu_count)
    started = time.perf_counter()
    tasks = list(groups.items())
    if worker_count == 1:
        results = [worker_fn(repository, group, stage) for repository, group in tasks]
    else:
        with executor_cls(max_workers=worker_count) as executor:
            futures = [executor.submit(worker_fn, repository, group, stage) for repository, group in tasks]
            # Consume results in repository order for deterministic merged output.
            results = [future.result() for future in futures]
    return combine_repository_results(results, wall_seconds=time.perf_counter() - started,
        requested_workers=requested_workers, actual_workers=worker_count)


def failure_analysis(rows):
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["repository"], row["commit"])][row["mode"]] = row
    failures = {"full_misses_top5": [], "semantic_succeeds_full_fails": [],
                "no_git_succeeds_full_fails": [], "full_succeeds_no_git_fails": []}
    for modes in by_key.values():
        semantic = modes.get("Semantic-only")
        no_git = modes.get("No Git Recency")
        full = modes.get("Full Smart Context")
        if not full:
            continue
        base = {key: full[key] for key in ("repository", "query", "commit", "parent", "targets")}
        if not full["hit@5"]:
            failures["full_misses_top5"].append({**base, "full": full["ranking_scores"]})
        if semantic and semantic["hit@5"] and not full["hit@5"]:
            failures["semantic_succeeds_full_fails"].append({**base, "semantic": semantic["ranking_scores"], "full": full["ranking_scores"]})
        if no_git and no_git["hit@5"] and not full["hit@5"]:
            failures["no_git_succeeds_full_fails"].append({**base, "no_git": no_git["ranking_scores"], "full": full["ranking_scores"]})
        if no_git and full["hit@5"] and not no_git["hit@5"]:
            failures["full_succeeds_no_git_fails"].append({**base, "no_git": no_git["ranking_scores"], "full": full["ranking_scores"]})
    return failures


def ranking_ablation_analysis(rows, modes=None, *, include_legacy_repository_focus=True):
    """Summarize each signal against Semantic Only without fitting any weights."""
    modes = modes or ABLATION_MODES
    summary = summarize_results(rows, modes)
    baseline_name = "Semantic Only"
    reported = ("hit@1", "hit@3", "hit@5", "reciprocal_rank", "recall@5", "average_context_tokens")
    deltas = {}
    signal_impact = []
    for label, _ in modes:
        deltas[label] = {}
        for scope, aggregate in (("micro", summary["micro"]),
                                 ("macro_repository_average", summary["macro_repository_average"])):
            deltas[label][scope] = {metric: aggregate[label][metric] - aggregate[baseline_name][metric]
                                    for metric in reported}
        deltas[label]["per_repository"] = {
            repo: {metric: summary["per_repository"][label][repo][metric] -
                         summary["per_repository"][baseline_name][repo][metric] for metric in reported}
            for repo in summary["per_repository"][label]
        }
        if label != baseline_name:
            delta = deltas[label]["micro"]
            signal_impact.append({"mode": label, **{key: delta[key] for key in reported}})

    by_query = defaultdict(dict)
    for row in rows:
        by_query[(row["repository"], row["commit"])][row["mode"]] = row
    changes = {}
    for label, _ in modes:
        if label == baseline_name:
            continue
        groups = {"improved_first_relevant_rank": [], "worsened_first_relevant_rank": [],
                  "relevant_file_entered_top5": [], "relevant_file_left_top5": []}
        for modes in by_query.values():
            baseline, compared = modes.get(baseline_name), modes.get(label)
            if baseline is None or compared is None:
                continue
            base_rank, new_rank = baseline["first_relevant_rank"], compared["first_relevant_rank"]
            data = {key: compared[key] for key in ("repository", "query", "commit", "parent", "targets")}
            data.update({"semantic_only_rank": base_rank, "mode_rank": new_rank,
                         "semantic_only_files": baseline["ranked_unique_files"],
                         "mode_files": compared["ranked_unique_files"],
                         "semantic_only_scores": baseline["ranking_scores"],
                         "mode_scores": compared["ranking_scores"]})
            if new_rank is not None and (base_rank is None or new_rank < base_rank):
                groups["improved_first_relevant_rank"].append(data)
            if base_rank is not None and (new_rank is None or new_rank > base_rank):
                groups["worsened_first_relevant_rank"].append(data)
            if not baseline["hit@5"] and compared["hit@5"]:
                groups["relevant_file_entered_top5"].append(data)
            if baseline["hit@5"] and not compared["hit@5"]:
                groups["relevant_file_left_top5"].append(data)
        changes[label] = groups

    analysis = {"summary": summary, "delta_vs_semantic_only": deltas,
                "signal_impact_micro": signal_impact, "query_signal_changes": changes}
    if include_legacy_repository_focus:
        focused = {}
        for repository in ("Express", "Pinia"):
            focused[repository] = {}
            for mode, buckets in changes.items():
                focused[repository][mode] = {
                    key: [row for row in values if row["repository"] == repository]
                    for key, values in buckets.items()
                }
        analysis["express_pinia_query_signal_changes"] = focused
    return analysis


def paired_mrr_bootstrap(rows, *, seed=20260930, replicates=10_000):
    """Percentile CI for paired V2.1-minus-semantic query-level reciprocal rank."""
    by_query = defaultdict(dict)
    for row in rows:
        by_query[(row["repository"], row["commit"])][row["mode"]] = row["reciprocal_rank"]
    baseline, hybrid = "Semantic Only", "Semantic + V2.1 Reranking"
    if not by_query or any(set(pair) != {baseline, hybrid} for pair in by_query.values()):
        raise ValueError("Paired MRR bootstrap requires both frozen modes for every query")
    paired = np.asarray([pair[hybrid] - pair[baseline] for pair in by_query.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(int(replicates), len(paired)))
    means = paired[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return {"comparison": f"{hybrid} minus {baseline}", "absolute_delta": float(paired.mean()),
            "paired_bootstrap_95_percent_confidence_interval": [float(low), float(high)],
            "bootstrap_seed": int(seed), "bootstrap_replicates": int(replicates),
            "resampling_unit": "paired query example"}


def transition_counts(analysis):
    counts = {}
    for mode, buckets in analysis.get("query_signal_changes", {}).items():
        mode_counts = {key: len(rows) for key, rows in buckets.items()}
        mode_counts["net_rank_change_count"] = (
            mode_counts["improved_first_relevant_rank"] - mode_counts["worsened_first_relevant_rank"])
        mode_counts["net_top5_change_count"] = (
            mode_counts["relevant_file_entered_top5"] - mode_counts["relevant_file_left_top5"])
        counts[mode] = mode_counts
    return counts


def compact_ablation_result(result):
    """Drop duplicated query detail; the dedicated per-query artifact keeps it."""
    compact = dict(result)
    compact.pop("per_query", None)
    analysis = dict(compact.get("ablation_analysis", {}))
    analysis.pop("summary", None)
    analysis.pop("express_pinia_query_signal_changes", None)
    if "query_signal_changes" in analysis:
        analysis["transition_counts"] = transition_counts(analysis)
        analysis.pop("query_signal_changes", None)
    compact["ablation_analysis"] = analysis
    return compact


def shared_embedding_entry_count(repositories):
    root = WORKSPACE / "shared_embeddings"
    total = 0
    for repository in repositories:
        namespace = "".join(char.lower() if char.isalnum() else "_" for char in repository).strip("_")
        folder = root / namespace
        if folder.is_dir():
            total += sum(1 for item in folder.glob("*.npy") if item.is_file())
    return total


def _persist(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def persist_pilot_outputs(result, stage, output_prefix=None):
    if stage == "ablation":
        prefix = output_prefix or "ranking_ablation_validation"
        output_name, query_name = f"{prefix}_results.json", f"{prefix}_per_query.json"
        _persist(compact_ablation_result(result), OUTPUT_DIR / output_name)
        _persist(result["per_query"], OUTPUT_DIR / query_name)
        return
    if stage == "heldout":
        output_name = output_prefix or "final_heldout_compact_results.json"
        _persist(result, OUTPUT_DIR / output_name)
        return
    else:
        output_name = "pilot_smoke_results.json" if stage == "smoke" else "pilot_results.json"
        query_name = "pilot_smoke_per_query.json" if stage == "smoke" else "pilot_per_query.json"
    _persist(result, OUTPUT_DIR / output_name)
    _persist(result["per_query"], OUTPUT_DIR / query_name)


def print_summary(result):
    modes = modes_for_stage(result["stage"])
    print(f"\n{result['stage']} metrics (micro):", flush=True)
    print(f"{'Mode':<24}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}{'Recall@5':>10}{'Avg Tokens':>12}", flush=True)
    for label, _ in modes:
        m = result["summary"]["micro"][label]
        print(f"{label:<24}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}{m['hit@5']:>8.3f}{m['reciprocal_rank']:>8.3f}{m['recall@5']:>10.3f}{m['average_context_tokens']:>12.1f}", flush=True)
    print("\nMacro repository average:", flush=True)
    for label, _ in modes:
        m = result["summary"]["macro_repository_average"][label]
        print(f"{label:<24}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}{m['hit@5']:>8.3f}{m['reciprocal_rank']:>8.3f}{m['recall@5']:>10.3f}{m['average_context_tokens']:>12.1f}", flush=True)
    print("\nPer-repository metrics:", flush=True)
    for repo in sorted(result["summary"]["per_repository"].get(modes[0][0], {})):
        print(f"  {repo}", flush=True)
        for label, _ in modes:
            m = result["summary"]["per_repository"][label][repo]
            print(f"    {label:<22} Hit@1={m['hit@1']:.3f} Hit@3={m['hit@3']:.3f} Hit@5={m['hit@5']:.3f} MRR={m['reciprocal_rank']:.3f} Recall@5={m['recall@5']:.3f} AvgTokens={m['average_context_tokens']:.1f}", flush=True)
    if result["stage"] in {"ablation", "heldout"} and "ablation_analysis" in result:
        baseline = result["summary"]["micro"]["Semantic Only"]
        print("\nMicro deltas versus Semantic Only (absolute; percentage points):", flush=True)
        for label, _ in modes[1:]:
            current = result["summary"]["micro"][label]
            diffs = [current[k] - baseline[k] for k in ("hit@1", "hit@3", "hit@5", "reciprocal_rank", "recall@5")]
            print(f"  {label:<32} " + " ".join(f"{d:+.3f}/{100*d:+.1f}pp" for d in diffs), flush=True)
        print("\nSignal-impact change counts versus Semantic Only:", flush=True)
        if "query_signal_changes" in result["ablation_analysis"]:
            for label, groups in result["ablation_analysis"]["query_signal_changes"].items():
                print(f"  {label:<32} improved={len(groups['improved_first_relevant_rank'])} worsened={len(groups['worsened_first_relevant_rank'])} entered_top5={len(groups['relevant_file_entered_top5'])} left_top5={len(groups['relevant_file_left_top5'])}", flush=True)
        if "paired_mrr_bootstrap" in result:
            ci = result["paired_mrr_bootstrap"]["paired_bootstrap_95_percent_confidence_interval"]
            print(f"\nPaired MRR delta={result['paired_mrr_bootstrap']['absolute_delta']:+.4f}; 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)
    else:
        print("\nFull - No Git Recency (absolute; percentage points):", flush=True)
        for metric, delta in result["summary"]["full_minus_no_git_recency"]["micro"].items():
            print(f"  {metric}: {delta['absolute']:+.3f}; {delta['percentage_points']:+.1f} pp", flush=True)
    print("Counters: " + json.dumps(result["counters"], sort_keys=True), flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot", "reuse", "ablation", "heldout"), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--repositories", nargs="+", choices=tuple(REPOSITORIES))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--parent1")
    parser.add_argument("--parent2")
    parser.add_argument("--parallel-repositories", type=int,
                        help="Run repository groups in separate processes (one serial worker per repository)")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.stage == "reuse":
        parents = (args.parent1, args.parent2) if args.parent1 and args.parent2 else None
        run_embedding_reuse_benchmark("Pinia", parents=parents)
        return 0
    if args.stage == "smoke":
        dataset = args.dataset or DATASET
        manifest = write_manifest(sample_pilot_records(read_jsonl(dataset)), args.manifest or OUTPUT_DIR / "pilot_test_63.jsonl")
        records = smoke_records(read_jsonl(manifest))
    elif args.stage == "pilot":
        records = read_jsonl(args.manifest or OUTPUT_DIR / "pilot_test_63.jsonl")
        if len(records) != 63:
            raise SystemExit(f"Pilot manifest has {len(records)} records, expected 63")
    elif args.stage == "heldout":
        manifest_path = args.manifest or OUTPUT_DIR / "final_heldout_compact_manifest.json"
        dataset = args.dataset or DATASET
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            # Persist the predeclared uniform sample before any index/cache access.
            manifest = write_final_heldout_manifest(read_jsonl(dataset), manifest_path,
                                                    seed=20260930, dataset_path=dataset)
        if manifest.get("seed") != 20260930:
            raise SystemExit("Held-out manifest seed must be 20260930")
        records = validate_final_heldout_manifest(manifest)
    else:
        dataset = args.dataset or (OUTPUT_DIR / "benchmark_validation.jsonl")
        records = read_jsonl(dataset)
        records = [row for row in records if row.get("category", "clean") == "clean" and
                   row.get("benchmark_included", True) and row.get("split", "validation") == "validation"]
        if args.repositories:
            records = [row for row in records if row.get("repository") in args.repositories]
        if not records:
            raise SystemExit(f"No clean validation records found in {dataset}")
    if args.stage == "heldout" and args.repositories and set(args.repositories) != {"Express", "Flask"}:
        raise SystemExit("Final held-out evaluation is fixed to Express and Flask")
    print(f"[START] {args.stage}: {len(records)} commit-parent queries; one index per unique repository + parent", flush=True)
    if args.stage in {"pilot", "ablation", "heldout"} and args.parallel_repositories is not None:
        result = evaluate_repositories_parallel(records, args.parallel_repositories, stage=args.stage)
    else:
        # Preserve the original single-process behavior when the flag is omitted.
        result = evaluate_records(records, stage=args.stage)
    print_summary(result)
    if args.stage in {"ablation", "heldout"}:
        result["ablation_analysis"] = ranking_ablation_analysis(
            result["per_query"], modes_for_stage(args.stage),
            include_legacy_repository_focus=(args.stage == "ablation" and "Pinia" in {r["repository"] for r in records}))
        result["ablation_analysis"].pop("summary", None)
        if args.stage == "heldout":
            result["paired_mrr_bootstrap"] = paired_mrr_bootstrap(result["per_query"])
            result["metadata"] = {"repositories": ["Express", "Flask"], "excluded_repositories": ["Pinia"],
                "validation_examples_used": 0, "diagnostic_pilot_examples_used": 0,
                "validation_or_pilot_examples_excluded": True, "sample_seed": 20260930,
                "manifest": str(args.manifest or OUTPUT_DIR / "final_heldout_compact_manifest.json"),
                "evaluation_modes": [label for label, _ in HELDOUT_MODES], "ranking_architecture_modified": False}
        else:
            counts = Counter(row["repository"] for row in records)
            result["metadata"] = {"repositories": sorted(counts), "excluded_repositories": ["Pinia"],
                "validation_examples": len(records), "repository_counts": dict(counts),
                "ranking_row_count": len(result["per_query"]), "split_reused": "validation"}
    else:
        result["failure_analysis"] = failure_analysis(result["per_query"])
    persist_pilot_outputs(result, args.stage, args.output_prefix)
    output_name = ({"smoke": "pilot_smoke_results.json", "pilot": "pilot_results.json",
                    "ablation": "ranking_ablation_validation_results.json"}.get(args.stage, ""))
    print(f"[DONE] saved {output_name}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"historical pilot: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
