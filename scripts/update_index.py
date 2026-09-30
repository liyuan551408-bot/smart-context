"""Incrementally update cached chunks and embeddings for a repository."""
import json
import sys
from pathlib import Path

import numpy as np

from semantic_chunker import adaptive_chunk_file
from utils import CONFIG, atomic_json, cache_dir, content_hash, embed, git, git_files, read_text, repo_root


def effective_chunker_version(config=None):
    """Return a mode-specific version so V1/V2 switches always trigger rechunking."""
    config = config or CONFIG
    settings = config.get("semantic_chunking", {})
    if not settings.get("enabled", False):
        return "1"
    return "2:{}:{}:{}".format(settings.get("config_version", "1"), settings.get("provider", "zhipu"), settings.get("model", "glm-4-flash"))


def index_requires_rechunk(old_meta, config=None):
    """True only when stored chunk boundaries were produced by another mode/version."""
    return old_meta.get("chunker_version", "1") != effective_chunker_version(config)


def source_paths(root):
    exts = set(CONFIG["extensions"])
    ignores = set(CONFIG["ignore_dirs"])
    lockfiles = set(CONFIG.get("ignore_files", []))
    tracked = git_files(root)
    candidates = (root / p for p in tracked) if tracked else root.rglob("*")
    result = []
    max_bytes = int(CONFIG.get("max_file_bytes", 1_000_000))
    for path in candidates:
        try:
            rel = path.relative_to(root).as_posix()
            if not path.is_file() or path.suffix.lower() not in exts or path.name in lockfiles or path.stat().st_size > max_bytes:
                continue
            if any(part in ignores for part in Path(rel).parts):
                continue
            if rel.startswith("cache/"):
                continue
            result.append((rel, path))
        except (OSError, ValueError):
            continue
    return sorted(result)


def update(root=None, force=False):
    root = repo_root(root)
    cache = cache_dir(root)
    old_chunks, old_files, old_meta = [], {}, {}
    try:
        old_chunks = json.loads((cache / "chunks.json").read_text(encoding="utf-8"))
        old_meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        old_files = old_meta.get("files", {})
        old_vecs = np.load(cache / "embeddings.npy")
        if len(old_vecs) != len(old_chunks):
            raise ValueError("index lengths differ")
        compatible = (old_meta.get("embedding_provider") == CONFIG["embedding_provider"] and old_meta.get("embedding_model") == CONFIG["embedding_model"] and int(old_meta.get("embedding_dimension", -1)) == (old_vecs.shape[1] if old_vecs.ndim == 2 else -2))
        if not compatible and not force:
            raise RuntimeError("Existing index uses a different embedding provider, model, or dimension. Run scripts/build_index.py to explicitly rebuild it.")
    except (OSError, ValueError, json.JSONDecodeError):
        old_chunks, old_files, old_vecs = [], {}, np.empty((0, 0), dtype=np.float32)
        old_meta = {}
        force = True
    # Inspect Git changes as an inexpensive signal; content hashes remain authoritative.
    git_changed = set()
    for args in (["diff", "--name-only", "--diff-filter=ACDMRTUXB", "HEAD"], ["diff", "--cached", "--name-only"], ["status", "--porcelain"]):
        for line in git(root, args).splitlines():
            val = line[3:] if line.startswith(" ") or (len(line) > 2 and line[2] == " ") else line
            if " -> " in val:
                val = val.split(" -> ", 1)[1]
            git_changed.add(val.replace("\\", "/").strip())

    entries = source_paths(root)
    chunker_version = effective_chunker_version(CONFIG)
    rechunk_all = index_requires_rechunk(old_meta, CONFIG)
    current = {}
    changed = set()
    for rel, path in entries:
        body = read_text(path)
        digest = content_hash(body)
        current[rel] = digest
        if force or rechunk_all or old_files.get(rel) != digest or rel in git_changed:
            changed.add(rel)
    deleted = set(old_files) - set(current)
    keep = [c for c in old_chunks if c["file"] not in changed | deleted]
    new_chunks = []
    chunk_metrics = {"local_chunks": 0, "glm_refinement_candidates": 0, "glm_plans_reused": 0, "glm_api_calls": 0, "glm_refined_chunks": 0, "small_chunks_merged": 0}
    for rel, path in entries:
        if rel in changed:
            chunks, metrics = adaptive_chunk_file(rel, read_text(path), root, CONFIG)
            new_chunks.extend(chunks)
            for key in chunk_metrics:
                chunk_metrics[key] += metrics.get(key, 0)
    all_chunks = keep + new_chunks
    # Build a content hash -> vector pool so unchanged chunks survive file edits and moves.
    reusable = {}
    if not force and old_vecs.ndim == 2:
        for i, chunk in enumerate(old_chunks):
            if i < len(old_vecs) and old_vecs.shape[1] == int(old_meta.get("embedding_dimension", old_vecs.shape[1])):
                reusable.setdefault(chunk["content_hash"], old_vecs[i])
    fresh = [c for c in all_chunks if force or c["content_hash"] not in reusable]
    vec_by_hash = dict(reusable)
    if fresh:
        vectors_new = embed([c["content"] for c in fresh])
        dim = vectors_new.shape[1]
        vec_by_hash.update({c["content_hash"]: v for c, v in zip(fresh, vectors_new)})
    elif old_vecs.ndim == 2 and old_vecs.size:
        dim = old_vecs.shape[1]
    else:
        dim = 0
    rows = []
    if all_chunks:
        embeddings = np.stack([vec_by_hash[c["content_hash"]] for c in all_chunks]).astype(np.float32)
    else:
        embeddings = np.empty((0, dim), dtype=np.float32)
    atomic_json(cache / "chunks.json", all_chunks)
    np.save(cache / "embeddings.npy", embeddings)
    atomic_json(cache / "metadata.json", {"root": str(root), "embedding_provider": CONFIG["embedding_provider"], "embedding_model": CONFIG["embedding_model"], "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else int(old_meta.get("embedding_dimension", 0)), "files": current, "chunk_count": len(all_chunks), "chunker_version": chunker_version})
    reused_count = max(0, len(all_chunks) - len(fresh))
    print(f"Indexed {len(changed)} changed and {len(deleted)} deleted files; {len(all_chunks)} chunks across {len(current)} files ({reused_count} reused, {len(fresh)} embedded).")
    print(f"Embedding dimension: {embeddings.shape[1] if embeddings.ndim == 2 else 0}")
    if CONFIG.get("semantic_chunking", {}).get("enabled", False):
        print(f"Local chunks: {chunk_metrics['local_chunks']}")
        print(f"GLM refinement candidates: {chunk_metrics['glm_refinement_candidates']}")
        print(f"GLM plans reused: {chunk_metrics['glm_plans_reused']}")
        print(f"GLM API calls: {chunk_metrics['glm_api_calls']}")
        print(f"GLM-refined chunks: {chunk_metrics['glm_refined_chunks']}")
        print(f"Small chunks merged: {chunk_metrics['small_chunks_merged']}")
    return root


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--root")
        args = parser.parse_args()
        update(root=args.root, force=args.force)
    except RuntimeError as exc:
        print(f"smart-context: {exc}", file=sys.stderr)
        sys.exit(1)
