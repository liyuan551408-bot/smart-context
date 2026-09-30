"""Retrieve a compact, semantically ranked context set."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dependency_graph import dependencies
from utils import CONFIG, embed, git, load_index, repo_root, token_estimate


SOURCE_INTENT = ("implementation", "code", "function", "service", "controller", "logic",
                 "实现", "代码", "函数", "服务", "控制器", "逻辑")
DOC_INTENT = ("documentation", "document", "docs", "readme", "说明文档", "文档", "依赖", "配置")


def identifier_terms(value):
    """Tokenize paths and camelCase/PascalCase identifiers into generic terms."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    value = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", value)
    return {part.lower() for part in re.findall(r"[A-Za-z0-9]+", value) if len(part) > 1}


def reranking_adjustments(query, chunk, name_score=0.0):
    """Compute conservative V2.1 modifiers, independent of the established base score."""
    token_count = token_estimate(chunk.get("content", ""))
    meaningful_symbol = bool(identifier_terms(chunk.get("symbol", "")))
    tiny = -0.04 if token_count < 30 else (-0.025 if token_count < 50 else 0.0)
    if token_count < 80 and not meaningful_symbol:
        tiny -= 0.01
    tiny = max(-0.05, tiny)
    # Preserve specifically requested symbols and filenames, even when their chunks are short.
    if name_score >= 1.0:
        tiny = 0.0

    query_terms = identifier_terms(query)
    path = Path(chunk.get("file", ""))
    basename_terms = identifier_terms(path.stem)
    directory_terms = set().union(*(identifier_terms(part) for part in path.parts[:-1])) if path.parts[:-1] else set()
    basename_hits = query_terms & basename_terms
    directory_hits = query_terms & directory_terms
    path_role = min(0.05, 0.025 * len(basename_hits) + 0.008 * len(directory_hits - basename_hits))

    query_lower = query.lower()
    source_intent = any(term in query_lower for term in SOURCE_INTENT)
    documentation_intent = any(term in query_lower for term in DOC_INTENT)
    source_type = 0.0
    filename = path.name.lower()
    if source_intent and not documentation_intent:
        if filename == "readme" or filename.startswith("readme."):
            source_type = -0.035
        elif filename in {"package.json", "package-lock.json"}:
            source_type = -0.03

    modifier_total = max(-0.06, min(0.06, tiny + path_role + source_type))
    return {"tiny_adjustment": tiny, "path_role_adjustment": path_role,
            "source_type_adjustment": source_type, "modifier_total": modifier_total}


def latest_change(root, rel):
    stamp = git(root, ["log", "-1", "--format=%ct", "--", rel]).strip()
    if not stamp:
        return 0.0
    try:
        age_days = max(0.0, (time.time() - int(stamp)) / 86400)
        return 0.25 * (0.5 ** (age_days / 30.0))
    except ValueError:
        return 0.0


def retrieve(query, top_k=None, max_tokens=None, root=None):
    if not query or not query.strip():
        raise ValueError("Query is empty. Supply a coding task.")
    root = repo_root(root)
    chunks, metadata, vectors = load_index(root)
    if not chunks:
        return {"task": query, "selected": [], "dependencies": [], "estimated_tokens": token_estimate(query), "max_tokens": max_tokens or CONFIG["max_context_tokens"]}
    qvec = embed([query])[0]
    if vectors.shape[1] != qvec.shape[0]:
        raise RuntimeError("Embedding dimension differs from index; run scripts/update_index.py to rebuild it.")
    sims = vectors @ qvec
    candidate_k = min(len(chunks), int(CONFIG["candidate_k"]))
    idxs = np.argpartition(sims, -candidate_k)[-candidate_k:]
    idxs = idxs[np.argsort(sims[idxs])[::-1]]
    seed_idxs = list(map(int, idxs))
    seed_files = {chunks[i]["file"] for i in seed_idxs[:min(5, len(seed_idxs))]}
    known_files = set(metadata.get("files", {}))
    by_file = {}
    for i, chunk in enumerate(chunks):
        by_file.setdefault(chunk["file"], []).append(i)
    file_contents = {f: "\n".join(chunks[i]["content"] for i in indices) for f, indices in by_file.items()}
    dep_files = set()
    for source in seed_files:
        dep_files.update(dependencies(root, source, file_contents.get(source, ""), known_files))
    # Keep semantic candidates and at most a few direct dependency chunks.
    candidate_idxs = set(seed_idxs)
    for dep_file in dep_files:
        for i in by_file.get(dep_file, [])[:2]:
            candidate_idxs.add(i)
    weights = CONFIG["weights"]
    changed = {ln[3:].strip().replace("\\", "/") for ln in git(root, ["status", "--porcelain"]).splitlines() if len(ln) > 3}
    terms = set(re.findall(r"[A-Za-z_$][\w$.-]{2,}", query.lower()))
    seeds = {chunks[i]["file"] for i in seed_idxs[:8]}
    scored = []
    for i in candidate_idxs:
        c = chunks[i]
        semantic = max(0.0, float(sims[i]))
        dep = 1.0 if c["file"] in dep_files else 0.0
        recent = latest_change(root, c["file"])
        names = {Path(c["file"]).stem.lower(), c.get("symbol", "").lower()}
        name_score = 0.0
        if terms & names:
            name_score = 1.0
        elif any(name and any(t in name or name in t for t in terms) for name in names):
            name_score = 0.45
        working = 1.0 if c["file"] in changed else 0.0
        score = (weights["semantic"] * semantic + weights["dependency"] * dep + weights["git_recency"] * recent + weights["name_match"] * name_score + weights["working_file"] * working)
        if CONFIG.get("semantic_chunking", {}).get("enabled", False):
            # Metadata is a small tie-breaker; source-code embedding remains dominant.
            purpose_terms = set(re.findall(r"[A-Za-z_$][\w$.-]{2,}", c.get("purpose", "").lower()))
            if terms & purpose_terms:
                score += 0.015
            if c.get("chunk_method") in {"local_symbol", "glm_semantic"} and c.get("symbol"):
                score += 0.01
        base_score = score
        adjustments = reranking_adjustments(query, c, name_score)
        score = base_score + adjustments["modifier_total"]
        # Dependency expansion can assist selection but cannot outrank strong semantic results by itself.
        reason = "semantic match"
        if c["file"] in dep_files:
            reason = "direct dependency of a semantic match"
        if name_score:
            reason += " + exact name match"
        if working:
            reason += " + current working tree"
        scored.append({"chunk": c, "semantic_similarity": semantic, "base_score": base_score,
                       **{key: adjustments[key] for key in ("tiny_adjustment", "path_role_adjustment", "source_type_adjustment")},
                       "score": score, "reason": reason})
    scored.sort(key=lambda x: x["score"], reverse=True)
    budget = int(max_tokens or CONFIG["max_context_tokens"])
    used = token_estimate("Task: " + query)
    selected, selected_deps = [], set()
    limit = int(top_k or CONFIG["top_k"])
    for item in scored:
        if len(selected) >= limit:
            break
        c = item["chunk"]
        cost = token_estimate(c["content"])
        if used + cost > budget:
            continue
        selected.append({**item, "tokens": cost})
        used += cost
        if c["file"] in dep_files:
            selected_deps.add(c["file"])
    return {"task": query, "selected": selected, "dependencies": sorted(selected_deps), "estimated_tokens": used, "max_tokens": budget}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query_pos", nargs="?", help="Natural-language coding task")
    ap.add_argument("--query", dest="query_opt", help="Natural-language coding task")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    ap.add_argument("--root", help="Repository path (defaults to current directory)")
    args = ap.parse_args()
    query = args.query_opt or args.query_pos
    try:
        result = retrieve(query, args.top_k, args.max_tokens, args.root)
    except (RuntimeError, ValueError) as exc:
        print(f"smart-context: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("SMART CONTEXT\n\nTask:\n" + result["task"] + "\n\nSelected context:")
    for n, item in enumerate(result["selected"], 1):
        c = item["chunk"]
        print(f"\n{n}. {c['file']}\n   Lines {c['start_line']}-{c['end_line']}  Score: {item['score']:.3f}\n   Reason: {item['reason']}\n```{c['language']}\n{c['content']}\n```")
    if result["dependencies"]:
        print("\nRelevant dependencies:\n" + "\n".join(f"- {x}" for x in result["dependencies"]))
    print(f"\nEstimated context tokens: {result['estimated_tokens']} / {result['max_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
