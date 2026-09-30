"""Structure-biased source chunking with a bounded line fallback."""
import re
from utils import CONFIG, content_hash, symbols_from

BOUNDARY = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|enum|def|fn|func|model|route)\b|^\s*<template\b|^\s*<script\b|^\s*<style\b|^\s*#+\s+|^\s*\[.+\]\s*$", re.I)


def chunk_file(relpath, text):
    lines = text.splitlines()
    if not lines:
        return []
    limit = int(CONFIG.get("chunk_lines", 100))
    starts = [0]
    # Structural boundaries are useful only when they produce enough separation.
    candidates = [i for i, line in enumerate(lines) if i and BOUNDARY.search(line)]
    if len(candidates) >= 2:
        starts = [0] + candidates
    out = []
    for ix, start in enumerate(starts):
        end = starts[ix + 1] if ix + 1 < len(starts) else len(lines)
        if end <= start:
            continue
        # Bound oversized structural regions; split near blank lines when possible.
        pos = start
        while pos < end:
            stop = min(pos + limit, end)
            if stop < end:
                soft = next((k for k in range(stop, max(pos + limit // 2, pos), -1) if not lines[k - 1].strip()), None)
                if soft:
                    stop = soft
            body = "\n".join(lines[pos:stop]).strip()
            if body:
                syms = symbols_from(body, relpath.rsplit(".", 1)[-1])
                out.append({"id": f"{relpath}:{pos + 1}:{content_hash(body)[:12]}", "file": relpath, "start_line": pos + 1, "end_line": stop, "language": relpath.rsplit(".", 1)[-1].lower(), "symbol": syms[0] if syms else "", "content": body, "content_hash": content_hash(body)})
            pos = stop
    return out
