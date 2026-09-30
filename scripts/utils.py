"""Shared repository, embedding, and index utilities."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

SKILL_DIR = Path(__file__).resolve().parents[1]
CONFIG = json.loads((SKILL_DIR / "config" / "config.json").read_text(encoding="utf-8"))


def repo_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return Path(out.stdout.strip()).resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    return start


def cache_dir(root):
    path = Path(root) / "cache"
    path.mkdir(parents=True, exist_ok=True)
    gi = Path(root) / ".gitignore"
    try:
        old = gi.read_text(encoding="utf-8") if gi.exists() else ""
        if "cache/" not in old.splitlines():
            with gi.open("a", encoding="utf-8") as f:
                if old and not old.endswith("\n"):
                    f.write("\n")
                f.write("\n# smart-context generated index\ncache/\n")
    except OSError:
        pass
    return path


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def git(root, args):
    try:
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def git_files(root):
    output = git(root, ["ls-files", "-co", "--exclude-standard"])
    return {x.replace("\\", "/") for x in output.splitlines() if x.strip()}


def load_index(root):
    c = cache_dir(root)
    try:
        chunks = json.loads((c / "chunks.json").read_text(encoding="utf-8"))
        metadata = json.loads((c / "metadata.json").read_text(encoding="utf-8"))
        embeddings = np.load(c / "embeddings.npy")
        if len(chunks) != len(embeddings):
            raise ValueError("chunk/embedding count mismatch")
        expected = (metadata.get("embedding_provider") == CONFIG["embedding_provider"] and metadata.get("embedding_model") == CONFIG["embedding_model"] and embeddings.ndim == 2 and int(metadata.get("embedding_dimension", -1)) == embeddings.shape[1])
        if chunks and not expected:
            raise RuntimeError("Index embedding provider, model, or dimension is incompatible; run scripts/build_index.py to rebuild it.")
        return chunks, metadata, embeddings
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Index is missing or outdated ({exc}); run scripts/update_index.py") from exc


def embed(texts, batch_size=32):
    """Embed texts via SiliconFlow; never include credentials in diagnostics."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not configured.")
    url = CONFIG["embedding_base_url"].rstrip("/") + "/embeddings"
    vectors = [None] * len(texts)
    size = max(1, int(CONFIG.get("embedding_batch_size", batch_size)))
    for offset in range(0, len(texts), size):
        batch = texts[offset:offset + size]
        payload = {"model": CONFIG["embedding_model"], "input": batch, "encoding_format": "float"}
        for attempt in range(4):
            try:
                response = requests.post(url, headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, json=payload, timeout=(10, 60))
            except requests.Timeout:
                if attempt == 3:
                    raise RuntimeError("SiliconFlow embedding request timed out after bounded retries.")
                time.sleep(0.5 * (2 ** attempt))
                continue
            except requests.RequestException as exc:
                if attempt == 3:
                    raise RuntimeError(f"SiliconFlow connection failed after bounded retries ({type(exc).__name__}).")
                time.sleep(0.5 * (2 ** attempt))
                continue
            if response.status_code in (401, 403):
                raise RuntimeError(f"SiliconFlow authentication failed (HTTP {response.status_code}); check SILICONFLOW_API_KEY and account access.")
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                if attempt == 3:
                    raise RuntimeError(f"SiliconFlow temporary error persisted after bounded retries (HTTP {response.status_code}).")
                time.sleep(0.5 * (2 ** attempt))
                continue
            if not response.ok:
                raise RuntimeError(f"SiliconFlow embedding request failed (HTTP {response.status_code}).")
            try:
                data = response.json()["data"]
                if len(data) != len(batch):
                    raise ValueError("result count mismatch")
                ordered = [None] * len(batch)
                for position, item in enumerate(data):
                    idx = int(item.get("index", position))
                    if idx < 0 or idx >= len(batch) or ordered[idx] is not None:
                        raise ValueError("invalid embedding index")
                    ordered[idx] = np.asarray(item["embedding"], dtype=np.float32)
                if any(v is None for v in ordered) or len({v.shape for v in ordered}) != 1:
                    raise ValueError("missing or inconsistent embedding vectors")
                mat = np.stack(ordered)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                if np.any(norms == 0) or not np.all(np.isfinite(mat)):
                    raise ValueError("invalid vector values")
                mat = mat / norms
                vectors[offset:offset + len(batch)] = list(mat)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise RuntimeError(f"Malformed or incomplete SiliconFlow embedding response ({exc}).") from None
            break
    return np.stack(vectors).astype(np.float32)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def symbols_from(text, language):
    patterns = [r"\b(?:async\s+)?function\s+([\w$]+)", r"\bclass\s+([\w$]+)", r"\b(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>", r"\b(?:def|class|fn|func)\s+([\w$]+)", r"\bmodel\s+([\w$]+)", r"^\s*(?:public|private|protected|static|async|fun|void|int|String|boolean|[\w<>?,.]+)\s+([\w$]+)\s*\([^;]*\)\s*\{" ]
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, re.M))
    return list(dict.fromkeys(found))


def token_estimate(text):
    # Conservative, model-independent approximation for code and prose.
    return max(1, int(len(text) / 3.6)) if text else 0


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
