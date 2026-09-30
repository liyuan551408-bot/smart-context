"""Selective GLM semantic boundary refinement layered over the deterministic V1 chunker."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

from chunk_quality import evaluate, is_meaningful_declaration
from chunker import chunk_file
from utils import CONFIG, content_hash, symbols_from, token_estimate

SYSTEM_PROMPT = """You are a source-code segmentation engine.

Your only task is to identify semantically coherent code regions.
Do not rewrite, summarize, fix, or generate code.
Return exactly one JSON object with a single top-level key named `chunks`. Do not return the array as the top-level JSON value. The client can normalize a direct chunk list, but the preferred response is the object envelope.
Prefer complete functions, methods, classes, components, models, route handlers, logical template sections, and logical style sections.
Never split a function unnecessarily.
Never omit source lines.
Never reorder source lines.
Avoid tiny fragments.
Prefer chunks that can be understood independently.
Target 150-600 tokens when practical, but preserve semantic integrity over exact size.
The supplied line numbers are absolute original-file line numbers; return absolute numbers, never region-relative offsets.
Returned ranges must collectively cover every line from REGION_START through REGION_END exactly once.
Do not omit function declarations, opening or closing braces, or blank/comment lines between semantic regions.
No overlaps. No gaps.
Every source line in the supplied region must belong to exactly one returned chunk."""


class SemanticChunkError(RuntimeError):
    """Safe, credential-free refinement error."""


class SemanticOutputError(SemanticChunkError):
    """The provider returned a malformed response that can be repaired once."""


def _settings(config=None):
    root_config = config or CONFIG
    return root_config.get("semantic_chunking", {})


def _cache_key(relpath, region_hash, settings):
    identity = "\0".join((relpath, region_hash, settings.get("model", ""), settings.get("config_version", "1")))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _load_plan_cache(root):
    path = Path(root) / "cache" / "semantic_chunk_plans.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") == 1 and isinstance(value.get("plans"), dict):
            return path, value
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return path, {"version": 1, "plans": {}}


def _save_plan_cache(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _local_method(relpath, chunk):
    ext = Path(relpath).suffix.lower()
    if chunk.get("symbol"):
        return "local_symbol"
    if ext == ".vue" and re.search(r"^\s*</?(?:template|script|style)\b", chunk.get("content", ""), re.I | re.M):
        return "local_structure"
    if ext in {".md", ".yaml", ".yml", ".json", ".prisma"} and chunk.get("symbol"):
        return "local_structure"
    return "local_fallback"


def _local_purpose(chunk, method):
    symbol = chunk.get("symbol", "")
    if symbol:
        return "Implementation of " + symbol
    if method == "local_structure":
        return "Structured source section"
    return ""


def _make_chunk(relpath, lines, start, end, method, symbol="", purpose="", confidence=None):
    body = "\n".join(lines[start - 1:end]).strip()
    digest = content_hash(body)
    chunk = {
        "id": f"{relpath}:{start}:{digest[:12]}",
        "file": relpath,
        "start_line": start,
        "end_line": end,
        "language": Path(relpath).suffix.lstrip(".").lower(),
        "symbol": symbol or (symbols_from(body, Path(relpath).suffix.lstrip(".")) or [""])[0],
        "content": body,
        "content_hash": digest,
        "chunk_method": method,
    }
    if purpose:
        chunk["purpose"] = purpose[:240]
    if confidence is not None:
        chunk["chunk_confidence"] = float(confidence)
    return chunk


def merge_small_chunks(chunks, config=None):
    """Attach low-information fragments to an adjacent, compatible region."""
    policy = _settings(config)
    threshold = int(policy.get("tiny_chunk_tokens", 55))
    result = []
    merged_count = 0
    for original in chunks:
        chunk = dict(original)
        if result and token_estimate(chunk.get("content", "")) < threshold and not is_meaningful_declaration(chunk):
            previous = result[-1]
            prev_symbol, next_symbol = previous.get("symbol", ""), chunk.get("symbol", "")
            if not (prev_symbol and next_symbol and prev_symbol != next_symbol):
                result[-1] = _combine(previous, chunk)
                merged_count += 1
                continue
        result.append(chunk)

    index = 0
    while index < len(result):
        current = result[index]
        if token_estimate(current.get("content", "")) >= threshold or is_meaningful_declaration(current) or len(result) == 1:
            index += 1
            continue
        previous = result[index - 1] if index else None
        following = result[index + 1] if index + 1 < len(result) else None
        can_prev = previous is not None and not (previous.get("symbol") and current.get("symbol") and previous["symbol"] != current["symbol"])
        can_next = following is not None and not (following.get("symbol") and current.get("symbol") and following["symbol"] != current["symbol"])
        if can_next and (not can_prev or not current.get("symbol")):
            result[index:index + 2] = [_combine(current, following)]
            merged_count += 1
            index = max(0, index - 1)
        elif can_prev:
            result[index - 1:index + 1] = [_combine(previous, current)]
            merged_count += 1
            index = max(0, index - 1)
        else:
            index += 1
    return result, merged_count


def _combine(left, right):
    body = (left.get("content", "").rstrip() + "\n" + right.get("content", "").lstrip()).strip()
    digest = content_hash(body)
    symbol = left.get("symbol") or right.get("symbol", "")
    purposes = [x for x in (left.get("purpose", ""), right.get("purpose", "")) if x]
    out = dict(left)
    out.update({
        "id": f"{left['file']}:{min(left['start_line'], right['start_line'])}:{digest[:12]}",
        "start_line": min(left["start_line"], right["start_line"]),
        "end_line": max(left["end_line"], right["end_line"]),
        "symbol": symbol,
        "content": body,
        "content_hash": digest,
        "chunk_method": "merged_small_chunk",
    })
    if purposes:
        out["purpose"] = "; ".join(dict.fromkeys(purposes))[:240]
    out.pop("chunk_confidence", None)
    return out


def _numbered_region(lines, start, end):
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))


def _split_large_chunk(relpath, chunk, lines, max_tokens):
    """Bound request size locally while retaining contiguous original line ranges."""
    result = []
    start = int(chunk["start_line"])
    end = int(chunk["end_line"])
    group_start = start
    for number in range(start + 1, end + 1):
        candidate = "\n".join(lines[group_start - 1:number]).strip()
        if token_estimate(candidate) > max_tokens:
            split_at = number - 1
            if split_at >= group_start:
                result.append(_make_chunk(relpath, lines, group_start, split_at, "local_fallback"))
                group_start = number
    if group_start <= end:
        result.append(_make_chunk(relpath, lines, group_start, end, "local_fallback"))
    return result or [dict(chunk)]


def _user_prompt(relpath, chunk, quality, settings, lines, repair_note=""):
    numbered = _numbered_region(lines, chunk["start_line"], chunk["end_line"])
    note = ("\nThe prior response failed local validation: " + repair_note + ". Correct the ranges; do not change source lines.\n") if repair_note else ""
    return f"""Identify semantic chunk boundaries for this source region.

File: {relpath}
Language: {chunk.get('language', 'unknown')}
Original region: lines {chunk['start_line']}-{chunk['end_line']}
Local structural metadata: method={chunk.get('chunk_method')}; symbol={chunk.get('symbol') or 'none'}; quality reasons={', '.join(quality['reasons'])}
Desired sizes: minimum {settings.get('min_chunk_tokens', 80)}, target {settings.get('target_chunk_tokens', 350)}, maximum {settings.get('max_chunk_tokens', 900)} tokens.

Return exactly one JSON object with the single top-level key `chunks`, using schema {{"chunks":[{{"start_line":1,"end_line":2,"symbol":"name","purpose":"short label","confidence":0.9}}]}}. Do not return the array as the top-level JSON value.
The supplied line numbers are absolute original-file line numbers. Region bounds are {chunk['start_line']} through {chunk['end_line']}; never return relative offsets.
Returned ranges must cover every line from {chunk['start_line']} through {chunk['end_line']} exactly once. Do not omit function declarations, opening/closing braces, or blank/comment lines between semantic regions. No overlaps and no gaps.
Do not use Markdown. Do not reproduce source code. Do not reorder source lines. Ranges must stay inside the supplied original bounds. Use no more than {settings.get('max_chunks_per_region', 12)} chunks.
{note}
Numbered source lines:\n{numbered}"""


def _response_text(response):
    try:
        text = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise SemanticOutputError("Malformed GLM response") from None
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    if not isinstance(text, str):
        raise SemanticOutputError("Malformed GLM response content")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    return text


def _request_plan(relpath, chunk, quality, settings, lines, repair_note, metrics):
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise SemanticChunkError("ZHIPU_API_KEY unavailable")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(relpath, chunk, quality, settings, lines, repair_note)},
    ]
    payload = {
        "model": settings.get("model", "glm-4-flash"),
        "temperature": 0,
        "max_tokens": min(2400, max(400, int(settings.get("max_input_tokens", 6000) * 0.35))),
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    url = settings.get("base_url", "https://open.bigmodel.cn/api/paas/v4").rstrip("/") + "/chat/completions"
    retries = max(0, min(2, int(settings.get("max_retries", 2))))
    for attempt in range(retries + 1):
        metrics["glm_api_calls"] += 1
        try:
            response = requests.post(url, headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, json=payload, timeout=(10, int(settings.get("timeout_seconds", 45))))
        except requests.Timeout:
            if attempt >= retries:
                raise SemanticChunkError("GLM request timed out") from None
            time.sleep(min(2.0, 0.5 * (2 ** attempt)))
            continue
        except requests.RequestException as exc:
            if attempt >= retries:
                raise SemanticChunkError("GLM connection failed (" + type(exc).__name__ + ")") from None
            time.sleep(min(2.0, 0.5 * (2 ** attempt)))
            continue
        if response.status_code in (401, 403):
            raise SemanticChunkError("GLM authentication failed (HTTP " + str(response.status_code) + ")")
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            if attempt >= retries:
                raise SemanticChunkError("GLM temporary error persisted (HTTP " + str(response.status_code) + ")")
            time.sleep(min(2.0, 0.5 * (2 ** attempt)))
            continue
        if not response.ok:
            raise SemanticChunkError("GLM request failed (HTTP " + str(response.status_code) + ")")
        return _response_text(response)
    raise SemanticChunkError("GLM request failed")


def _normalize_response(value):
    """Normalize only the documented object envelope and direct chunk-list shapes."""
    if isinstance(value, list):
        return value, ""
    if isinstance(value, dict) and isinstance(value.get("chunks"), list):
        return value["chunks"], ""
    if isinstance(value, dict):
        return None, "JSON object must contain a chunks list"
    return None, "top-level JSON must be an object envelope or a chunk list"


def _classify_gap_line(line):
    stripped = line.strip()
    if not stripped:
        return "blank"
    if re.match(r"^(?://|#|/\*|\*|<!--|-->)", stripped) or stripped in {"*/", "-->"}:
        return "comment-only"
    if re.fullmatch(r"[{}()\[\],.;:]+", stripped):
        return "punctuation/brace-only"
    return "meaningful code"


def _punctuation_owner(line):
    stripped = line.strip()
    if stripped.startswith(("}", ")", "]")) or stripped.startswith(";"):
        return "previous"
    if stripped.startswith(("{", "(", "[")):
        return "following"
    return None


def _uncovered_lines(pieces, chunk):
    covered = set()
    for item in pieces or []:
        if not isinstance(item, dict):
            continue
        start, end = item.get("start_line"), item.get("end_line")
        if isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool) and start <= end:
            covered.update(range(max(start, chunk["start_line"]), min(end, chunk["end_line"]) + 1))
    return [line for line in range(chunk["start_line"], chunk["end_line"] + 1) if line not in covered]


def reconcile_plan_ranges(pieces, chunk, lines):
    """Repair only harmless omitted source lines by extending adjacent semantic ranges."""
    if not isinstance(pieces, list) or not pieces:
        return None, [], "chunks must be a non-empty list"
    ranges = []
    for item in pieces:
        if not isinstance(item, dict):
            return None, [], "chunk item must be an object"
        start, end = item.get("start_line"), item.get("end_line")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            return None, [], "line bounds must be integers"
        if start > end or start < chunk["start_line"] or end > chunk["end_line"]:
            return None, [], "line range outside supplied region"
        ranges.append(dict(item))
    ranges.sort(key=lambda item: item["start_line"])
    for previous, following in zip(ranges, ranges[1:]):
        if following["start_line"] <= previous["end_line"]:
            return None, [], "overlapping line ranges"

    gaps = []
    cursor = int(chunk["start_line"])
    for item in ranges:
        if item["start_line"] > cursor:
            gaps.append((cursor, item["start_line"] - 1))
        cursor = item["end_line"] + 1
    if cursor <= int(chunk["end_line"]):
        gaps.append((cursor, int(chunk["end_line"])))

    repaired_lines = []
    for gap_start, gap_end in gaps:
        classifications = [(number, _classify_gap_line(lines[number - 1])) for number in range(gap_start, gap_end + 1)]
        meaningful = [number for number, kind in classifications if kind == "meaningful code"]
        if meaningful:
            numbers = ", ".join(str(number) for number in meaningful)
            return None, repaired_lines, "meaningful source omitted at line(s): " + numbers
        if any(kind == "punctuation/brace-only" for _, kind in classifications):
            owners = {_punctuation_owner(lines[number - 1]) for number, kind in classifications if kind == "punctuation/brace-only"}
            if None in owners or len(owners) != 1:
                return None, repaired_lines, "ambiguous standalone punctuation in uncovered line(s) " + str(gap_start) + "-" + str(gap_end)
            owner = owners.pop()
        elif any(kind == "comment-only" for _, kind in classifications):
            owner = "following" if gap_end < int(chunk["end_line"]) else "previous"
        else:
            owner = "previous" if gap_start > int(chunk["start_line"]) else "following"

        if owner == "previous":
            previous = next((item for item in reversed(ranges) if item["end_line"] == gap_start - 1), None)
            if previous is None:
                return None, repaired_lines, "unowned leading closing punctuation"
            previous["end_line"] = gap_end
        else:
            following = next((item for item in ranges if item["start_line"] == gap_end + 1), None)
            if following is None:
                return None, repaired_lines, "unowned trailing opening punctuation"
            following["start_line"] = gap_start
        repaired_lines.extend(range(gap_start, gap_end + 1))
    return ranges, repaired_lines, ""


def _validate_plan(plan, chunk, lines, settings):
    pieces, normalize_error = _normalize_response(plan)
    if pieces is None:
        return None, normalize_error
    if not pieces or len(pieces) > int(settings.get("max_chunks_per_region", 12)):
        return None, "invalid chunk count"
    expected = int(chunk["start_line"])
    validated = []
    threshold = float(settings.get("confidence_threshold", 0.70))
    hard_max = int(settings.get("hard_max_chunk_tokens", 1400))
    for item in pieces:
        if not isinstance(item, dict):
            return None, "chunk item must be an object"
        start, end = item.get("start_line"), item.get("end_line")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            return None, "line bounds must be integers"
        if start > end or start < chunk["start_line"] or end > chunk["end_line"]:
            return None, "line range outside supplied region"
        if start != expected:
            return None, "overlap or gap in line ranges"
        confidence = item.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                return None, "confidence must be between zero and one"
            if confidence < threshold:
                return None, "confidence below threshold"
        body = "\n".join(lines[start - 1:end]).strip()
        if token_estimate(body) > hard_max:
            return None, "chunk exceeds hard token maximum"
        symbol = item.get("symbol", "")
        purpose = item.get("purpose", "")
        if not isinstance(symbol, str) or not isinstance(purpose, str):
            return None, "symbol and purpose must be strings"
        validated.append({"start_line": start, "end_line": end, "symbol": symbol[:100], "purpose": purpose[:240], "confidence": confidence})
        expected = end + 1
    if expected != int(chunk["end_line"]) + 1:
        return None, "line ranges do not cover the supplied region"
    return validated, ""


def _semantic_pieces(relpath, chunk, lines, root, settings, quality, metrics, cache_state):
    region_body = "\n".join(lines[chunk["start_line"] - 1:chunk["end_line"]]).strip()
    region_hash = content_hash(region_body)
    key = _cache_key(relpath, region_hash, settings)
    plans = cache_state[1]["plans"]
    if settings.get("cache_plans", True) and key in plans:
        cached, error = _validate_plan({"chunks": plans[key]}, chunk, lines, settings)
        if cached:
            metrics["glm_plans_reused"] += 1
            return cached
        plans.pop(key, None)

    repair_note = ""
    for validation_attempt in range(2):
        try:
            raw = _request_plan(relpath, chunk, quality, settings, lines, repair_note, metrics)
            parsed = json.loads(raw)
        except SemanticOutputError as exc:
            metrics["glm_response_top_level_type"] = "malformed"
            metrics["glm_normalized_chunk_count"] = 0
            metrics["glm_validation_result"] = "rejected"
            repair_note = str(exc)
            metrics["glm_rejection_reason"] = repair_note
            continue
        except SemanticChunkError:
            return None
        except (ValueError, TypeError):
            parsed = None
        metrics["glm_response_top_level_type"] = type(parsed).__name__
        normalized, normalize_error = _normalize_response(parsed)
        metrics["glm_normalized_chunk_count"] = len(normalized) if normalized is not None else 0
        metrics["glm_validation_result"] = "rejected"
        metrics["glm_rejection_reason"] = normalize_error
        metrics["glm_ranges_before_reconciliation"] = [[item.get("start_line"), item.get("end_line")] for item in normalized if isinstance(item, dict)] if normalized is not None else []
        metrics["glm_uncovered_lines_before_reconciliation"] = _uncovered_lines(normalized, chunk) if normalized is not None else list(range(chunk["start_line"], chunk["end_line"] + 1))
        if normalized is None:
            repair_note = normalize_error
            continue
        reconciled, repaired_lines, reconcile_error = reconcile_plan_ranges(normalized, chunk, lines)
        metrics["glm_harmless_lines_reconciled"] = repaired_lines
        metrics["glm_ranges_after_reconciliation"] = [[item["start_line"], item["end_line"]] for item in reconciled] if reconciled is not None else []
        if reconciled is None:
            repair_note = reconcile_error
            metrics["glm_rejection_reason"] = reconcile_error
            continue
        pieces, error = _validate_plan(reconciled, chunk, lines, settings)
        if pieces:
            metrics["glm_validation_result"] = "accepted"
            metrics["glm_rejection_reason"] = ""
            if settings.get("cache_plans", True):
                plans[key] = pieces
                _save_plan_cache(cache_state[0], cache_state[1])
            return pieces
        repair_note = error or "response was not valid JSON"
        metrics["glm_rejection_reason"] = repair_note
    return None


def adaptive_chunk_file(relpath, text, root, config=None):
    """Apply selective semantic refinement, preserving local V1 chunks on every failure."""
    root_config = config or CONFIG
    settings = _settings(root_config)
    local = chunk_file(relpath, text)
    metrics = {"local_chunks": len(local), "glm_refinement_candidates": 0, "glm_plans_reused": 0, "glm_api_calls": 0, "glm_refined_chunks": 0, "small_chunks_merged": 0, "fallback_count": 0, "glm_response_top_level_type": "", "glm_normalized_chunk_count": 0, "glm_validation_result": "not_run", "glm_rejection_reason": ""}
    if not settings.get("enabled", False):
        return local, metrics
    lines = text.splitlines()
    enriched = []
    cache_state = _load_plan_cache(root)
    refinement_regions = []
    max_input_tokens = int(settings.get("max_input_tokens", 6000))
    for original in local:
        if token_estimate(original.get("content", "")) > max_input_tokens:
            split = _split_large_chunk(relpath, original, lines, max_input_tokens)
            refinement_regions.extend(split)
            metrics["local_chunks"] += max(0, len(split) - 1)
        else:
            refinement_regions.append(original)
    for original in refinement_regions:
        chunk = dict(original)
        method = _local_method(relpath, chunk)
        chunk["chunk_method"] = method
        purpose = _local_purpose(chunk, method)
        if purpose:
            chunk["purpose"] = purpose
        quality = evaluate(chunk, text, root_config)
        if not quality["needs_refinement"]:
            enriched.append(chunk)
            continue
        metrics["glm_refinement_candidates"] += 1
        if quality["tokens"] > max_input_tokens:
            # Preserve bounded API input. V1's deterministic line splitter remains the fallback.
            enriched.append(chunk)
            continue
        try:
            pieces = _semantic_pieces(relpath, chunk, lines, root, settings, quality, metrics, cache_state)
        except Exception:
            pieces = None
        if not pieces:
            metrics["fallback_count"] += 1
            enriched.append(chunk)
            continue
        for piece in pieces:
            refined = _make_chunk(relpath, lines, piece["start_line"], piece["end_line"], "glm_semantic", piece["symbol"], piece["purpose"], piece["confidence"])
            enriched.append(refined)
            metrics["glm_refined_chunks"] += 1
    merged, count = merge_small_chunks(enriched, root_config)
    metrics["small_chunks_merged"] = count
    return merged, metrics
