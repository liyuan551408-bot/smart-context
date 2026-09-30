"""Local, inexpensive checks that decide whether a chunk merits semantic refinement."""
import re

from utils import CONFIG, symbols_from, token_estimate

VUE_OPEN = re.compile(r"^\s*<(template|script|style)\b[^>]*>", re.I)
VUE_CLOSE = re.compile(r"^\s*</(template|script|style)\s*>", re.I)
DECLARATION = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|enum|def|fn|func|model|route)\b|^\s*(?:const|let|var)\s+[\w$]+\s*=", re.I)


def vue_section_at(file_text, line_number):
    """Return the enclosing Vue SFC block kind without parsing source content."""
    active = None
    for number, line in enumerate(file_text.splitlines(), 1):
        close = VUE_CLOSE.match(line)
        if close:
            active = None
        opening = VUE_OPEN.match(line)
        if opening:
            active = opening.group(1).lower()
        if number == line_number:
            return active
    return active


def is_meaningful_declaration(chunk):
    content = chunk.get("content", "")
    if chunk.get("symbol"):
        return bool(DECLARATION.search(content))
    return bool(re.search(r"^\s*(?:model|enum|generator|datasource)\s+\w+\s*\{", content, re.M))


def evaluate(chunk, file_text="", config=None):
    """Return token count, structural signals, and a conservative refinement decision."""
    settings = config or CONFIG
    policy = settings.get("semantic_chunking", settings)
    content = chunk.get("content", "")
    tokens = token_estimate(content)
    language = str(chunk.get("language", "")).lower()
    symbols = symbols_from(content, language)
    reasons = []
    hard_large = tokens > int(policy.get("hard_max_chunk_tokens", 1400))
    if tokens > int(policy.get("max_chunk_tokens", 900)):
        reasons.append("oversized")
    if hard_large:
        reasons.append("hard_limit")
    if len(symbols) > 1 and tokens > int(policy.get("target_chunk_tokens", 350)):
        reasons.append("multiple_symbols")
    method = chunk.get("chunk_method", "")
    if method == "local_fallback" and tokens > int(policy.get("target_chunk_tokens", 350)):
        reasons.append("fallback_region")
    section = None
    if language == "vue" and file_text:
        section = vue_section_at(file_text, int(chunk.get("start_line", 1)))
        if section in {"template", "style"} and tokens > int(policy.get("target_chunk_tokens", 350)):
            reasons.append("large_vue_" + section)
    if not symbols and tokens > int(policy.get("max_chunk_tokens", 900)):
        reasons.append("no_meaningful_symbol")
    minimum = int(policy.get("min_chunk_tokens", 80))
    refine = bool(reasons) and (tokens >= minimum or hard_large)
    # Complete small symbols are already useful and should never incur an API call.
    if tokens < minimum and is_meaningful_declaration(chunk) and not hard_large:
        refine = False
    return {
        "tokens": tokens,
        "symbols": symbols,
        "vue_section": section,
        "reasons": list(dict.fromkeys(reasons)),
        "needs_refinement": refine,
        "meaningful_declaration": is_meaningful_declaration(chunk),
    }
