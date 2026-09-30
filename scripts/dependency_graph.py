"""Small local dependency extractor; expansion is bounded by the caller."""
import re
from pathlib import Path

IMPORT_PATTERNS = [
    r"(?:import|export)\s+(?:.+?\s+from\s+)?['\"]([^'\"]+)['\"]",
    r"require\(\s*['\"]([^'\"]+)['\"]\s*\)",
    r"^\s*from\s+([\w.]+)\s+import\s+",
    r"^\s*import\s+([\w.]+)",
]


def dependencies(root, relpath, content, known_files):
    root = Path(root)
    refs = set()
    ext = Path(relpath).suffix
    for pattern in IMPORT_PATTERNS:
        for ref in re.findall(pattern, content, flags=re.M):
            ref = ref.split(" as ")[0]
            if not (ref.startswith(".") or ext in {".py", ".java", ".kt"}):
                continue
            base = (Path(relpath).parent / ref).as_posix()
            options = [base, base + ext, base + ".js", base + ".ts", base + ".tsx", base + ".jsx", base + ".vue", (Path(base) / "index.js").as_posix(), (Path(base) / "index.ts").as_posix()]
            for opt in options:
                norm = Path(opt).as_posix().lstrip("./")
                if norm in known_files and norm != relpath:
                    refs.add(norm)
    # Prisma relations stay inside the schema file, whose chunks can be pulled by model name.
    if ext == ".prisma":
        refs.update(re.findall(r"\b([A-Z]\w+)\s*(?:\[\])?\s*\?*\s+\w+", content))
    return sorted(refs)
