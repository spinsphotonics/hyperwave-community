#!/usr/bin/env python3
"""Check that every quotation in quotes.jsonl (or every kept passage in
charter_abridged.md with --charter) appears verbatim in the named text file.

Usage:
    python check_quotes.py research/<slug>
    python check_quotes.py research/<slug> --charter

Exit code 0 = all found; 1 = at least one not found.
Matching normalizes whitespace only; spelling and punctuation must match.
"""
import json
import re
import sys
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_texts(folder: Path) -> dict:
    texts = {}
    for f in (folder / "text").glob("*.txt"):
        texts[f.stem] = norm(f.read_text(encoding="utf-8", errors="replace"))
    return texts


def check_quotes(folder: Path) -> int:
    texts = load_texts(folder)
    failures = 0
    total = 0
    qfile = folder / "quotes.jsonl"
    for line_no, line in enumerate(qfile.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"FAIL line {line_no}: invalid JSON ({e})")
            failures += 1
            continue
        doc = texts.get(q.get("document_id", ""))
        if doc is None:
            print(f"FAIL {q.get('quote_id')}: no text file for document_id {q.get('document_id')}")
            failures += 1
            continue
        for key in ("text", "before", "after"):
            val = norm(q.get(key, ""))
            if val and val not in doc:
                print(f"FAIL {q.get('quote_id')}: '{key}' not found verbatim in {q['document_id']}")
                failures += 1
        # context adjacency check
        combo = norm(f"{q.get('before','')} {q.get('text','')} {q.get('after','')}")
        if combo and combo not in doc:
            print(f"WARN {q.get('quote_id')}: before+text+after not contiguous in {q['document_id']}")
    print(f"{total - failures if failures <= total else 0}/{total} quote records passed; {failures} failures")
    return 1 if failures else 0


def check_charter(folder: Path) -> int:
    texts = load_texts(folder)
    cfile = folder / "charter_abridged.md"
    content = cfile.read_text(encoding="utf-8")
    m = re.search(r"document_id\s+`([^`]+)`", content)
    if not m:
        print("FAIL: charter_abridged.md does not name a document_id in backticks")
        return 1
    doc = texts.get(m.group(1))
    if doc is None:
        print(f"FAIL: no text file for document_id {m.group(1)}")
        return 1
    body = content.split("## Text", 1)[-1]
    # remove editorial insertions [like this] and split on cuts [...]
    body = re.sub(r"\[(?!\.\.\.)[^\]]*\]", "", body)
    passages = [norm(p) for p in body.split("[...]")]
    failures = 0
    checked = 0
    for p in passages:
        if len(p) < 20:
            continue
        checked += 1
        if p not in doc:
            failures += 1
            print(f"FAIL: passage not verbatim: '{p[:80]}...'")
    print(f"{checked - failures}/{checked} passages verbatim; {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    folder = Path(sys.argv[1])
    if "--charter" in sys.argv:
        sys.exit(check_charter(folder))
    sys.exit(check_quotes(folder))
