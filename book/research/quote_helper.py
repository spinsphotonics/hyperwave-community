#!/usr/bin/env python3
"""Helper: given a text file and an exact substring, print the quote plus
20 words before/after, verified to be contiguous and present."""
import sys, re

def words(s):
    return s.split()

def find_quote(path, needle):
    text = open(path, encoding='utf-8').read()
    idx = text.find(needle)
    if idx == -1:
        print("NOT FOUND:", needle[:80])
        return None
    before_text = text[:idx]
    after_text = text[idx+len(needle):]
    before_words = words(before_text)[-20:]
    after_words = words(after_text)[:20]
    before = " ".join(before_words)
    after = " ".join(after_words)
    return {"text": needle, "before": before, "after": after}

if __name__ == "__main__":
    path = sys.argv[1]
    needle = sys.argv[2]
    r = find_quote(path, needle)
    if r:
        import json
        print(json.dumps(r, ensure_ascii=False, indent=2))
