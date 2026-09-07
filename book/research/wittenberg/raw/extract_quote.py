#!/usr/bin/env python3
"""Helper: given a text file and an exact substring (whitespace-normalized),
find it and print the quote plus 20 words before/after, whitespace-normalized,
and the nearest preceding [[p. N]] marker.
Usage: python3 extract_quote.py <file> "<substring, normalized whitespace>"
"""
import re
import sys

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

def main():
    path, needle = sys.argv[1], sys.argv[2]
    raw = open(path, encoding="utf-8").read()
    text = norm(raw)
    needle_n = norm(needle)
    idx = text.find(needle_n)
    if idx == -1:
        print("NOT FOUND")
        return
    before_text = text[:idx]
    after_text = text[idx+len(needle_n):]
    before_words = before_text.split()[-20:]
    after_words = after_text.split()[:20]
    # find nearest page marker before idx
    pm = re.findall(r"\[\[p\. (\d+)\]\]", before_text)
    page = pm[-1] if pm else "?"
    print("PAGE:", page)
    print("BEFORE:", " ".join(before_words))
    print("TEXT:", needle_n)
    print("AFTER:", " ".join(after_words))

if __name__ == "__main__":
    main()
