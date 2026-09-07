#!/usr/bin/env python3
"""Build the finished, reader-facing edition of the book.

Stage 1 (`html`): assemble book/assembly/final.html from the Introduction, the six
Part introductions, the 50 chapters (each with its abridged charter body merged in
from research/<slug>/charter_abridged.md), and Appendices A-D.
Stage 2 (`pdf`): print final.html to book/assembly/The_School_of_Death.pdf with the
pre-installed Chromium via the DevTools protocol (page numbers, PDF outline).

All transformations are presentation-only and applied at build time; nothing under
book/research/ or book/chapters/ is modified. The research record stays the audit
trail; this script only translates its working vocabulary into book language.

Usage: python3 build_final.py html|pdf|all
"""
import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import markdown

BOOK = Path(__file__).resolve().parent.parent
CHAPTERS = BOOK / "chapters"
SYNTH = BOOK / "synthesis"
RESEARCH = BOOK / "research"
OUT_HTML = BOOK / "assembly" / "final.html"
OUT_PDF = BOOK / "assembly" / "The_School_of_Death.pdf"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

PART_TITLES = {
    "I": "Reformation Academies (1518–1620)",
    "II": "Puritans, Pietists and Dissenters (1636–1790)",
    "III": "The Missionary Awakening and the Seminary (1792–1860)",
    "IV": "Faith Missions and the Bible Institute (1865–1930)",
    "V": "Translators, Tribes and the Auca Five (1930–1960)",
    "VI": "Lloyd-Jones and the Reformed Recovery (1938–1980)",
}
PART_ORDER = ["I", "II", "III", "IV", "V", "VI"]

# --------------------------------------------------------------------------- text transforms

def strip_to_first_heading(text: str, pattern: str = r"^# ") -> str:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return (text[m.start():] if m else text).rstrip() + "\n"


def demote_headings(text: str, by: int = 2) -> str:
    return re.sub(r"^(#{1,4})\s", lambda m: "#" * min(6, len(m.group(1)) + by) + " ", text, flags=re.MULTILINE)


def charter_body(slug: str) -> str | None:
    p = RESEARCH / slug / "charter_abridged.md"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    m = re.search(r"^## Text[^\n]*\n", text, flags=re.MULTILINE)
    if not m:
        return None
    body = text[m.end():].strip()
    if not body or (body.split()[0].rstrip(".,;:").lower() == "none" and len(body.split()) < 80):
        return None  # the chapter's own Charter section already states that no document exists
    return demote_headings(body)


def editorial_note(preamble: str) -> str | None:
    """Turn a chapter file's working preamble (coverage note) into a reader-facing note."""
    lines = []
    for line in preamble.splitlines():
        s = line.strip()
        if not s or s.startswith(("STATUS:", "TICKET:", "ROLE:", "INPUTS READ", "|")):
            continue
        lines.append(line.rstrip())
    note = "\n".join(lines).strip()
    if not note:
        return None
    note = re.sub(r"^\*\*Coverage note[^*]*\*\*:?\s*", "", note)
    note = re.sub(r"^\*\*[^*]*Coverage note[^*]*\*\*:?\s*", "", note)
    note = note.replace("Coverage note:", "").strip()
    return '<div class="editorial" markdown="1">\n\n*Editorial note on sources.* ' + note + "\n\n</div>"


def book_language(text: str, slug: str | None = None) -> str:
    """Translate working-document vocabulary into reader-facing language."""
    t = text
    # inline rights flags were for the rights log; the log is settled (private edition)
    t = re.sub(r"\s*\[RIGHTS:[^\]]*\]", "", t)
    t = t.replace("[RIGHTS/ACCESS:", "[Access:")
    # source paragraphs are hard-wrapped, so multi-word phrases may straddle a line break
    t = re.sub(r"\bcoverage\s+note\b", "editorial note", t)
    t = re.sub(r"\bCoverage\s+note\b", "Editorial note", t)
    t = re.sub(r"\b(?:per|under)\s+the\s+task\s+instructions\b", "for this study", t)
    t = re.sub(r"\bthe\s+task\s+instructions\b", "the scope set for this study", t)
    t = re.sub(r"\btask\s+instructions\b", "the scope set for this study", t)
    t = re.sub(r"\bthe\s+companion\s+file\s+`charter_abridged\.md`", "the Charter text below", t)
    t = re.sub(r"(?<!research )\bdossier\b", "research dossier", t)
    t = re.sub(r"\bthe dossier'?s? (?:field )?([FACLTMSRX]\d)\b", r"the research dossier for this chapter, field \1", t)
    t = re.sub(r"\bdossier (?:field )?([FACLTMSRX]\d)\b", r"the research dossier for this chapter, field \1", t)
    # companion-file references
    t = t.replace("the companion file `charter_abridged.md`", "the Charter text below")
    t = t.replace("`charter_abridged.md`", "the Charter section of this chapter")
    t = t.replace("charter_abridged.md", "the Charter section of this chapter")
    # research-record paths
    t = re.sub(r"`?research/[a-z0-9-]+/discrepancies\.md`?", "the discrepancy record for this chapter", t)
    t = re.sub(r"`?research/[a-z0-9-]+/dossier\.md`?", "the research dossier for this chapter", t)
    t = re.sub(r"`?research/[a-z0-9-]+/quotes\.jsonl`?", "the quote records for this chapter", t)
    t = re.sub(r"`?research/[a-z0-9-]+/sources\.csv`?", "the source log for this chapter", t)
    t = re.sub(r"`?research/[a-z0-9-]+/verification\.md`?", "the verification report for this chapter", t)
    t = re.sub(r"`?research/[a-z0-9-]+/scout_notes\.md`?", "the search notes for this chapter", t)
    t = re.sub(r"`?research/[a-z0-9-]+/text/([A-Za-z0-9._-]+)`?", r"the transcription \1", t)
    t = re.sub(r"`?research/[a-z0-9-]+/raw/([A-Za-z0-9._-]+)`?", r"the raw fetched file \1", t)
    t = re.sub(r"`?research/[a-z0-9-]+/?`?", "this chapter's research files", t)
    t = t.replace("`discrepancies.md`", "the discrepancy record for this chapter").replace("discrepancies.md", "the discrepancy record for this chapter")
    t = t.replace("`dossier.md`", "the research dossier for this chapter").replace("dossier.md", "the research dossier for this chapter")
    t = t.replace("`quotes.jsonl`", "the quote records for this chapter").replace("quotes.jsonl", "the quote records for this chapter")
    t = t.replace("`sources.csv`", "the source log for this chapter").replace("sources.csv", "the source log for this chapter")
    t = t.replace("`verification.md`", "the verification report for this chapter").replace("verification.md", "the verification report for this chapter")
    t = t.replace("`scout_notes.md`", "the search notes for this chapter").replace("scout_notes.md", "the search notes for this chapter")
    # plan documents and roles
    t = t.replace(" (`02_institution_roster.md`)", "").replace("`02_institution_roster.md`", "the book's plan")
    t = t.replace("`01_book_design.md`", "the book's design rules").replace("01_book_design.md", "the book's design rules")
    t = re.sub(r"`?05_repositories_and_search_strings\.md`?", "the book's search plan", t)
    t = re.sub(r"`?07_risk_register_and_qa\.md`?", "the book's risk register", t)
    t = re.sub(r"`?0\d_[a-z_]+\.md`?", "the book's plan", t)
    t = t.replace("book/plan/", "the book's plan documents, ")
    t = t.replace("book/roster.csv", "the book's institution list").replace("roster.csv", "the book's institution list")
    t = t.replace("per roster", "per the book's plan").replace("the roster's", "the book's plan's")
    t = re.sub(r"\bthe roster\b", "the book's plan", t)
    t = re.sub(r"\bRule (\d+)\b", "this book's method", t)
    t = re.sub(r"\bProcedure [A-Z]\d?\b", "this book's method", t)
    t = t.replace("check_quotes.py", "the verbatim-quotation check")
    t = re.sub(r"\bFetcher\b", "transcription", t)
    t = re.sub(r"\bTier A\b", "first-tier", t).replace("Tier B", "second-tier")
    t = re.sub(r"\bfollow-up tickets?\b", lambda m: "follow-up task" + ("s" if m.group(0).endswith("s") else ""), t)
    t = re.sub(r"\btickets?\b", lambda m: "task" + ("s" if m.group(0).endswith("s") else ""), t)
    # "this session" -> study language (adverbial use is the common case)
    t = re.sub(r"\bthis\s+session's", "this study's", t)
    t = re.sub(r"\bthis\s+session\s+(was|is|has|had|did|could|does|can)\b", r"this study \1", t)
    t = re.sub(r"\bThis\s+session\b", "This study", t)
    t = re.sub(r"\b(in|during)\s+this\s+session\b", r"\1 this study", t)
    t = re.sub(r"\bthis\s+session\b", "for this study", t)
    t = re.sub(r"\b[Tt]he\s+roster\s+that\s+commissioned\s+this\s+book\b", "The plan that commissioned this book", t)
    t = re.sub(r"\bthe\s+roster\b", "the book's plan", t)
    t = re.sub(r"\bThe\s+roster\b", "The book's plan", t)
    t = re.sub(r"\bfollow-up\s+tickets?\b", lambda m: "follow-up task" + ("s" if m.group(0).endswith("s") else ""), t)
    t = re.sub(r"\(out of scope for this study", "(outside this study's scope", t)
    # a footnote whose text now begins with a substituted phrase should start with a capital
    t = re.sub(r"^(\[\^[^\]]+\]:\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t, flags=re.MULTILINE)
    return t


def drop_placeholder_paragraphs(text: str) -> str:
    paras = re.split(r"\n\s*\n", text)
    kept = []
    for p in paras:
        s = p.strip()
        if s.startswith(("*(See the companion file", "[See the companion file", "(See the companion file")) and "companion file" in s:
            continue
        kept.append(p)
    return "\n\n".join(kept)


def namespace_footnotes(text: str, prefix: str) -> str:
    return re.sub(r"\[\^([A-Za-z0-9_-]+)\]", lambda m: f"[^{prefix}-{m.group(1)}]", text)


def build_chapter(row: dict, number: int, part: str) -> str:
    slug = row["slug"]
    matches = sorted(CHAPTERS.glob(f"{part}-*-{slug}.md"))
    if not matches:
        return f"# Chapter {number}: {row['name']}\n\n*Chapter not found on disk.*\n"
    raw = matches[0].read_text(encoding="utf-8")
    mh = re.search(r"^# Chapter", raw, flags=re.MULTILINE)
    preamble, text = (raw[:mh.start()], raw[mh.start():]) if mh else ("", raw)
    text = text.rstrip() + "\n"
    text = re.sub(r"^# Chapter \d+:", f"# Chapter {number}:", text, count=1, flags=re.MULTILINE)
    note = editorial_note(preamble)
    if note:
        # place the note directly under the title, so in-text references to it read correctly
        first_nl = text.index("\n")
        text = text[:first_nl + 1] + "\n" + note + "\n" + text[first_nl + 1:]
    text = drop_placeholder_paragraphs(text)
    body = charter_body(slug)
    if body:
        block = "\n\n<div class=\"charter\" markdown=\"1\">\n\n" + body + "\n\n</div>\n\n"
        # insert at end of the Charter section, i.e. just before the Admission heading
        m = re.search(r"^## Admission", text, flags=re.MULTILINE)
        if m:
            text = text[:m.start()].rstrip() + block + text[m.start():]
        else:
            text = text.rstrip() + block
    text = book_language(text, slug)
    text = namespace_footnotes(text, f"c{number}")
    return text


def load_roster():
    with open(BOOK / "roster.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["tier"] in ("A", "B")]
    parts = {}
    for r in rows:
        parts.setdefault(r["part"], []).append(r)
    for p in parts:
        parts[p].sort(key=lambda r: int(r["chapter_order"]))
    return parts


# --------------------------------------------------------------------------- html

CSS = """
@page { size: A4; margin: 22mm 20mm 24mm 20mm; }
html { font-size: 10.5pt; }
body { font-family: "DejaVu Serif", Georgia, "Times New Roman", serif; line-height: 1.45; color: #111;
       max-width: 100%; margin: 0; }
p { margin: 0 0 0.75em 0; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
h1 { page-break-before: always; font-size: 19pt; font-weight: normal; margin: 0 0 1.2em 0;
     padding-bottom: 0.3em; border-bottom: 1px solid #999; line-height: 1.25; }
h2 { font-size: 12.5pt; font-weight: bold; margin: 1.6em 0 0.5em 0; page-break-after: avoid; }
h3 { font-size: 11pt; font-weight: bold; margin: 1.3em 0 0.4em 0; page-break-after: avoid; }
h4, h5, h6 { font-size: 10.5pt; font-weight: bold; margin: 1.1em 0 0.3em 0; page-break-after: avoid; }
.titlepage { page-break-before: always; text-align: center; padding-top: 30%; }
.titlepage h1 { border: none; font-size: 30pt; page-break-before: auto; margin-bottom: 0.4em; }
.titlepage .subtitle { font-size: 14pt; font-style: italic; margin: 0 8% 2em 8%; line-height: 1.4; }
.titlepage .edition { font-size: 10pt; margin-top: 4em; color: #444; }
.part { page-break-before: always; text-align: center; padding-top: 28%; }
.part .partlabel { font-size: 12pt; letter-spacing: 0.25em; text-transform: uppercase; color: #555; }
.part h1 { border: none; font-size: 26pt; page-break-before: auto; margin-top: 0.3em; }
.partintro h1 { border-bottom: 1px solid #999; }
.toc h1 { page-break-before: always; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { margin: 0.15em 0; }
.toc li.part { text-align: left; padding: 0; page-break-before: auto; margin-top: 1em; font-weight: bold; }
.toc li.chapter { padding-left: 1.5em; }
.toc a { color: #111; text-decoration: none; }
.toc li { display: flex; justify-content: space-between; gap: 1em; }
.toc li.chapter a { padding-left: 1.5em; }
.toc .pg { flex: none; font-variant-numeric: tabular-nums; }
.editorial { font-size: 9.3pt; color: #333; border-top: 1px dotted #999; border-bottom: 1px dotted #999;
             padding: 0.5em 0; margin: 0 0 1.2em 0; }
blockquote { margin: 0.8em 0 0.8em 1.5em; padding-left: 0.8em; border-left: 2px solid #ccc; color: #222; }
.charter { border: 1px solid #999; padding: 0.8em 1.1em 0.3em 1.1em; margin: 1.2em 0 1.4em 0;
           background: #f7f6f2; font-size: 10pt; }
.charter p { text-align: left; }
.charter blockquote { border-left-color: #999; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 8pt; margin: 0.8em 0;
        page-break-inside: auto; }
th, td { border: 1px solid #bbb; padding: 2px 4px; vertical-align: top; word-wrap: break-word;
         overflow-wrap: anywhere; text-align: left; }
th { background: #eee; }
tr { page-break-inside: avoid; }
.appendix table { font-size: 7.2pt; }
.footnote { font-size: 8.6pt; margin-top: 1.5em; }
.footnote hr { border: none; border-top: 1px solid #aaa; width: 30%; margin-left: 0; }
.footnote ol { padding-left: 1.4em; }
.footnote li { margin-bottom: 0.25em; }
.footnote-backref { font-size: 8pt; }
sup { font-size: 7.5pt; line-height: 0; }
pre { white-space: pre-wrap; font-size: 8pt; background: #f4f4f4; padding: 0.5em; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
a { color: #111; text-decoration: none; }
.note { font-size: 9.5pt; color: #333; }
"""


def md_to_html(text: str) -> str:
    md = markdown.Markdown(
        extensions=["footnotes", "tables", "fenced_code", "md_in_html", "smarty", "sane_lists"],
        extension_configs={"footnotes": {"BACKLINK_TEXT": "↩", "PLACE_MARKER": "///Footnotes Go Here///"}},
        output_format="html5",
    )
    return md.convert(text)


def toc_page_map() -> dict:
    """Map each Contents entry's heading text to its PDF page, from a first print pass."""
    if not OUT_PDF.exists():
        return {}
    txt = subprocess.run(["pdftotext", "-layout", str(OUT_PDF), "-"], capture_output=True, text=True).stdout
    pages = txt.split("\f")
    pagemap = {}
    # skip the front matter (title, note, contents) when locating headings; keys are
    # structural ("Chapter 12", "Part III", "Appendix B", "Introduction"), which survive
    # the typographic changes (curly quotes, dashes, wrapping) that break exact titles
    for i, p in enumerate(pages, 1):
        if i <= 4:
            continue
        for line in p.splitlines():
            s = line.strip()
            m = re.match(r"^(Chapter \d+|Appendix [A-D]):", s)
            if m and m.group(1) not in pagemap:
                pagemap[m.group(1)] = i
            else:
                # the divider page label is letter-spaced, which pdftotext renders as spaced capitals
                mp = re.match(r"^P\s*A\s*R\s*T\s+((?:[IV]\s*)+)$", s, flags=re.IGNORECASE)
                numeral = re.sub(r"\s", "", mp.group(1)).upper() if mp else None
                if numeral in PART_ORDER and f"Part {numeral}" not in pagemap:
                    pagemap[f"Part {numeral}"] = i
                elif s == "Introduction: The School of Death" and "Introduction" not in pagemap:
                    pagemap["Introduction"] = i
    return pagemap


def build_html(pagemap: dict | None = None) -> dict:
    pagemap = pagemap or {}
    parts = load_roster()
    sections = []          # html fragments
    toc = []               # (kind, number, title, anchor)
    chapter_number = 0

    # Title page
    sections.append(f"""
<div class="titlepage">
  <h1>The School of Death</h1>
  <div class="subtitle">Charters of Protestant Seminaries and Missionary Societies,<br>
  from Calvin's Geneva to the Faith Missions</div>
  <div class="edition">Privately circulated edition. Not for sale.<br>Assembled September 2026.</div>
</div>
""")

    # Note on this edition
    note_md = """# A Note on This Edition

This book was assembled from the founding documents of fifty Protestant schools, seminaries,
and missionary societies, and from the histories that reprint those documents. Every quotation
in it was copied from a source that was actually opened and read, and then checked, character
by character, against that source before the book was assembled. Where a founding document
could not be found, or could be found only in a scan too damaged to read, the chapter says so
in plain words rather than supplying a likely-sounding substitute. Where two sources disagree,
both readings are given and neither is chosen.

Each chapter follows the same plan: the founding, the charter itself (abridged, with every cut
marked by `[...]`), and then five questions asked of every institution: whom it admitted, what
it taught, how teachers and students lived together, what its founders said mattered most, and
where it sent its people and at what cost. The notes at the end of each chapter give the page
and the source for every fact. The four appendices set the fifty institutions side by side.

This is a private, non-commercial edition. Quotations from works still in copyright, chiefly
the published writings of Martyn Lloyd-Jones and of Jim and Elisabeth Elliot, are brief and
are used for study and comment only. The complete research record behind the book, including
the transcribed sources, the quote-by-quote verification, and the log of every discrepancy
left open, accompanies the text and is the final authority wherever this edition and the record
differ.
"""
    sections.append(md_to_html(note_md))

    # Introduction
    intro = strip_to_first_heading((SYNTH / "introduction.md").read_text(encoding="utf-8"))
    intro = namespace_footnotes(book_language(intro), "intro")
    sections.append('<div id="introduction">' + md_to_html(intro) + "</div>")
    toc.append(("intro", None, "Introduction: The School of Death", "introduction"))

    for part in PART_ORDER:
        pid = f"part-{part}"
        sections.append(f"""
<div class="part" id="{pid}">
  <div class="partlabel">Part {part}</div>
  <h1>{PART_TITLES[part]}</h1>
</div>
""")
        toc.append(("part", part, PART_TITLES[part], pid))
        pi = SYNTH / f"part-intro-{part}.md"
        if pi.exists():
            pit = strip_to_first_heading(pi.read_text(encoding="utf-8"))
            pit = namespace_footnotes(book_language(pit), f"p{part}")
            sections.append('<div class="partintro">' + md_to_html(pit) + "</div>")
        for row in parts.get(part, []):
            chapter_number += 1
            cid = f"ch-{chapter_number}"
            text = build_chapter(row, chapter_number, part)
            title_line = re.search(r"^# (Chapter \d+: .+)$", text, flags=re.MULTILINE)
            title = title_line.group(1) if title_line else f"Chapter {chapter_number}: {row['name']}"
            sections.append(f'<div class="chapter" id="{cid}">' + md_to_html(text) + "</div>")
            toc.append(("chapter", chapter_number, title, cid))

    # Appendices
    appendices = [("A", "comparative_tables.md", "Comparative Tables"),
                  ("B", "lineage_chart.md", "Lineage Chart"),
                  ("C", "timeline.md", "Timeline"),
                  ("D", "glossary.md", "Glossary")]
    for letter, fname, title in appendices:
        text = strip_to_first_heading((SYNTH / fname).read_text(encoding="utf-8"))
        text = re.sub(r"```mermaid.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"^# .+$", f"# Appendix {letter}: {title}", text, count=1, flags=re.MULTILINE)
        text = namespace_footnotes(book_language(text), f"app{letter}")
        aid = f"appendix-{letter}"
        sections.append(f'<div class="appendix" id="{aid}">' + md_to_html(text) + "</div>")
        toc.append(("appendix", letter, f"Appendix {letter}: {title}", aid))

    # Table of contents (placed after the note, before the introduction)
    toc_items = []
    for kind, num, title, anchor in toc:
        cls = "part" if kind == "part" else ("chapter" if kind == "chapter" else "top")
        label = f"Part {num}: {title}" if kind == "part" else title
        key = {"intro": "Introduction", "part": f"Part {num}", "chapter": f"Chapter {num}",
               "appendix": f"Appendix {num}"}[kind]
        pg = f'<span class="pg">{pagemap[key]}</span>' if key in pagemap else ""
        toc_items.append(f'<li class="{cls}"><a href="#{anchor}">{label}</a>{pg}</li>')
    toc_html = '<div class="toc"><h1>Contents</h1><ul>' + "\n".join(toc_items) + "</ul></div>"
    sections.insert(2, toc_html)

    html_doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<title>The School of Death</title><style>" + CSS + "</style></head><body>"
                + "\n".join(sections) + "</body></html>")
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    words = len(re.sub(r"<[^>]+>", " ", html_doc).split())
    print(f"wrote {OUT_HTML} ({len(html_doc)/1e6:.1f} MB, ~{words} words, {chapter_number} chapters)")
    return {"chapters": chapter_number, "words": words}


# --------------------------------------------------------------------------- pdf via DevTools

def build_pdf():
    import websocket  # websocket-client
    port = 9333
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={port}", "--remote-allow-origins=*", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("chrome did not start")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], max_size=None, timeout=600)
        mid = 0

        def send(method, params=None, timeout=600):
            nonlocal mid
            mid += 1
            ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    return msg.get("result", {})

        send("Page.enable")
        send("Page.navigate", {"url": OUT_HTML.resolve().as_uri()})
        # wait for load
        deadline = time.time() + 300
        while time.time() < deadline:
            msg = json.loads(ws.recv())
            if msg.get("method") == "Page.loadEventFired":
                break
        time.sleep(2)
        footer = ('<div style="font-size:8.5px;width:100%;text-align:center;'
                  'font-family:\'DejaVu Serif\',serif;color:#444;">'
                  '<span class="pageNumber"></span></div>')
        params = {
            "printBackground": True,
            "displayHeaderFooter": True,
            "headerTemplate": "<div></div>",
            "footerTemplate": footer,
            "paperWidth": 8.27, "paperHeight": 11.69,
            "marginTop": 0.87, "marginBottom": 0.95, "marginLeft": 0.79, "marginRight": 0.79,
            "preferCSSPageSize": False,
            "generateDocumentOutline": True,
            "transferMode": "ReturnAsStream",
        }
        try:
            res = send("Page.printToPDF", params)
        except RuntimeError:
            params.pop("generateDocumentOutline")
            res = send("Page.printToPDF", params)
        stream = res["stream"]
        chunks = []
        while True:
            r = send("IO.read", {"handle": stream, "size": 1 << 20})
            data = r["data"]
            chunks.append(base64.b64decode(data) if r.get("base64Encoded") else data.encode("latin-1"))
            if r.get("eof"):
                break
        send("IO.close", {"handle": stream})
        OUT_PDF.write_bytes(b"".join(chunks))
        ws.close()
        print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size/1e6:.1f} MB)")
    finally:
        proc.terminate()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("html", "all"):
        build_html()
    if stage in ("pdf", "all"):
        build_pdf()
    if stage in ("toc", "all"):
        # second pass: add page numbers to the Contents from the first print, then reprint
        pm = toc_page_map()
        print(f"page map: {len(pm)} headings located")
        build_html(pm)
        build_pdf()
