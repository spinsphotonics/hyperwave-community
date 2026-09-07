#!/usr/bin/env python3
"""Concatenate the manuscript in reading order: front matter, Introduction,
Part introductions + chapters in roster chapter_order, then appendices.
Strips each file's internal working-document header (STATUS/TICKET/ROLE/
INPUTS READ block and any coverage-note/word-count-table preamble) down to
its actual title heading, since those are working metadata, not reader
content.
"""
import csv
import re
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
CHAPTERS = BOOK / "chapters"
SYNTH = BOOK / "synthesis"
OUT = BOOK / "assembly" / "manuscript.md"

PART_TITLES = {
    "I": "Part I: Reformation Academies (1518-1620)",
    "II": "Part II: Puritans, Pietists and Dissenters (1636-1790)",
    "III": "Part III: The Missionary Awakening and the Seminary (1792-1860)",
    "IV": "Part IV: Faith Missions and the Bible Institute (1865-1930)",
    "V": "Part V: Translators, Tribes and the Auca Five (1930-1960)",
    "VI": "Part VI: Lloyd-Jones and the Reformed Recovery (1938-1980)",
}

def strip_chapter_header(text: str) -> str:
    m = re.search(r"^# Chapter", text, flags=re.MULTILINE)
    if m:
        return text[m.start():].rstrip() + "\n"
    return text.rstrip() + "\n"

def strip_synth_header(text: str) -> str:
    # drop everything before the first real markdown heading (the working
    # STATUS/TICKET/ROLE/INPUTS READ block, however many lines it wraps to)
    m = re.search(r"^# ", text, flags=re.MULTILINE)
    if m:
        return text[m.start():].rstrip() + "\n"
    return text.rstrip() + "\n"

def load_roster():
    rows = []
    with open(BOOK / "roster.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    roster = load_roster()
    parts = {}
    for r in roster:
        if r["tier"] == "C":
            continue  # appendix-only, handled separately
        parts.setdefault(r["part"], []).append(r)
    for p in parts:
        parts[p].sort(key=lambda r: int(r["chapter_order"]))

    out = []
    out.append("# The School of Death\n")
    out.append("## Charters of Protestant Seminaries and Missionary Societies, "
                "from Calvin's Geneva to the Faith Missions\n")
    out.append("\n*Assembled manuscript -- working draft. See `book/plan/07_risk_register_and_qa.md` "
                "for the acceptance checklist this draft has not yet cleared (rights sign-off, "
                "human review of the Introduction, Appendix A charter selection).*\n")
    out.append("\n---\n\n# Table of Contents\n")
    out.append("\n- Introduction: The School of Death\n")
    for part_num in ["I", "II", "III", "IV", "V", "VI"]:
        out.append(f"- {PART_TITLES[part_num]}\n")
        for r in parts.get(part_num, []):
            out.append(f"  - {r['name']} ({r['founded']})\n")
    out.append("- Appendix B: Comparative Tables\n")
    out.append("- Appendix C: Lineage Chart\n")
    out.append("- Appendix D: Timeline\n")
    out.append("- Appendix E: Glossary\n")
    out.append("\n---\n\n")

    intro = (SYNTH / "introduction.md").read_text(encoding="utf-8")
    out.append(strip_synth_header(intro))
    out.append("\n\n---\n\n")

    missing_chapters = []
    for part_num in ["I", "II", "III", "IV", "V", "VI"]:
        part_intro_path = SYNTH / f"part-intro-{part_num}.md"
        if part_intro_path.exists():
            out.append(strip_synth_header(part_intro_path.read_text(encoding="utf-8")))
        else:
            out.append(f"# {PART_TITLES[part_num]}\n\n*Part introduction not yet written.*\n")
        out.append("\n\n---\n\n")
        for r in parts.get(part_num, []):
            slug = r["slug"]
            matches = sorted(CHAPTERS.glob(f"{part_num}-*-{slug}.md"))
            if not matches:
                missing_chapters.append(slug)
                out.append(f"\n\n*[Chapter for {r['name']} not found on disk -- {slug}]*\n\n")
                continue
            chapter_text = matches[0].read_text(encoding="utf-8")
            out.append(strip_chapter_header(chapter_text))
            out.append("\n\n---\n\n")

    # Appendices B-E: the synthesis files, header-stripped
    for label, fname in [("B", "comparative_tables.md"), ("C", "lineage_chart.md"),
                           ("D", "timeline.md"), ("E", "glossary.md")]:
        p = SYNTH / fname
        out.append(f"# Appendix {label}\n\n")
        out.append(strip_synth_header(p.read_text(encoding="utf-8")))
        out.append("\n\n---\n\n")

    OUT.write_text("".join(out), encoding="utf-8")
    print(f"Wrote {OUT} ({sum(len(x.split()) for x in out)} words, approx)")
    if missing_chapters:
        print("MISSING CHAPTERS:", missing_chapters)
    else:
        print("All roster chapters found and included.")

if __name__ == "__main__":
    main()
