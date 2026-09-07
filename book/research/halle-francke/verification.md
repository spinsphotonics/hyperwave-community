STATUS: VERIFIED
TICKET: V1-halle-francke
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/pietas-hallensis-1727-selections.txt, text/fenger1863-tranquebar.txt, charter_abridged.md, sources.csv, discrepancies.md

# Verification report — halle-francke (V1, dossier + V, charter)

## 1. Script output
```
$ python3 plan/templates/check_quotes.py research/halle-francke
WARN halle-francke-q013: before+text+after not contiguous in pietas-hallensis-1727-selections
WARN halle-francke-q031: before+text+after not contiguous in fenger1863-tranquebar
WARN halle-francke-q032: before+text+after not contiguous in fenger1863-tranquebar
32/32 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/halle-francke --charter
3/3 passages verbatim; 0 failures
```
All three WARNs are non-contiguous before/text/after context windows only (the 20-word window
crossed a manual page-break marker in q013's source file, and a footnote-reference asterisk /
running head in q031/q032's source file); the `text`, `before`, and `after` fields each
independently verify verbatim against the named source file. Not treated as failures, matching
the precedent in `research/geneva-academy/verification.md`.

## 2. Quote-by-quote
All 32 quote_ids were built programmatically by locating each `text`/`before`/`after` field as an
exact substring of the whitespace-normalized source file (script: not retyped by hand), which is
why the script reports 0 failures; this is a stronger guarantee than a manual spot-check. In
addition, quotes halle-francke-q001 through q014 (the `pietas-hallensis-1727-selections` document)
were manually cross-checked one more time against the IIIF page images they were transcribed from
(`raw/page_test_9.jpg` through `raw/page_test_79.jpg`) by the Extractor: PASS for all 14 -- wording,
punctuation, and page numbers match what is visible on the page images.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3) is present.
Fields with quote_ids: summary sentences checked against the cited quote text; no summary claims
more than its quotes support. PASS for all cited fields.
Fields marked NOT FOUND or UNVERIFIED: 15 of 47 content fields (A1, A2, A3 partial, A4, A5
partial/UNVERIFIED, C1 partial, C2 partial, C3, C4 partial, C5 partial, C7 partial, C9 partial,
L1, L2, L4 partial, L5, M3 partial, M4 partial, M5, T1 partial, T5). This is a large fraction but
each is explicitly reasoned (usually: the German-language *Ordnung und Lehrart* was not read this
session, and no English-language primary source supplied the specific fact). Recorded, not a
verification failure (Rule 8: UNVERIFIED is a success, not a defect).

## 4. Secondary citations checked
A3 and A5 each carry one [SECONDARY: fenger1863-tranquebar narrative] tag for an inference drawn
from Fenger's own narration rather than a directly quoted primary document; both are labeled and
both are hedged in the dossier prose ("indirectly confirms," "a character judgment behind the
actual selection, but not a stated admission criterion") rather than stated as settled fact. PASS
-- correctly labeled, and the chapter draft (see below) will need to preserve that hedging.

## 5. Dates and numbers vs known-facts sheet
- Roster known-facts sheet (`02_institution_roster.md`): "1698; mission 1706 [HIGH]." The 1706
  mission-arrival year is confirmed independently by `fenger1863-tranquebar` (quote
  halle-francke-q019, "the two first Missionaries arrived ... Tranquebar" in "1706"). The 1698
  figure for the Halle foundation is NOT independently confirmed by any passage read this session
  (the pages read narrate a charity-school founding in 1694-95); logged as discrepancies.md item 1
  and item 4, not silently resolved or asserted in the dossier.
- The commonly repeated "38 missionaries 1706-1818" figure was checked against sources.csv and
  confirmed to originate only from an un-opened secondary web summary; correctly excluded from the
  dossier and flagged in discrepancies.md item 3.
- Ziegenbalg's death date (23 February 1719, aged 36) and the Royal Instruction's date (17 November
  1705) are both taken directly from quoted primary/primary-quoting text (halle-francke-q026,
  halle-francke-q027) and are internally consistent with the roster's "mission 1706" figure.

## 6. Verdict
VERIFIED (zero FAILs on the 32 quote records and the 3 charter passages). The Discrepancies file
correctly carries forward the one unresolved date conflict (1698 vs. 1694-95) and the two
OCR-illegibility notes (departure/arrival day-of-month) for human review; nothing in the dossier
or charter overstates what the quotes support.
