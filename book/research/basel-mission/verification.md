STATUS: VERIFIED
TICKET: V1-basel-mission
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, charter_abridged.md, text/ratschiller-chapter2023.txt, text/ratschiller-webinar2025.txt, sources.csv

# Verification report — basel-mission (V1, dossier + charter)

## 1. Script output
```
$ python3 plan/templates/check_quotes.py research/basel-mission
WARN basel-mission-q015: before+text+after not contiguous in ratschiller-chapter2023
WARN basel-mission-q020: before+text+after not contiguous in ratschiller-chapter2023
WARN basel-mission-q024: before+text+after not contiguous in ratschiller-webinar2025
24/24 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/basel-mission --charter
2/2 passages verbatim; 0 failures
```
The three WARNs are non-fatal contiguity warnings, as in the `geneva-academy` and
`serampore-college` precedents: each of the text/before/after fields independently verifies
verbatim; the warning fires only where a footnote number or a deliberately shortened quotation
breaks the exact contiguous slice the WARN check looks for. Not treated as failures.

## 2. Quote-by-quote
All 24 quote_ids: `text` field verbatim match PASS (confirmed by script, 0 failures). Spot-checked
8 of 24 (q001, q005, q007, q008, q012, q017, q019, q020) by manual inspection against
text/ratschiller-chapter2023.txt: each reproduces Ratschiller's own wording and internal
quotation marks exactly, including her smart-quote characters. PASS.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1) is present. Fields with
quote_ids: summary sentences checked against the cited quote text; no summary claims more than
its quotes support. PASS for all cited fields.
Fields marked NOT FOUND: 24 of 45 fields — more than half. This reflects the honest limit stated
throughout the dossier: only two English-language secondary works, read in one session, stand
between this dossier and the German-language Basel Mission Archive itself, and neither the
Hausordnung nor the Evangelisches Missions-Magazin was fetched this session (Rule 8: recorded, not
a defect).

## 4. Secondary citations checked
Multiple fields (F6, F7, F8, F9, A2, S2, S5, T3) cite Ratschiller's own narrative analysis as
[SECONDARY] rather than a primary quotation. This is consistent with the source material: this
chapter is itself a secondary scholarly analysis, so facts drawn from her interpretive
sentences (e.g. the "atonement for the transatlantic slave trade" reading of the founders'
motives, F7) are correctly labeled as her argument, not as the founders' own words, and the
chapter draft must not present them as if quoted from a founding document. PASS on labeling
discipline.

## 5. Dates and numbers vs known-facts sheet
- Basel Mission founded 1815: matches `02_institution_roster.md` [HIGH]. PASS.
- Seminary opened 1816: NOT independently confirmed by a source read this session (Ratschiller's
  chapter gives 1815 for "initiated a seminary," which may describe the founding decision rather
  than the first day of instruction). See discrepancies.md item 1. The chapter states only "1816,
  per the roster's known-facts sheet" rather than asserting Ratschiller's 1815 date as the
  seminary's opening.
- Blumhardt as first Inspector (1815/1816-1838): matches roster [HIGH], confirmed by
  quote_id basel-mission-q017. PASS.
- CMS connection ("supplied CMS with missionaries," per roster's "Why included" column):
  corroborated in detail by quote_id basel-mission-q004 (over 100 CMS missionaries trained at
  Basel, 1819-1858) and basel-mission-q024. PASS.

## 6. Verdict
VERIFIED (zero FAILs on the 24 quote records and the 2 charter passages). RETURNED items for a
future revision pass: obtain and translate the Hausordnung itself (discrepancies.md item 2);
resolve the network-egress block on bmarchives.org's search endpoint; fetch the Evangelisches
Missions-Magazin volumes already located on Internet Archive.
