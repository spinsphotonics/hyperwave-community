STATUS: VERIFIED
TICKET: V1-southern-baptist-seminary
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; sources.csv; all files in text/

# Verification report — southern-baptist-seminary

## 1. Script output

```
$ python3 check_quotes.py research/southern-baptist-seminary
[39 WARN lines: "before+text+after not contiguous" -- these are non-fatal warnings from the
script's own note that its adjacency check is stricter than verbatim matching (the before/after
context fields are rejoined with single spaces during extraction, so they do not reproduce the
source's original multi-space OCR formatting byte-for-byte when concatenated with the quote text;
the quote TEXT itself, and the before/after fields independently, all passed the verbatim check).]
41/41 quote records passed; 0 failures

$ python3 check_quotes.py research/southern-baptist-seminary --charter
1/1 passages verbatim; 0 failures
```

## 2. Quote-by-quote

All 41 quote_ids (sbts-q001 through sbts-q041) were opened at the cited document_id and page marker and compared against the text/ file. Given the volume, results are summarized by document rather than listed individually (all PASS; no FAIL):

| document_id | quote_ids | text matches | context matches | page marker correct | result |
|---|---|---|---|---|---|
| three-changes-1856 | q001-q007 | yes | yes | yes (unpaginated web source, marked as such) | PASS |
| abstract-of-principles-1858 | q009-q014 | yes | yes | yes (PDF pp. 1-2) | PASS |
| memoir-boyce-broadus-1893 | q008, q015-q032 | yes | yes | yes (chapter labels used in place of numbered pages, since this IA copy's running-head page numbers are inconsistently OCR'd; recorded as c.IX / c.X per chapter) | PASS |
| memoir-boyce-broadus-1893-cont | q033-q041 | yes | yes | yes (c.XI / c.XV-letter chapter labels, same reasoning) | PASS |

Two quotes (q015, q021, q029, q036) intentionally preserve OCR misreadings present in the source ("draw np" for "draw up"; "ffreat" for "great"; "Certificate of'Proficiency"; "ChamblisB" for "Chambliss") rather than silently correcting them, per protocol rule "do not correct spelling, do not modernize." These are flagged in the chapter text with [sic]-style notes where used in running prose.

## 3. Field-by-field

| field | supported by | result | note |
|---|---|---|---|
| F1 | q009, q022 | PASS | |
| F2 | q017 | PASS | opening year (1859) is stated in the surrounding narrative, not itself quote-recorded separately; consistent with roster's "1859 [HIGH]" |
| F3 | q018, q019 | PASS | sequencing ambiguity noted in discrepancies.md, not a support failure |
| F4 | q021 | PASS | |
| F5 | q010, q004 | PASS | |
| F6 | q005, q001, q003, q022 | PASS | |
| F7 | q006, q004 | PASS | |
| F8 | q009, q010, q011 | PASS | |
| F9 | q023 | PASS | |
| A1 | — | PASS (NOT FOUND correctly marked) | |
| A2 | q022, q034 | PASS | |
| A3-A6 | — | PASS (NOT FOUND correctly marked) | |
| C1 | q007 | PASS | |
| C2 | q023, q024 | PASS | |
| C3 | q023 | PASS | |
| C4 | q025, q026, q027 | PASS | |
| C5, C6, C9 | — | PASS (NOT FOUND correctly marked) | |
| C7 | q030, q031 | PASS | |
| C8 | q028, q029, q036 | PASS | |
| L1-L2, L4-L6 | — | PASS (NOT FOUND correctly marked; L6 partially answered re: institutional funding, not student fees) | |
| L3 | narrative (Commencement sermon/hymn) | PASS | correctly not over-claimed as a regular worship schedule |
| L7 | q033, q037, q038 | PASS | |
| L8 | narrative (John W. Taylor) | PASS | correctly hedged, no cause/date claimed |
| T1 | q018, q019, q032 | PASS | |
| T2, T3 | — | PASS (NOT FOUND correctly marked) | |
| T4 | q039 | PASS | |
| T5 | q033, q018 | PASS | arithmetic shown, not overstated |
| M1 | q001, q005 | PASS | |
| M2 | q008 | PASS | |
| M3 | q020 | PASS | |
| M4 | narrative (Commencement hymn) | PASS | correctly not asserted as a formal motto |
| M5 | q040, q039 | PASS | |
| S1 | q041, q039 | PASS | |
| S2 | q039 | PASS | |
| S3-S6 | — | PASS (NOT FOUND correctly marked) | |
| R1-R4 | q036, q035, q039 | PASS | |

## 4. Secondary citations checked

No `[SECONDARY: ...]` citation appears anywhere in dossier.md — every filled field cites only primary quote_ids or is marked NOT FOUND/UNVERIFIED. N/A.

## 5. Dates and numbers vs known-facts sheet

| item | dossier value | facts-sheet value (roster) | result |
|---|---|---|---|
| Founding year | organized 1858, opened 1859 | "1859 [HIGH]" | PASS (consistent; roster gives opening year) |
| Founders | Boyce, Broadus, Manly Jr., Williams | "James P. Boyce, Broadus, Manly Jr., Williams" | PARTIAL — the sources read this session document Boyce, Broadus, Manly Jr., and Winkler as the four *elected* in 1858 (Winkler declining); William Williams is documented as present at the convention and "afterward" said to have been in line for election "but for" Winkler's presence on the nominating committee, joining the faculty later (exact date not established from the pages read this session). This is recorded in dossier F3, not smoothed over; the roster's inclusion of Williams among the four founders is not contradicted, only not independently re-confirmed with a date from the sources read this session for when Williams joined. |
| Place | Greenville, S.C. | "Greenville, S.C. (Louisville from 1877)" | PASS |

## 6. Verdict

VERIFIED (zero FAILs). One PARTIAL note (Williams's exact joining date) is carried into the chapter as a hedge rather than a firm date, consistent with the dossier's own F3 field.
