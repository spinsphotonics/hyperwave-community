STATUS: VERIFIED
TICKET: V1-college-of-new-jersey
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; text/maclean1877-v1.txt; sources.csv

# Verification report — college-of-new-jersey

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/college-of-new-jersey
46/46 quote records passed; 0 failures
(WARN lines for "before+text+after not contiguous" appear on most records: these are an
artifact of the checker inserting a forced space when concatenating before+text+after for its
combo test, which does not match the source's own punctuation spacing in many places, e.g. a
quote ending right before a comma or closing quotation mark. Each of the three fields — text,
before, after — was independently confirmed to appear verbatim in the source; only the
checker's own combo-join is affected. This is a WARN, not a FAIL, and does not block
verification.)

$ python3 plan/templates/check_quotes.py research/college-of-new-jersey --charter
9/9 passages verbatim; 0 failures
```

## 2. Quote-by-quote

All 46 quote records (cnj-q001 through cnj-q046) were built by direct programmatic substring
extraction from `text/maclean1877-v1.txt` (a Python script located each quotation by an anchor
phrase in the whitespace-normalized document text and read off the exact 20-word windows before
and after), then re-verified by `check_quotes.py`. Every record: text matches, context matches,
page marker matches the nearest legible printed page number in the running heads (noted as
approximate where OCR garbled the numeral, per the header note in text/maclean1877-v1.txt).
Result: PASS for all 46.

Five records (cnj-q004, cnj-q005, cnj-q013, cnj-q018, cnj-q026) initially failed on a first pass
because of stray extra spaces introduced when the "after" field was hand-transcribed from a
debug print; each was corrected by re-extracting the exact substring from the source file and
re-run through the checker until it passed. This is recorded here rather than silently fixed,
per the project's transparency norm — no other record required correction.

## 3. Field-by-field

| field | supported by | result | note |
|---|---|---|---|
| F1 | cnj-q011, cnj-q020 | PASS | "later Princeton University" is explicitly NOT stated as a chapter fact, flagged unverified |
| F2 | cnj-q003, cnj-q007, cnj-q008, cnj-q017-019 | PASS | |
| F3 | cnj-q006, cnj-q017 | PASS | |
| F4 | cnj-q017, cnj-q018, cnj-q046, cnj-q020 | PASS | |
| F5 | cnj-q007 (loss of 1746 text) | PASS | |
| F6 | cnj-q001, cnj-q002, cnj-q004, cnj-q005 | PASS | |
| F7 | cnj-q014, cnj-q016 | PASS | |
| F8 | cnj-q009, cnj-q010 | PASS | summary does not overclaim a specific confession requirement, which was not found |
| F9 | cnj-q015 | PASS | Synod-of-Philadelphia 1739 school project noted but explicitly marked "not directly claimed by the College's own founders as their model" |
| A1 | — | PASS (NOT FOUND stated) | |
| A2 | cnj-q025, cnj-q034 | PASS | |
| A3 | cnj-q009, cnj-q010 | PASS | |
| A4 | cnj-q025, cnj-q034, cnj-q035 | PASS | |
| A5, A6 | — | PASS (NOT FOUND stated) | |
| C1-C9 | cnj-q027 through cnj-q032, cnj-q045 | PASS | |
| L1-L8 | cnj-q019, cnj-q032, cnj-q036, cnj-q038, cnj-q045 | PASS | L5, L6 marked NOT FOUND correctly |
| T1-T5 | cnj-q044, cnj-q045 | PASS | T5 marked NOT FOUND correctly |
| M1-M5 | cnj-q001, cnj-q021-024, cnj-q033, cnj-q037, cnj-q043 | PASS | M4 marked NOT FOUND correctly |
| S1-S6 | cnj-q021, cnj-q022, cnj-q039, cnj-q040 | PASS | S4-S6 marked NOT FOUND correctly; S3's Dickinson-era 5-of-6 figure is explicitly labeled a Maclean tabulation (secondary), not directly quoted |
| R1-R4 | cnj-q039, cnj-q040, cnj-q041, cnj-q042 | PASS | |

No summary sentence was found to claim more than its cited quote(s) support.

## 4. Secondary citations checked

No `[SECONDARY: ...]` citation appears in this dossier; every filled field cites either a
PRIMARY quote_id from `maclean1877-v1` (Maclean's own narrative sentences used to state a fact
are, per the dossier's sourcing note, treated as the connective secondary layer around
block-quoted primary material, and are flagged inline as such where used, e.g. F9's Synod note
and S3's graduate tabulation) or is marked NOT FOUND.

## 5. Dates and numbers vs known-facts sheet

| item | dossier value | roster/known-facts value | result |
|---|---|---|---|
| Founding (first charter) | 22 October 1746 | "1746 charter" [HIGH] | MATCH |
| Second charter | 14 September 1748 | "1748 second charter" [HIGH] | MATCH |
| Founders | Dickinson, Burr (named on roster) | Dickinson, Burr both confirmed as founding Trustees/Presidents (cnj-q006, cnj-q017) | MATCH |
| Place | Elizabeth (opening), then Newark, then Princeton | roster: "Elizabeth, Newark, then Princeton" | MATCH (cnj-q017/018 Elizabethtown; cnj-q046 Newark; cnj-q020 Princeton/Nassau Hall) |
| Not conflated with Princeton Theological Seminary (1812) | This chapter covers only the 1746/1748 undergraduate College; no Princeton Seminary material used | Risk register item B | MATCH — checked explicitly; no 1811/1812 Plan or Archibald Alexander material appears anywhere in this dossier |

## 6. Verdict

VERIFIED (zero FAILs).
