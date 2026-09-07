STATUS: VERIFIED
TICKET: V1-westminster-seminary
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; sources.csv; all files in text/

# Verification report — westminster-seminary

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/westminster-seminary
WARN westminster-seminary-q019: before+text+after not contiguous in rian1940-ch4
WARN westminster-seminary-q035: before+text+after not contiguous in hart-muether-pt1
39/39 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/westminster-seminary --charter
3/3 passages verbatim; 0 failures
```

Both WARNs are punctuation-adjacency artifacts, not FAILs: q019's quoted text ends with a closing parenthesis immediately followed in the source by a comma with no space ("...Dr. Allis),"), and q035's quoted text is immediately preceded and followed by curly quotation marks with no space ("...1929, "Princeton..."" alive."" The). The check script's contiguity check inserts a space at the join when concatenating before+text+after, which the source punctuation does not have; the "text", "before", and "after" fields themselves each independently verified against the source text (the PASS/FAIL check), so this does not affect the record's validity.

## 2. Quote-by-quote

All 39 records in `quotes.jsonl` (westminster-seminary-q001 through q039) were checked by the script above and individually re-opened against their named `text/` file. All match verbatim, with page markers as recorded (`[[excerpt A/B/C]]` for the three Machen-address excerpt files reflecting that those pages are not separately paginated in the secondary web sources; `[[p. body]]` for the three prose secondary sources, which are likewise unpaginated web pages).

| quote_id range | document_id | text matches | context matches | page correct | result |
|---|---|---|---|---|---|
| q001-q005 | machen-pcahistory | yes | yes | yes (unpaginated source, marked accordingly) | PASS |
| q006-q007 | machen-wscal | yes | yes | yes | PASS |
| q008-q012 | machen-opctoday | yes | yes | yes | PASS |
| q013-q034 | rian1940-ch4 | yes | yes (q019 WARN, see above) | yes | PASS |
| q035-q036 | hart-muether-pt1 | yes | yes (q035 WARN, see above) | yes | PASS |
| q037-q039 | wikipedia-wts | yes | yes | yes | PASS |

## 3. Field-by-field (V1)

| field | supported by | result | note |
|---|---|---|---|
| F1 | q039 + PRIMARY institutional usage | PASS | |
| F2 | q013,q014,q016,q018,q012 | PASS | |
| F3 | q017,q020,q037,q036 | PASS | |
| F4 | q012,q019,q029 | PASS | |
| F5 | q001-q011 (composite) | PASS | correctly flags BLOCKED status of full original |
| F6 | q001,q002,q008,q009,q025 | PASS | |
| F7 | q013,q015,q026 | PASS | |
| F8 | q010,q023 | PASS | |
| F9 | q008,q009,q020,q037 | PASS | |
| A1-A4, A6 | none | PASS | correctly marked NOT FOUND |
| A5 | wikipedia-wts (partial sentence) | PASS | correctly flagged as later-period, UNVERIFIED for 1929 |
| A2 (denominational note) | q024 | PASS | |
| C1-C3, C5-C7, C9 | none / partial | PASS | correctly marked NOT FOUND or partial with caveat |
| C4 | q003,q004 | PASS | |
| C8 | q030,q031 | PASS | |
| L1, L3-L5, L8 | none | PASS | correctly marked NOT FOUND |
| L2, L6 | q019 | PASS | |
| L7 | q012,q018,q031 | PASS | |
| T1 | q020,q021,q022 | PASS | |
| T2-T4 | none | PASS | correctly marked NOT FOUND |
| T5 | derived from q018,q020 | PASS | explicitly labeled as derived/not directly quoted |
| M1 | q006,q007,q011 | PASS | |
| M2 | q003,q004,q005,q006 | PASS | |
| M3 | q001,q002 + Rian narrative | PASS | contested claim (MacRae) correctly flagged as disputed, not settled |
| M4 | q038,q039 | PASS | date of adoption correctly marked UNVERIFIED |
| M5 | Rian narrative (Kuiper quotation summarized) | PASS | |
| S1 | q033 + Rian narrative | PASS | |
| S2, S3 | q031,q032 | PASS | |
| S4-S6 | none | PASS | correctly marked NOT FOUND |
| R1-R6 | q028,q034,q021,q020 + Rian narrative | PASS | |

No field claims more than its cited quotes support. Every NOT FOUND / UNVERIFIED / BLOCKED marker in the dossier is honestly the result of an actual search that did not locate the material this session, not an unexamined gap.

## 4. Secondary citations checked

All `[SECONDARY: ...]` and `[PRIMARY-OR-SECONDARY: ...]` citations in the dossier resolve to a row in `sources.csv` and a file in `text/`. The Rian (`rian1940-ch4`) and Hart & Muether (`hart-muether-pt1`) secondary sources were each read in full (not excerpted before use) and their content checked against the dossier's summaries; neither is asked to support more than it states. The Wikipedia citations were limited to background facts (founders' names, motto, campus history) and were not used for any claim central to the founding narrative that a primary excerpt could instead support.

## 5. Dates and numbers vs. known-facts sheet

The roster's known-facts sheet (`02_institution_roster.md`) gives, for `westminster-seminary`: founding year 1929 [HIGH], founder(s) "J. Gresham Machen and others leaving Princeton," place Philadelphia. All three are consistent with the dossier (F2, F3, F4). The roster names no other specific date or number to check against (unlike the Geneva Academy entry, this institution has no dedicated known-facts checklist item beyond the summary row itself).

| item | dossier value | roster value | result |
|---|---|---|---|
| founding year | 1929 (multiple dated steps, June-Sept 1929) | 1929 [HIGH] | MATCH |
| founder(s) | Machen + named executive committee and founding faculty | "J. Gresham Machen and others leaving Princeton" | MATCH |
| place | Philadelphia | Philadelphia | MATCH |

## 6. Verdict

VERIFIED (zero FAILs). Two WARNs from the automated script were investigated and found to be punctuation-adjacency artifacts, not substantive errors (see section 1). The chief limitation of this dossier, disclosed throughout rather than concealed, is that the principal founding document (Machen's address) survives here only as three verbatim excerpts (~1,059 of an estimated ~2,000-2,500 words in the original four-page article) recovered from secondary transcriptions, not as a complete primary text fetched directly; this is recorded as BLOCKED in `discrepancies.md` #3 and in `sources.csv`, and no dossier field or chapter sentence claims knowledge of the missing portions.
