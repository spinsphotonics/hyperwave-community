STATUS: VERIFIED
TICKET: V1-omf
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; text/*.txt; sources.csv

# Verification report -- omf

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/omf
21/21 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/omf --charter
FAIL: charter_abridged.md does not name a document_id in backticks
```

The quotes.jsonl check is a clean PASS: all 20 quote records' `text`, `before`, and `after` fields were found verbatim, and contiguous, in their named `text/` files.

The `--charter` check FAILs, but this is the expected and documented outcome of a genuine block, not a quote-fidelity error: `charter_abridged.md` states plainly (see its own STATUS line and body) that no primary post-1964 document was located or opened this session, so there is no document_id to name and no verbatim passages to check. Re-running the script against `research/china-inland-mission` (the chapter this coda extends) continues to pass its own `--charter` check unaffected; nothing in this session touched that folder except reading it. This FAIL is therefore accepted as correct and is not corrected by inventing a document_id or a passage.

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| omf-q001 | PASS | PASS | n/a (web page, no pagination) | PASS |
| omf-q002 | PASS | PASS | n/a | PASS |
| omf-q003 | PASS | PASS | n/a | PASS |
| omf-q004 | PASS | PASS | n/a | PASS |
| omf-q005 | PASS | PASS | n/a | PASS |
| omf-q006 | PASS | PASS | n/a | PASS |
| omf-q007 | PASS | PASS | n/a | PASS |
| omf-q008 | PASS | PASS | n/a | PASS |
| omf-q009 | PASS | PASS | n/a | PASS |
| omf-q010 | PASS | PASS | n/a | PASS |
| omf-q011 | PASS | PASS | n/a | PASS |
| omf-q012 | PASS | PASS | n/a | PASS |
| omf-q013 | PASS | PASS | n/a | PASS |
| omf-q014 | PASS | PASS | n/a | PASS |
| omf-q015 | PASS | PASS | n/a | PASS |
| omf-q016 | PASS | PASS | n/a | PASS |
| omf-q017 | PASS | PASS | n/a | PASS |
| omf-q018 | PASS | PASS | n/a | PASS |
| omf-q019 | PASS | PASS | n/a | PASS |
| omf-q020 | PASS | PASS | n/a | PASS |
| omf-q021 | PASS | PASS | n/a | PASS |

All web-page sources carry no printed pagination; `page` is recorded as "n/a (web page)" for every record, consistent across quotes.jsonl and sources.csv.

## 3. Field-by-field

| field | supported by | result | note |
|---|---|---|---|
| F1 | omf-q007, omf-q008, omf-q009, omf-q012, omf-q014, omf-q020 | PASS | Summary states only what the quotes give: CIM 1865-1964, an unresolved-date interim compound name, OMF 1964-1993, OMF International from 1993. |
| F2 | omf-q002, omf-q007, omf-q008, omf-q010, omf-q011 | PASS | Summary explicitly flags the 1954/1964 conflict rather than silently picking one; matches discrepancies.md item 1. |
| F3 | omf-q017, omf-q018, omf-q019, omf-q020 | PASS | |
| F4 | omf-q005, omf-q012, omf-q014 | PASS | |
| F5 | (none -- NOT FOUND) | PASS | Correctly marked NOT FOUND with the search documented, per Rule 1. |
| F6 | omf-q003, omf-q015 | PASS | Both labeled SECONDARY as required for a PO field filled only by secondary material. |
| F7 | omf-q003 | PASS | Labeled SECONDARY. |
| F8 | (none -- NOT FOUND) | PASS | |
| F9 | (NOT APPLICABLE, coda) | PASS | Correctly distinguished from NOT FOUND; a reason is given. |
| A1-A6, C1-C9, L1-L8, T1-T5 | (NOT APPLICABLE, coda) | PASS | Each points to the CIM dossier rather than asserting continuity as a new fact; no claim is smuggled in. |
| M1 | omf-q016 | PASS | Labeled SECONDARY, and further flagged that the secondary source's own citation (Broomhall 1984) was not independently checked this session. |
| M2 | omf-q004, omf-q005, omf-q006, omf-q021 | PASS | |
| M3 | (none -- NOT FOUND) | PASS | |
| M4 | (none -- NOT FOUND) | PASS | |
| M5 | James Hudson Taylor III 1989 quotation, source's own [citation needed] | PASS | Correctly marked UNVERIFIED rather than presented as fact; explicitly excluded from the chapter. This is a case worth flagging in this report: it would have been the single most quotable sentence found this session ("We can never forget that we came into existence as the China Inland Mission"), and it was deliberately not used, per Rule 1. |
| S1 | omf-q006, omf-q013 | PASS | |
| S2 | context of omf-q001/omf-q002, omf-q015 | PASS | |
| S3, S4, S5, S6 | (none -- NOT FOUND) | PASS | |
| R1-R4 | omf-q013, omf-q017, omf-q018, omf-q019, omf-q020 | PASS | R3 (Taylor III) correctly limited to the sourced list entry only, excluding the unverified 1989 quotation. |

No field's summary sentence claims more than its cited quotes support.

## 4. Secondary citations checked

| citation | page says what is claimed? | result |
|---|---|---|
| aim25-cim-omf-finding-aid, all cited passages | Yes -- re-read against text/aim25-cim-omf-finding-aid.txt in full | PASS |
| wikipedia-omf-international, all cited passages | Yes -- re-read against text/wikipedia-omf-international.txt in full | PASS |
| wikipedia-jo-sanders, all cited passages | Yes -- re-read against text/wikipedia-jo-sanders.txt in full | PASS |

## 5. Dates and numbers vs known-facts sheet

The roster's known-facts sheet (`02_institution_roster.md` section 3) has no dedicated entry for `omf` beyond the roster row itself, which reads "Renamed 1964/1965 [VERIFY exact year]." This dossier's F2 resolves that flag to 1964, sourced independently from two documents (AIM25 and Wikipedia, the latter in three of its four internal mentions), and separately records the one conflicting internal date (Wikipedia's own "14 October 1954" chronology line) rather than silently discarding it. No other roster fact touches this slug.

## 6. Verdict

**VERIFIED** (zero FAILs in the quote-accuracy and field-support checks). The one script FAIL, under `--charter`, is a correctly documented BLOCKED condition (no primary document available to abridge), not a quote or field error, and is explained in section 1 above and in `charter_abridged.md`'s own STATUS line.
