STATUS: VERIFIED
TICKET: V1-princeton-seminary
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; sources.csv; text/plan1811.txt; text/inaug1812.txt; text/biocatalogue1932.txt

# Verification report — princeton-seminary

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/princeton-seminary
34/34 quote records passed; 0 failures
(24 WARN lines: "before+text+after not contiguous")

$ python3 plan/templates/check_quotes.py research/princeton-seminary --charter
8/8 passages verbatim; 0 failures
```

Note on the WARN lines: `check_quotes.py` reconstructs a contiguity-check string as `f"{before} {text} {after}"`, always inserting a literal space between the three fields. Whenever a quotation's `text` field starts or ends immediately at a punctuation mark that is *not* preceded/followed by a space in the original 1811/1812 printing (e.g. "...United States of America." running straight into "And to the intent..." with no space before the period), that hard-coded space makes the reconstructed string diverge from the source even though `before`, `text`, and `after` are each independently exact, contiguous substrings of the source file (verified by direct substring search for every WARN'd quote_id, see section 2). This is a property of the checker's fixed-format reconstruction, not an error in the quote records. It does not affect the checker's pass/fail verdict (WARN is not counted in `failures`).

## 2. Quote-by-quote

All 34 quote_ids in `quotes.jsonl` (princeton-seminary-q001 through q034): `text`, `before`, and `after` each individually confirmed present, verbatim (whitespace-normalized), in the named `text/` file, by direct Python substring search re-run against the current file contents (not just the original generation script) as part of this verification pass. Page markers were assigned by direct inspection of the raw OCR line numbers (for plan1811, cross-checked against the standalone folio lines visible in `raw/plan1811_djvu.txt`; for inaug1812, against the inline "( N )" folios; for biocatalogue1932, against front-matter and running-head labels) rather than solely by automated inference, after an initial automated pass produced two mismatched pages that were manually corrected (Article II Sect. 8-9 boundary, corrected from an undetected "p. 8" to the correct "p. 9" once the OCR's non-digit-prefixed "■9" folio line was accounted for).

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| q001-q021 (plan1811) | PASS | PASS | PASS (manually cross-checked against raw line numbers, see above) | PASS |
| q022, q023, q027, q028 (inaug1812, narrative) | PASS | PASS | PASS (inline "(N)" folios read directly) | PASS |
| q024, q025, q026 (inaug1812, Milledoler's Charge) | PASS | PASS | PASS (p. 108 / p. 100 read directly from inline folios) | PASS |
| q029-q031 (biocatalogue1932, chronology) | PASS | PASS | labeled "front matter" (chronology table precedes the book's own numbered pagination in the fetched OCR; not independently foliated) | PASS |
| q032-q034 (biocatalogue1932, faculty roster) | PASS | PASS | labeled by running head ("FACULTY", roman-numeral pages per the printed volume's front-matter numbering; exact roman numeral not independently confirmed against a page image this session) | PASS (text/context exact; page label approximate, flagged) |

## 3. Field-by-field (dossier.md)

Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3) was re-read against the quote_ids it cites.

- F1: PASS — supported by q001.
- F2: PASS — supported by q022, q029, q030, q031; the three-date sequence is presented descriptively, not asserted as a single uncontested date, and cross-references discrepancies.md #1.
- F3: PASS — supported by q020, q028, q032, q033; Milledoler's authorship of the Charge is stated correctly (not attributed to Miller), consistent with discrepancies.md #2.
- F4: PASS — supported by plan1811 closing minutes (paraphrase, no separate quote_id required beyond the general reading; the Raritan/Potowmac and College-of-New-Jersey-conference content is separately quoted at q021).
- F5: PASS — matches sources.csv row `plan1811`.
- F6: PASS — supported by q001, q002, q003, q004; no clause is claimed beyond what these quotes contain.
- F7: PASS — supported by q023, q027; the summary sentence explicitly flags that the specific "same causes" in q027 are not further quoted, avoiding overclaiming.
- F8: PASS — summarized without a separate quote_id for the full professor's subscription formula (the formula was read in text/plan1811.txt during extraction; the dossier's summary does not exceed what Article III Sect. 3's text, as read, supports). Flagged for a future pass to add a dedicated quote_id if the chapter needs to quote the formula directly.
- F9: PASS — supported by q021, q023; correctly framed as a proposed conference/link, not a merger, consistent with discrepancies.md #3 and the risk register.
- A1-A6: PASS — A1 and A5 correctly marked NOT FOUND rather than guessed; A2, A3, A4, A6 supported by q015, q016, q017.
- C1-C9: PASS — C5 and C9 correctly marked NOT FOUND rather than guessed; C1-C4, C6-C8 supported by q006-q010, q019.
- L1-L8: PASS — L1, L2, L7, L8 correctly marked NOT FOUND / not independently counted rather than estimated; L3-L6 supported by q011-q013, q018 and paraphrase of Article VIII Sect. 6 (read, not separately quote-recorded).
- T1-T5: PASS — T4 correctly marked NOT FOUND; T1-T3, T5 supported by q006, q014, q020, q026, q032, q033.
- M1-M5: PASS — M4 correctly marked NOT FOUND; M5 correctly limited to the 1812 pieces only (Calhoun/Moorhead not fetched, per sources.csv); M1-M3 supported by q005, q014, q024, q025.
- S1-S6: PASS — S2, S3, S5, S6 correctly marked NOT FOUND rather than guessed; S1, S4 supported by q004, q010 and correctly state the Seminary's own charter does not itself license, ordain, or address field support.
- R1-R4: PASS — supported by q023, q028, q032, q033, q034; R4 explicitly states the catalogue was not exhaustively tallied, avoiding an invented headcount.
- X1-X3: PASS — epigraph candidate (q005) and its text both verified against the source.

## 4. Secondary citations checked

- `biocatalogue1932` citations (marked PS throughout): the front-matter chronology (q029-q031) and faculty roster (q032-q034) were re-opened in `text/biocatalogue1932.txt` and confirm the claimed content. This document is treated as an official institutional register (published by the Seminary's own Trustees, compiled by its Registrar from Seminary records) rather than a third-party narrative history, per the note in sources.csv; no dossier field marked PRIMARY-ONLY (PO) cites it (see section 6).
- Calhoun and Moorhead (roster's "Secondary" cell): not fetched this session (see sources.csv rows `calhoun-princeton-seminary`, `moorhead-secondary`); no dossier field cites either, so there is nothing to check here. Flagged as a follow-up for a future session with library/database access.

## 5. Dates and numbers vs known-facts sheet

| item | dossier value | roster known-facts value | result |
|---|---|---|---|
| Founding/Plan date | Plan adopted May 1811 (F2, q030) | "1812; *Plan* adopted by General Assembly 1811 [HIGH]" | MATCH |
| Opening date | 12 August 1812 (F2, q022, q031) | "1812" | MATCH |
| First professor | Archibald Alexander, inaugurated 12 Aug 1812 (F3, q028, q032) | "Archibald Alexander first professor" | MATCH |
| Institution distinct from College of New Jersey | F4, F9, discrepancies.md #3 | Risk register section B: "two separate institutions; two roster rows" | MATCH — the chapter's Founding section (see chapters/III-07-princeton-seminary.md) states this distinction explicitly rather than leaving it implicit. |
| Additional 1810 date (biocatalogue1932) | Recorded as an additional catalogue-sourced data point, not asserted as the headline founding date (F2, q029) | Roster known-facts sheet does not mention 1810 | Logged in discrepancies.md #1, not silently adopted or discarded. |

## 6. PRIMARY-ONLY fields checked against secondary sources

Every field marked (PO) in dossier.md was checked to confirm it cites only `plan1811` and/or `inaug1812` (both primary: a constitutional Plan and a published inauguration-address volume), never `biocatalogue1932`. Confirmed: no (PO) field cites `biocatalogue1932`; the catalogue is cited only in fields marked (PS) (F2, F3, L7, L8, T1, T5, R1-R4), consistent with the template's rule.

## 7. Confidence labels

Every dossier field carries a PO/PS type marker in its heading (per the template) and every filled field's content is followed by a bracketed `[PRIMARY: ...]` or `[PRIMARY: ...; PS: ...]` label, or is explicitly marked `NOT FOUND (searched: ...)` / `UNVERIFIED`. No field is stated without a label. PASS.

## 8. Verdict

VERIFIED (zero FAILs from `check_quotes.py` in either mode; zero unsupported claims found in the field-by-field and sentence-level review above). One approximate item is flagged, not a FAIL: the exact roman-numeral page citations for q032-q034 (biocatalogue1932 faculty roster) are read from the running head "FACULTY" rather than from a directly visible page number in the OCR text at that point, and are noted as approximate in section 2 above; a human with access to a page-imaged copy could tighten this citation in a later pass.
