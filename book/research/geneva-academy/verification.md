STATUS: VERIFIED
TICKET: V1-geneva-academy
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/borgeaud1900-v1.txt, sources.csv

# Verification report — geneva-academy (V1, dossier)

## 1. Script output
```
WARN geneva-academy-q002: before+text+after not contiguous in borgeaud1900-v1
WARN geneva-academy-q004: before+text+after not contiguous in borgeaud1900-v1
11/11 quote records passed; 0 failures
```
Two WARNs (q002, q004) are non-contiguous before/text/after windows (context slicing crossed a
running-head artifact in the OCR); the text, before, and after fields each independently verify
verbatim against the source file. Not treated as failures.

## 2. Quote-by-quote
All 11 quote_ids: text field verbatim match PASS. Spot-checked 4 of 11 (q001, q003, q005, q009)
by manual inspection against text/borgeaud1900-v1.txt at the stated line ranges: page/running-head
markers in the "page" field match what is visible in the OCR text. PASS.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4) is present.
Fields with quote_ids: summary sentence checked against the cited quote text; no summary claims
more than its quotes support. PASS for all cited fields.
Fields marked NOT FOUND or UNVERIFIED: 26 of 50 fields. This is a large fraction — the dossier
is honest but thin, reflecting that only one document was fetched this session. Recorded, not a
verification failure (Rule 8: UNVERIFIED is a success, not a defect).

## 4. Secondary citations checked
F7 and L6 (partial) cite Borgeaud's narrative prose as [SECONDARY] without a formal quote_id
(inline reference to line numbers instead). This does not meet the template's normal standard
(every fact needs a quote_id). FLAGGED for a revision pass: either formalize as quote records or
downgrade to UNVERIFIED. Recommend downgrading for now.

## 5. Dates and numbers vs known-facts sheet
- 5 June 1559 inauguration: matches book/plan/02_institution_roster.md known-facts sheet [HIGH].
- Beza as first rector: matches.
- "88 pastors sent to France" figure from the roster's known-facts sheet: NOT confirmed by any
  source fetched this session (see discrepancies.md item 2). The roster itself already marks
  this [VERIFY]. Chapter must not state the number without a fetched citation.
- "School of death" origin: roster already marks this [UNVERIFIED origin]; this session's search
  is consistent with that and adds a specific negative-search result (see discrepancies.md item 1).

## 6. Verdict
VERIFIED (zero FAILs on the 11 quote records; the field-level gaps are honestly marked, not
verification failures). RETURNED items for a future revision pass: formalize or downgrade F7/L6
secondary citations (item 4 above).
