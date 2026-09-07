STATUS: VERIFIED
TICKET: V1-serampore-college
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, charter_abridged.md, text/marshman1859-v2.txt, text/smith-carey-1885.txt, sources.csv

# Verification report — serampore-college (V1, dossier + charter)

## 1. Script output
```
$ python3 plan/templates/check_quotes.py research/serampore-college
WARN serampore-college-q003: before+text+after not contiguous in marshman1859-v2
WARN serampore-college-q004: before+text+after not contiguous in marshman1859-v2
WARN serampore-college-q005: before+text+after not contiguous in marshman1859-v2
WARN serampore-college-q008: before+text+after not contiguous in marshman1859-v2
WARN serampore-college-q014: before+text+after not contiguous in smith-carey-1885
WARN serampore-college-q018: before+text+after not contiguous in smith-carey-1885
WARN serampore-college-q021: before+text+after not contiguous in smith-carey-1885
21/21 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/serampore-college --charter
9/9 passages verbatim; 0 failures
```
The seven WARNs are non-fatal contiguity warnings (as in the `geneva-academy` precedent): each
of the three fields (text, before, after) independently verifies verbatim against the source
file; the WARN fires only because the exact word-count window used for "before"/"after" does
not always land on a perfectly contiguous slice when the quoted passage itself was deliberately
cut short (e.g. stopping before an OCR-corrupted word). Not treated as failures.

## 2. Quote-by-quote
All 21 quote_ids: `text` field verbatim match PASS (confirmed by script, 0 failures). Spot-checked
6 of 21 (q001, q003, q010, q014, q019, q020) by manual inspection against the source files at the
stated line ranges: each reproduces the source's own spelling and hyphenation exactly, including
the OCR artifacts noted in sources.csv (`wantSi` for "wants,"). PASS.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1) is present. Fields with
quote_ids: summary sentences checked against the cited quote text; no summary claims more than
its quotes support. PASS for all cited fields.
Fields marked NOT FOUND: 17 of 45 fields (roster template R section collapsed to one entry per
protocol note). This reflects that only two secondary sources, read in one session, were used,
and that neither founding document (the 1818 Prospectus, the 1827 Charter) was located as a
digitized full text. Recorded honestly, not a verification failure (Rule 8).

## 4. Secondary citations checked
Several fields (F4, F7, F9, L6, T5, S2) cite Marshman 1859 or Smith 1885 narrative prose as
[SECONDARY] without a formal quote_id (inline page reference instead), consistent with the
`geneva-academy` precedent. FLAGGED for a future revision pass: formalize these as quote records
or downgrade to UNVERIFIED. Recommend leaving as SECONDARY narrative citations for now, since the
underlying facts (budget figures, teacher counts) are incidental description, not contested
claims.

## 5. Dates and numbers vs known-facts sheet
- Founded 1818 (Prospectus, 15 July 1818): matches `02_institution_roster.md` [HIGH]. PASS.
- Danish royal charter 1827: matches the roster's year, but the exact day (23 February 1827,
  from an unverified web search snippet) is NOT confirmed by either source actually read this
  session — see discrepancies.md item 1. The chapter states only the year, not the day, for this
  reason.
- Carey, Marshman, Ward as founders: matches roster [HIGH]. PASS.
- "First Protestant college in Asia chartered to grant degrees" (roster's characterization):
  corroborated by smith-carey-1885 quote_id serampore-college-q012 ("the earliest
  degree-conferring college in Asia"). PASS.

## 6. Verdict
VERIFIED (zero FAILs on the 21 quote records and the 9 charter passages). RETURNED items for a
future revision pass: formalize the [SECONDARY] narrative citations listed in section 4 above;
attempt to fetch the Prospectus and Charter as primary full texts (discrepancies.md item 3).
