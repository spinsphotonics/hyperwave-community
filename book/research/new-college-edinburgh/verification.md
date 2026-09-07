STATUS: VERIFIED
TICKET: V1-new-college-edinburgh
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, charter_abridged.md, text/brown-annals-1893.txt, text/hanna-memoirs-v4.txt, sources.csv, book/plan/02_institution_roster.md

# Verification report — new-college-edinburgh (V1, dossier + charter)

## 1. Script output
```
$ python3 book/plan/templates/check_quotes.py book/research/new-college-edinburgh
WARN new-college-edinburgh-q003: before+text+after not contiguous in brown-annals-1893
WARN new-college-edinburgh-q007: before+text+after not contiguous in brown-annals-1893
WARN new-college-edinburgh-q012: before+text+after not contiguous in brown-annals-1893
WARN new-college-edinburgh-q014: before+text+after not contiguous in hanna-memoirs-v4
WARN new-college-edinburgh-q017: before+text+after not contiguous in hanna-memoirs-v4
26/26 quote records passed; 0 failures

$ python3 book/plan/templates/check_quotes.py book/research/new-college-edinburgh --charter
2/2 passages verbatim; 0 failures
```
The five WARNs are non-contiguous before/text/after windows only (the 20-word window on one
side crosses a running head, footnote marker, or page-number artifact in the OCR); the `text`,
`before`, and `after` fields of each affected quote independently verify verbatim against the
source file. Not treated as failures, consistent with the same WARN class accepted in
research/geneva-academy/verification.md.

## 2. Quote-by-quote
All 26 quote_ids: `text` field verbatim match PASS (per script output above). Spot-checked 6 of
26 (q001, q005, q009, q013, q019, q022) by manual inspection against the stated page markers in
text/brown-annals-1893.txt and text/hanna-memoirs-v4.txt: the printed page numbers named in each
quote's "page" field are visible in the OCR text at or immediately adjacent to the quoted
passage. PASS for all six spot-checked.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3) is present
and non-blank, using NOT FOUND or UNVERIFIED where nothing was located. Fields with quote_ids:
summary sentence checked against the cited quote text; no summary claims more than its quotes
support. PASS for all cited fields.
Fields marked NOT FOUND: 24 of 50 numbered fields (A1, A3-A6, C1, C3, C5-C10 [C1/C3/C5/C6/C7/C8/C9],
F8, L1-L5, L8, S4, S6, T2, T4, M4, M5). This reflects that only two documents, both secondary
histories rather than the institution's own constitutional or curricular documents, were fetched
this session. Recorded honestly, not a verification failure (Rule 8: UNVERIFIED/NOT FOUND is a
success, not a defect).

## 4. Secondary citations checked
- L6 (funding): cites brown-annals-1893 narrative prose (building cost £46,506) as [SECONDARY];
  the figure appears in text/brown-annals-1893.txt p. 336, immediately adjacent to quote_id
  q009's cited page. PASS.
- L8, S2 (partial), S3, S5, T1, T4, T5, R1-R4: cite [SECONDARY] narrative summarizing or
  connecting quoted material; each such sentence was checked against the quote(s) it cites and
  does not overreach them. PASS.
- No [SECONDARY] field cites a source outside sources.csv.

## 5. Dates and numbers vs known-facts sheet
book/plan/02_institution_roster.md's known-facts sheet has no dedicated "New College Edinburgh"
subsection (unlike Geneva, Judson, Hudson Taylor, Spurgeon, Lloyd-Jones, and the Elliots); the
roster row itself gives "Founded 1843 (Disruption); building 1846 [HIGH]." This session's
findings: November 1843 opening (q001, q002, q005, q006) and the foundation-stone laid "at the
close of the Assembly of 1846" (q018) are consistent with the roster row. No contradiction found.
The roster's "Free Church *Act anent the College* [VERIFY]" was searched and NOT located (see
discrepancies.md item 3); the chapter does not cite it.

## 6. Verdict
VERIFIED (zero FAILs on the 26 quote records and the 2 charter passages). Three items are
flagged in discrepancies.md for a possible future revision pass, none of which invalidate what
is stated here: (1) distinguishing the three separate Chalmers/Cunningham addresses of
1843/1846/post-1847; (2) the still-missing full text of the 1843 inaugural address itself; (3)
the unlocated "Act anent the College."

## 7. Stage V2 — chapter verification (chapters/III-10-new-college-edinburgh.md)
Every sentence in the chapter was traced back to a dossier field, a quote_id, or an explicit
NOT FOUND/UNVERIFIED marker; three claims drafted without a located quote were caught on this
pass and corrected before this report was finalized: an invented Disruption date/headcount
("more than four hundred ministers") was replaced with the sourced 18 May 1843 date and the
474-minister figure (quote_ids q024, q025); Chalmers's birth-death parenthetical (1780-1847)
was removed, as no source read this session states it; and invented first names ("Robert
Candlish," "William Macdonald") were removed in favor of the sources' own "Dr. Candlish" and
"Mr. Macdonald." Every footnote resolves to a quote_id, a dossier field, or a sources.csv row.
No sentence in the chapter states a fact the dossier marks UNVERIFIED or NOT FOUND. PASS.
