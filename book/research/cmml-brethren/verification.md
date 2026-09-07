STATUS: VERIFIED
TICKET: V1-cmml-brethren
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/*.txt (all six documents), sources.csv, discrepancies.md

# Verification report — cmml-brethren (V1, dossier)

## 1. Script output
```
WARN cmml-brethren-q001: before+text+after not contiguous in groves-christian-devotedness-1829
WARN cmml-brethren-q003: before+text+after not contiguous in groves-christian-devotedness-1829
WARN cmml-brethren-q024: before+text+after not contiguous in cmml-what-we-believe
27/27 quote records passed; 0 failures
```
The three WARNs are non-contiguous before/text/after window checks only; each of the three
fields (`text`, `before`, `after`) independently verifies verbatim against its named source file
(the script's own per-field checks above the WARN line report no FAIL for any of the three). As
with the geneva-academy precedent, this is not treated as a failure.

## 2. Quote-by-quote
All 27 quote_ids: `text` field verbatim match PASS (confirmed by check_quotes.py). Manually
spot-checked 8 of 27 (q001, q002, q009, q013, q016, q017, q022, q026) by opening the named
text/ file and confirming the "page" field's description matches the surrounding content.
PASS for all 8 spot-checked.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4) is present. Fields
with quote_ids: the summary sentence was checked against the cited quote text; no summary claims
more than its quotes support. PASS for all cited fields.

Fields marked NOT FOUND: the C (Curriculum) and T (Teachers) sections are almost entirely NOT
FOUND, with an explanatory note that this reflects the Brethren "no formal seminary" position
described in the roster, not a research gap. This is judged an honest and correctly-labeled
result, not a defect (Rule 8).

One field (F9, the Groves-to-Müller influence claim) was deliberately written as UNVERIFIED
rather than as fact, because the only source for it was a WebSearch results summary rather than
a document this session actually opened and read. PASS — this is exactly the discipline Rule 1
requires.

## 4. Secondary citations checked
Fields citing `cmml-whoweare`, `cmml-whatwedo`, `cmml-our-history`, `cmml-commendation`, and
`cmml-what-we-believe` are labeled (PS) throughout, since these are the organization's own
current website rather than an archival founding-era document; this matches the `is_primary = no`
rows in sources.csv for those five document_ids. `groves-christian-devotedness-1829` fields are
labeled (PO), matching its `is_primary = yes` row. Cross-check: PASS, no field marked
PRIMARY-ONLY (PO) cites a document_id whose sources.csv row has `is_primary = no`.

## 5. Dates and numbers vs known-facts sheet
- 1921 CMML incorporation, 1971 merger: no conflicting figure appears in
  `02_institution_roster.md`'s Part V table or its known-facts sheet (the sheet does not carry a
  separate cmml-brethren entry beyond the roster row itself, which already marks the 1921 date
  "[VERIFY]"). This session's fetched source (cmml-our-history) independently states 1921 for the
  Ltd. incorporation and 1971 for the Inc. merger, consistent with the roster's own [VERIFY]'d
  figure. Recorded as consistent, not as independently authoritative.
- Jim Elliot's death (8 January 1956) and the Ecuador mission: matches the known-facts sheet
  ("Elisabeth and Jim Elliot" subsection) and is independently corroborated by a Wheaton College
  archives source (see S1, citing wheaton-college-q023), not merely asserted from the roster.
- No date or number in this dossier contradicts the known-facts sheet; two items (headquarters
  location; Groves-Müller connection) are recorded in discrepancies.md as sourcing gaps rather
  than contradictions.

## 6. Verdict
VERIFIED (zero FAILs on the 27 quote records; the field-level gaps in the C and T sections are
honestly and correctly marked as not applicable/NOT FOUND per Rule 8, not verification failures).
