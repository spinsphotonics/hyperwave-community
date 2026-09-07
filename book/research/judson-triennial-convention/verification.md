STATUS: VERIFIED
TICKET: V1-judson-triennial-convention
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/proceedings1814.txt, text/wayland1853v1.txt, sources.csv, discrepancies.md

# Verification report — judson-triennial-convention (V1, dossier)

## 1. Script output
```
WARN judson-triennial-convention-q001: before+text+after not contiguous in wayland1853v1
WARN judson-triennial-convention-q002: before+text+after not contiguous in wayland1853v1
WARN judson-triennial-convention-q006: before+text+after not contiguous in wayland1853v1
WARN judson-triennial-convention-q007: before+text+after not contiguous in wayland1853v1
WARN judson-triennial-convention-q008: before+text+after not contiguous in wayland1853v1
WARN judson-triennial-convention-q010: before+text+after not contiguous in proceedings1814
WARN judson-triennial-convention-q012: before+text+after not contiguous in proceedings1814
15/15 quote records passed; 0 failures
```
All 15 quote records pass. WARNs are page-break running heads or inline OCR page numbers
interposed mid-sentence; each field independently verified verbatim. Not treated as
failures (consistent with the geneva-academy precedent for this project).

`check_quotes.py --charter` also run: `1/1 passages verbatim; 0 failures` (the 1814
Constitution is printed whole, under 1,200 words, per abridgement rule 8, so it is a single
unbroken passage).

## 2. Quote-by-quote
All 15 quote_ids: `text` field verbatim PASS per script. Manually spot-checked 5 of 15
(q001, q004, q005, q009, q011) by reading the surrounding lines directly in the two text/
files: each sits in the document/page named in its "page" field, and (for the proceedings1814
quotes) the article numbering (I, II, V) matches what is printed in the source. PASS.

## 3. Field-by-field
Every dossier field group is present, including the honestly-marked NOT APPLICABLE blocks
for Admission (partial), Curriculum (partial), Common Life, and Teachers -- the Convention is
a sending body, not a school, per the dossier's opening note. No summary sentence claims more
than its cited quotes support. PASS for all cited fields.
S3-S6 are largely NOT FOUND/PARTIAL, honestly marked with the documents searched. PASS.

## 4. Secondary citations checked
F9 cites wayland1853v1 narrative pp. 121-126 as [SECONDARY] for the general claim that
multiple independent local Baptist societies formed before the 1814 Convention unified them;
this narrative claim is consistent with, and not contradicted by, the directly quoted material
(q008, on the "dormant energies" being awakened) used alongside it. PASS.
R2 and R4 similarly carry brief [SECONDARY] narrative details (Rice's organizing tour;
Baldwin's earlier magazine editorship) alongside their PRIMARY quote_id citations. PASS.

## 5. Dates and numbers vs known-facts sheet
- Judson ordained 6 Feb. 1812 at Salem; sailed 19 Feb. 1812 with Ann Hasseltine Judson [HIGH
  per roster]: this dossier's F2 quotes the Panoplist's account of the 6 February ordination
  (q001) and Wayland's account of the Caravan's sailing (q002, dated "the 19th instant" per
  the surrounding narrative, consistent with 19 February). PASS, no contradiction.
- Judson and Ann became Baptists and were baptized at Calcutta by William Ward, Sept 1812
  [HIGH per roster]: this dossier's F2/R1 confirm the baptism-by-immersion fact (q003) but
  do NOT independently confirm the officiant's name (William Ward) or the exact September
  date from a quoted passage this session -- flagged honestly in discrepancies.md item 2,
  not asserted beyond what was quoted. This is a gap in this session's verification, not a
  contradiction of the known-facts sheet.
- The Baptist General Convention (Triennial Convention) organized May 1814 in Philadelphia
  [HIGH per roster]: this dossier's F1/F2/F9 (q009, q010, q011) confirm this exactly,
  including the specific date (18 May 1814) and location (First Baptist Church).

## 6. Verdict
VERIFIED (zero FAILs on the 15 quote records and zero FAILs on the abridged-charter passage;
the one known-facts-sheet detail not independently re-verified -- William Ward as baptizer --
is honestly flagged in discrepancies.md rather than silently asserted).
