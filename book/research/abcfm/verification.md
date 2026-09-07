STATUS: VERIFIED
TICKET: V1-abcfm
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/firstten1834.txt, text/anderson1861.txt, sources.csv, discrepancies.md

# Verification report — abcfm (V1, dossier)

## 1. Script output
```
WARN abcfm-q001: before+text+after not contiguous in firstten1834
WARN abcfm-q003: before+text+after not contiguous in firstten1834
WARN abcfm-q004: before+text+after not contiguous in firstten1834
WARN abcfm-q005: before+text+after not contiguous in firstten1834
WARN abcfm-q006: before+text+after not contiguous in firstten1834
WARN abcfm-q008: before+text+after not contiguous in firstten1834
WARN abcfm-q012: before+text+after not contiguous in anderson1861
WARN abcfm-q013: before+text+after not contiguous in anderson1861
14/14 quote records passed; 0 failures
```
All 14 quote records pass. WARNs are page-break running heads interposed between before/text/
after (e.g. "40 INSTRUCTIONS TO MISSIONARIES. 181*2." breaking a sentence); each field
independently verified verbatim. Not treated as failures (see geneva-academy precedent).

`check_quotes.py --charter` also run: `11/11 passages verbatim; 0 failures`.

## 2. Quote-by-quote
All 14 quote_ids: `text` field verbatim PASS per script. Manually spot-checked 5 of 14
(abcfm-q001, abcfm-q004, abcfm-q009, abcfm-q011, abcfm-q014) by reading the surrounding
lines directly in the two text/ files: each sits in the article/section named in its "page"
field. PASS.

## 3. Field-by-field
Every dossier field group is present, including the honestly-marked NOT APPLICABLE/NOT FOUND
blocks for Admission, Curriculum, and most of Common Life and Teachers (the ABCFM is a
sending board, not a school, per the dossier's opening note). No summary sentence claims
more than its cited quotes support. PASS for all cited fields.
F8 (doctrinal basis) is marked NOT FOUND for a creedal-subscription clause, correctly
distinguishing the ABCFM's practical/pastoral Instructions from Andover's confessional
Constitution. PASS as an honest field.

## 4. Secondary citations checked
F7 cites anderson1861 as [SECONDARY: anderson1861, p. 41] for the framing of the Board's
origin as a felt personal call among Andover students; the cited page (via quote_id
abcfm-q011) supports the claim as summarized, without overclaiming an institutional-crisis
narrative not present in the source. PASS.

## 5. Dates and numbers vs known-facts sheet
- ABCFM organized 29 June 1810 at Bradford, Mass. [HIGH per roster known-facts sheet]:
  consistent with this dossier's F2, which quotes Anderson's own retrospective use of that
  same date ("From the first appointment of the Board, at Bradford, June 29, 1810...").
- Judson ordained 6 Feb. 1812 at Salem; sailed 19 Feb. 1812 [HIGH per roster]: consistent
  with this dossier's F2/S3, cross-referenced to the `judson-triennial-convention` dossier
  for the fuller ordination/sailing account (this dossier's own primary focus is the
  Instructions dated 7 Feb. 1812, one day after the ordination -- no contradiction: the
  ordination and the Instructions are two distinct events one day apart, both attested).
- No other date or number in this dossier conflicts with the known-facts sheet.

## 6. Verdict
VERIFIED (zero FAILs on the 14 quote records and zero FAILs on the 11 abridged-charter
passages; field-level NOT APPLICABLE/NOT FOUND markers are honest, not defects).
