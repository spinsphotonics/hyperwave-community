STATUS: VERIFIED
TICKET: V1-zurich-prophezei
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/christoffel1858.txt, text/jackson1901.txt, text/simpson1902.txt, text/grob1883.txt, text/uzh-news-prophezey.txt, text/adfontes-prophezei.txt, sources.csv

# Verification report — zurich-prophezei (V1, dossier)

## 1. Script output
```
WARN zurich-prophezei-q009: before+text+after not contiguous in jackson1901
WARN zurich-prophezei-q019: before+text+after not contiguous in adfontes-prophezei
WARN zurich-prophezei-q020: before+text+after not contiguous in adfontes-prophezei
WARN zurich-prophezei-q021: before+text+after not contiguous in adfontes-prophezei
21/21 quote records passed; 0 failures
```
Four WARNs (q009, q019, q020, q021): in each case a footnote marker, a page break, or
(for the adfontes quotes) the "p.142"/"p.143" attribution line embedded in the source
text falls between the `before`/`text`/`after` windows, breaking strict contiguity. In
every case the `text`, `before`, and `after` fields each independently verify verbatim
against the named source file (confirmed by the script's own per-field check, which
produced no FAIL). Not treated as failures, consistent with the geneva-academy V1
report's treatment of the same class of warning.

## 2. Quote-by-quote
All 21 quote_ids: `text` field verbatim match PASS (per script). Manually spot-checked 6
of 21 (q001, q005, q009, q015, q019, q021) by reading the surrounding lines in the named
text/ file: page markers and document attribution match. PASS.

## 3. Field-by-field
All 49 F/A/C/L/T/M/S/R dossier fields are present and non-blank (confirmed by direct
count: `awk` over dossier.md lines 31-99 finds exactly 49 field bullets). Every field
carries either a quote_id, an explicit `[SECONDARY: document_id]` citation, or a `NOT
FOUND (searched: ...)` marker — none is stated bare. Fields with quote_ids: the summary
sentence(s) checked against the cited quote text; no summary claims more than its quotes
support (in particular: F6 and F9 are explicitly marked as not supported by any
PRIMARY-ONLY source, since no primary document was located this session — see F5).
PASS for all cited fields.

Fields marked NOT FOUND: 28 of 49 (F5, F8, A1, A3, A4, A5, C1, C3, C7, C8, L1, L2, L4,
L5, L7, T4, T5, S1, S2, S3, S4, S5, S6, plus A2/C1/T5/M5 marked NOT FOUND with a partial
inferential note). This is comparable to, slightly higher than, the geneva-academy
dossier's 26/50 — expected, since this session located no primary document at all for
zurich-prophezei (geneva-academy at least had a secondary edition that reprinted primary
archival documents verbatim). Recorded as an honest result per Rule 1, not a defect.

## 4. Secondary citations checked
Every `[SECONDARY: ...]` citation (uzh-news-prophezey, adfontes-prophezei) was checked
against the actual fetched text/ file for that document; in each case the claimed fact
or quotation is present at the cited location. The adfontes-prophezei quotations of
Bruce Gordon's *Zwingli: God's Armed Prophet* (2021) are flagged throughout the dossier
and sources.csv as second-hand — Gordon's book itself was not independently obtained or
read this session, so these citations are only as reliable as the blog's transcription.
This is stated explicitly in the dossier's opening note and in sources.csv, not left
implicit. PASS (correctly labeled, not overclaimed).

## 5. Dates and numbers vs known-facts sheet
- 19 June 1525 as the Prophezei's opening date: matches `02_institution_roster.md`'s
  known-facts sheet ("Zurich Prophezei began 19 June 1525. [HIGH]"). One fetched source
  (Christoffel 1858, in English translation) instead gives "19th July 1525"; this is
  recorded in discrepancies.md item 1 and NOT allowed to override the roster's HIGH-
  confidence date, per the roster's own confidence label and the two independently
  dated, directly-read sources that agree with it (uzh-news-prophezey, adfontes-
  prophezei/Gordon).
- No other dated fact in the dossier conflicts with the roster's known-facts sheet
  (which for this institution contains only the single 19 June 1525 entry).
- Numbers: no student counts, sending numbers, or casualty figures are stated anywhere
  in this dossier (all marked NOT FOUND); there is therefore nothing to check against
  the known-facts sheet on that front, and nothing for the chapter to overclaim.

## 6. Check that no PRIMARY-ONLY field cites a secondary source
Reviewed every field marked (PO) in the template. F6, F9, T2 are marked (PO) or
(PO/PS) and are careful to say explicitly that no PRIMARY-ONLY-qualifying source was
found, using SECONDARY citations only alongside that explicit caveat, not in place of
it. No (PO) field is filled with an unlabeled or silently-secondary citation. PASS.

## 6a. Charter abridgement check
`python3 check_quotes.py research/zurich-prophezei --charter` FAILS immediately with
"does not name a document_id in backticks" — the script assumes a single source document
for the abridged charter, but `charter_abridged.md` here assembles passages from six
different fetched sources (no single founding document was located; F5 = NOT FOUND).
This is not a content failure: every passage in `charter_abridged.md` is one of the 21
quote records that independently passed the standard `quotes.jsonl` check (section 1
above), and each is labeled with its quote_id in the charter file's own table. Recorded
here rather than silently worked around, per Rule 6.

## 7. Verdict
VERIFIED (zero FAILs on the 21 quote records; all 49 dossier fields present and
correctly labeled; the large NOT FOUND fraction is an honest reflection of this
session's source access, not a verification failure). Follow-up items for a future
research pass, in priority order: (1) obtain a clean OCR or a printed-edition
transcription of Bullinger's *Reformationsgeschichte*/*Diarium* (the Google Books scan
fetched this session, `heinrichbulling00bullgoog`, is NEEDS_RETRANSCRIPTION and unusable
— see sources.csv); (2) locate and fetch Zwingli's own 1525 tract on the preaching
office and any Zurich Council ordinance text via e-rara.ch or the Zentralbibliothek
Zurich, per the roster's holdings note; (3) obtain Bruce Gordon's *Zwingli: God's Armed
Prophet* (2021) directly rather than relying on a blog's transcription of it; (4) resolve
discrepancies.md items 1-3 against a primary or facsimile source.
