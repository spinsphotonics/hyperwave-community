STATUS: VERIFIED
TICKET: V1-emmanuel-cambridge
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/commdoc1852v3.txt, text/shuckburgh1904.txt,
text/fuller1655.txt, text/vch1959.txt, sources.csv, discrepancies.md

# Verification report — emmanuel-cambridge (V1, dossier)

## 1. Script output
```
38/38 quote records passed; 0 failures
```
No WARNs were produced (every quote's `before`+`text`+`after` window is contiguous in its source
file, unlike the geneva-academy dossier, which had two WARNs from OCR running-head artifacts).

## 2. Quote-by-quote
All 38 quote_ids: `text`, `before`, and `after` fields verbatim-match PASS per the script above.
Manually spot-checked 10 of 38 (emmanuel-cambridge-q001, q004, q006, q015, q017, q020, q023, q027,
q030, q033) by re-opening the cited line ranges in their `text/` files and confirming the quoted
text, the surrounding context, and the stated page/running-head marker all agree with what is
visible in the OCR (or, for vch1959, the rekeyed HTML text). PASS for all 10 spot-checked.
For the nine `commdoc1852v3` (Latin) quote records, each carries a `[New translation, draft]`
literal English rendering (`translation_source: NEW-DRAFT`) since no freely available published
translation was located this session (Stubbings 1983 is copyrighted — see sources.csv). Per
Procedure B rule 6, these translations should be checked by a competent reader of Latin before
publication; they are used in the dossier and chapter as working translations, clearly marked, not
as an authoritative published rendering. FLAGGED for human review, not a verification FAIL.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3) is present and
non-blank, per the template. Each summary sentence was checked against its cited quote_id(s): no
summary claims more than its quotes support. PASS for all cited fields.

Fields marked NOT FOUND: A1, A5, C1 (partial — no explicit fixed course-length figure), L8 (as a
collective founding-era fact; the individual Harvard death is recorded separately at R1, not
double-counted as answering L8), M4, S4, S6 — 6 of 49 template fields (using the template's own
count, in which R1-R4 is a single field). This is a markedly higher fill rate than the
geneva-academy dossier (24/50 filled there), reflecting that this session located an actual
scholarly reprint of the founder's original 1585 statutes (commdoc1852v3) with 40+ numbered
chapters, rather than only a secondary history's narrative paraphrase. The higher fill rate is a
genuine product of source richness, not of relaxed sourcing discipline — every filled field still
carries a quote_id or an explicit PS/PO label, and every one of the 6 empty fields is marked
NOT FOUND with the documents searched named, per Rule 1.

## 4. Secondary citations checked
- `[SECONDARY]`-labeled content: F1, F4, F7, F9 (partial), L7 (VCH corroboration), S2, S3, R1-R4
  (narrative facts drawn from Shuckburgh's and the VCH's own prose, not quotations of a primary
  document). Each such field's summary was checked against the cited quote_id and found to state no
  more than the quote supports. PASS.
- The M4 field's parenthetical note about the door inscription ("Sacrae Theologiae Studiosis...")
  is drawn from shuckburgh1904 but was deliberately NOT given its own quote_id, since the dossier
  field itself is marked NOT FOUND (no motto was located) and the inscription is offered only as a
  related fact, explicitly labeled as not answering the question asked. This is consistent with
  Rule 7 (no paraphrase inside quotation marks) since the inscription text is not placed in quotation
  marks in the dossier prose; it is presented as background. PASS (no fix needed).
- The R4 field explicitly declines to manufacture a quote_id for a composite fact drawn from
  ordinary institutional-history narrative in two sources (shuckburgh1904, vch1959) rather than from
  a quoted document or eyewitness statement; this is a defensible reading of Procedure E's
  instruction that a quote record captures "the exact text copied," which does not fit a fact stated
  independently, in different words, by two secondary narratives. FLAGGED for the Coordinator's
  attention as a borderline case, not a FAIL: the underlying facts (D.D. 1598, Regius Professor 1607,
  KJV translation) are standard, uncontested institutional-history claims corroborated by both
  sources read this session.

## 5. Dates and numbers vs known-facts sheet
- Founding date "1584 [HIGH]" (roster): matches this dossier's F2 (charter 11 Jan 1583/84; deed of
  foundation 25 May 1584). The roster's date is the charter/foundation date, not the 1585 statutes
  date — consistent, not contradictory; flagged in discrepancies.md item 5 only so the Writer does
  not conflate them.
- Founder "Sir Walter Mildmay" (roster): matches F3.
- "Mildmay's letters to Elizabeth ('I have set an acorn...') [VERIFY wording and source]" (roster):
  addressed in F6/M1 and discrepancies.md item 4. Earliest source found is Fuller 1655, not a letter
  — the roster's own description ("letters to Elizabeth") is not confirmed; the anecdote as found is
  a reported spoken exchange at Court, not an epistolary source. This is itself worth flagging: the
  roster's characterization of the source-form ("letters") does not match what was actually located
  (a secondhand narrative exchange). Recorded here rather than silently corrected, per Rule 9.
- "John Harvard's college" / "founded expressly to make preachers" (roster, "Why included" cell):
  both fully supported — F6, R1.

## 6. Verdict
VERIFIED (zero FAILs on the 38 quote records; the field-level gaps are honestly marked, not
verification failures). One item flagged above (R4's composite-fact sourcing) for the Coordinator's
awareness, not requiring a RETURNED verdict. The nine `[New translation, draft]` Latin-to-English
renderings are flagged for a human Latin-reader's review before final publication, per Procedure B
rule 6, which is a routine follow-up step built into the abridgement process rather than a defect in
this dossier.
