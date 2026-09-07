STATUS: VERIFIED
TICKET: V1-andover-seminary
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/woods1885.txt, sources.csv, discrepancies.md

# Verification report — andover-seminary (V1, dossier)

## 1. Script output
```
WARN andover-seminary-q002: before+text+after not contiguous in woods1885
WARN andover-seminary-q004: before+text+after not contiguous in woods1885
WARN andover-seminary-q006: before+text+after not contiguous in woods1885
WARN andover-seminary-q008: before+text+after not contiguous in woods1885
WARN andover-seminary-q009: before+text+after not contiguous in woods1885
WARN andover-seminary-q010: before+text+after not contiguous in woods1885
WARN andover-seminary-q011: before+text+after not contiguous in woods1885
WARN andover-seminary-q012: before+text+after not contiguous in woods1885
WARN andover-seminary-q015: before+text+after not contiguous in woods1885
WARN andover-seminary-q017: before+text+after not contiguous in woods1885
WARN andover-seminary-q018: before+text+after not contiguous in woods1885
18/18 quote records passed; 0 failures
```
All 18 quote records pass (text, before, and after each independently verified verbatim
against text/woods1885.txt). The WARNs indicate that the exact concatenation of
before+text+after was not found as one contiguous run -- this happens where a running head
or page-break page number is interposed between the fields as printed (e.g., "CONSTITUTION
OF THE THEOLOGICAL SEMINARY. 235" breaking into the middle of a sentence). Each field was
independently confirmed to match verbatim; not treated as a failure, consistent with the
`geneva-academy` precedent for this project.

## 2. Quote-by-quote
All 18 quote_ids: `text` field verbatim match PASS (per script). Manually spot-checked 6 of
18 (q001, q003, q008, q013, q014, q016) by reading the surrounding lines of
text/woods1885.txt directly: in each case the quoted passage sits inside the article/section
named in the "page" field, and the article numbering (FIRST, SECOND, FOURTH, ELEVENTH,
THIRTEENTH, etc.) matches what is printed in the source. PASS.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4) is present.
Fields with quote_ids: summary sentences checked against their cited quotes; no summary
claims more than its quotes support. PASS for all cited fields.
Fields marked NOT FOUND: A1, A6 (partial), C5, L2, L5, L7, L8, T4 (partial), T5, M4, M5,
S2-S5 (as Andover-specific), R2-R4. This is a substantial fraction of the 50 fields, but
every NOT FOUND names the document(s) searched, per Rule 5. Recorded, not a verification
failure (Rule 1/8: an honest NOT FOUND is a success).
F9 (Antecedents) is PARTIAL: the dossier reports the merger of two prior seminary projects
from Woods's narrative but flags that no primary passage naming an explicit outside model
was located and quoted. PASS as an honestly hedged field.

## 4. Secondary citations checked
No field in this dossier cites a secondary source with a `[SECONDARY: ...]` tag; all cited
material is the woods1885 documentary appendix, treated PRIMARY per this dossier's opening
note (verbatim reprints of the 1808 constitutional instruments). R1 cross-references the
`abcfm` dossier's own quote_ids (abcfm-q003, abcfm-q004), which are themselves PRIMARY
(Anderson 1861's quoted 1810 memorial account) -- flagged here as a cross-folder citation,
not a same-session PRIMARY-ONLY violation, since R1-R4 is a PS (PRIMARY-OR-SECONDARY) field
per the template. PASS.

## 5. Dates and numbers vs known-facts sheet
- Andover founded 1808 [HIGH per roster]: consistent with this dossier's documented sequence
  (original Constitution subscribed 31 Aug 1807; Associate Statutes 21 March 1808; Additional
  Statutes/Associate Creed uniting the two foundations 3 May 1808). The roster's single-year
  label "1808" is consistent with, though a compression of, this multi-instrument sequence;
  no contradiction found. Recorded as a nuance in discrepancies.md item 3, not a discrepancy
  with the known-facts sheet.
- Judson at Andover 1808-1810 [HIGH per roster known-facts sheet]: this dossier's R1 entry
  (via the abcfm cross-reference) is consistent with, though does not itself independently
  re-verify, this date range; the 1810 Bradford memorial date is confirmed by quotes in the
  abcfm dossier, not re-derived here.
- No other date or number in this dossier conflicts with the known-facts sheet in
  `02_institution_roster.md`.

## 6. Verdict
VERIFIED (zero FAILs on the 18 quote records; field-level gaps are honestly marked per Rule
1/8, not verification failures).
