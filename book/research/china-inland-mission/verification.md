STATUS: VERIFIED
TICKET: V1-china-inland-mission
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/*.txt, sources.csv, discrepancies.md

# Verification report — china-inland-mission (V1, dossier)

## 1. Script output
```
28/28 quote records passed; 0 failures
```
No WARNs this session (all `before`/`text`/`after` windows are contiguous in their source
files, unlike the geneva-academy dossier's two WARNs).

## 2. Quote-by-quote
All 28 quote_ids: `text`, `before`, and `after` fields verified verbatim against the named
`text/` file by the automated checker. Manually spot-checked 8 of 28 (cim-q001, cim-q003,
cim-q009, cim-q013, cim-q021, cim-q022, cim-q025, cim-q028) by re-opening the cited page/line
region in the corresponding `text/*.txt` file and confirming the page/running-head note in the
`page` field matches what is visible in the OCR text at that point. PASS for all 8 spot-checked;
no reason to doubt the remaining 20, which passed the automated character-for-character check.

## 3. Field-by-field
All 51 dossier fields (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3) are
present and non-blank.
- Fields with quote_ids: summary sentences checked against their cited quotes; no summary claims
  more than its quotes support. PASS for all cited fields.
- Fields marked NOT FOUND (pure, no partial content): F9, A1, C5, C9, L1, T4, T5 — 7 of 51
  fields (14%). PASS (Rule 1: NOT FOUND is a success, not a defect).
- Fields with partial UNVERIFIED content (M4's motto claim, R3's incomplete Fruit entry) are
  explicitly labeled as such in the dossier text itself, not silently omitted. PASS.
- The remaining 42 of 51 fields (82%) carry at least one dated, cited fact from a document
  actually read this session. This is a substantially thicker dossier than the geneva-academy
  precedent (which had 26/50 fields NOT FOUND/UNVERIFIED, 52%), reflecting that six primary/
  primary-adjacent full texts were fetched this session rather than one.

## 4. Secondary citations checked
- F3 (Berger/Gough at Brighton) and F4 (30 Coborn Road) each carry one [SECONDARY:
  jubileechinamis00broouoft] narrative citation without a formal quote_id (page number given
  inline instead). This is a minor departure from the template's normal standard (every fact
  needs a quote_id); flagged, consistent with how the geneva-academy verification handled the
  same pattern (recommend downgrading to UNVERIFIED or formalizing in a follow-up pass; not
  treated as a verification FAIL here because the underlying claim is low-stakes background
  color, not a number or a doctrinal claim).
- L3 (annual/weekly worship pattern) similarly carries [SECONDARY: jubileechinamis00broouoft,
  pp. 34, 39] without a formal quote_id. Same disposition as above.
- S1's summary of Article 3's Home/China Department division is drawn from the same document
  already quoted as cim-q003 but paraphrases additional sentences of Article 3 not literally
  quoted in cim-q003; this is a PRIMARY-ONLY field summarizing PRIMARY-ONLY text actually read
  (principlespracti00chin_0), so no [SECONDARY] tag is needed, but it is flagged that not every
  sentence paraphrased there has its own quote_id. Recommend a follow-up pass adding one or two
  more quote records from Article 3 if this field is drawn on heavily in the chapter.

## 5. Dates and numbers vs known-facts sheet
- 25 June 1865 (Brighton) and the Lammermuir party sailing 1866: both match
  `book/plan/02_institution_roster.md`'s known-facts sheet ("25 June 1865 (Brighton); Lammermuir
  party 1866 [HIGH]"). PASS.
- Lammermuir party size: the roster does not give a specific number, so there is no roster
  figure to check against; this session's own two sources disagree with each other (17 adults +
  4 children per Taylor 1895 vs. 22 in all per Broomhall 1915) — recorded in discrepancies.md
  item 1, not resolved, per Rule 9.
- *Principles and Practice* edition date: the roster itself marks this "[VERIFY]" and this
  session's findings (an undated fetched copy catalogued 1900, plus separately catalogued 1904
  and 1905 items not fetched) are consistent with the roster's flagged uncertainty, not a
  contradiction of it. See discrepancies.md item 2.
- Boxer-crisis casualty figures (52 adults + 16 children CIM dead, cim-q028) and the "hundred who
  sailed in 1887" cohort figures (cim-q025) are both from primary/primary-adjacent sources
  actually read this session and are kept distinct per discrepancies.md item 4; the roster does
  not give its own casualty figure to check against.
- *Arrangements of the China Inland Mission* (1886): the roster names this specific document;
  this session could not locate it as digitized text (print/microfilm only at SOAS). This is a
  gap, not a contradiction — recorded in discrepancies.md item 3 and text/cim-arrangements-1886.REQUEST.md.

## 6. PRIMARY-ONLY fields checked for secondary citations
All fields marked PO in the template (F5-F9, A1-A6, C1-C9, L1-L6, T2-T3, S1, S4, S6) were
checked; none cites a secondary source as its sole support. Where secondary narrative color is
used (F3, F4, L3, as noted in §4 above), it supplements rather than replaces primary-sourced
content, and is explicitly labeled [SECONDARY]. PASS.

## 7. Unlabeled statements
Every fact-bearing sentence in the dossier carries a confidence label (PO), (PS), or an inline
[SECONDARY: ...] tag, and every fact is followed by quote_ids or a NOT FOUND/UNVERIFIED marker.
Spot-checked all 51 fields; none found without a label. PASS.

## 8. Verdict
VERIFIED (zero FAILs on the 28 quote records; the 7 pure NOT FOUND fields and the M4/R3 partial-
UNVERIFIED fields are honestly marked, not verification failures). RETURNED items for a future
revision pass: formalize the F3/F4/L3 [SECONDARY] narrative citations as proper quote records
(§4 above), and pursue the follow-up fetches noted in scout_notes.md and discrepancies.md
(the 1886 *Arrangements*, the 1904/1905 *Principles and Practice* items, a primary shipping
manifest for the Lammermuir party, and further reading of `hudsontaylorchin00tayl` for the
Fruit section).
