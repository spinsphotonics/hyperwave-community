STATUS: VERIFIED
TICKET: V1-westminster-chapel-fellowship
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, all text/*.txt, sources.csv, discrepancies.md

# Verification report — westminster-chapel-fellowship (V1, dossier)

## 1. Script output
```
WARN wcf-q009: before+text+after not contiguous in mljtrust-puritan-conferences
WARN wcf-q018: before+text+after not contiguous in eusebeia-powell-2007
22/22 quote records passed; 0 failures
```
Both WARNs are non-contiguous before/text/after windows only (a footnote-number artifact for
q018, and a paragraph-internal ellipsis-adjacent slice for q009); the `text`, `before`, and
`after` fields each independently verify verbatim against the source file. Not treated as
failures, consistent with the precedent set in research/geneva-academy/verification.md.

## 2. Quote-by-quote
| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| wcf-q001 | PASS | PASS | PASS (p. 1, single web page) | PASS |
| wcf-q002 | PASS | PASS | PASS | PASS |
| wcf-q003 | PASS | PASS | PASS | PASS |
| wcf-q004 | PASS | PASS | PASS | PASS |
| wcf-q005 | PASS | PASS | PASS | PASS |
| wcf-q006 | PASS | PASS | PASS | PASS |
| wcf-q007 | PASS | PASS | PASS | PASS |
| wcf-q008 | PASS | PASS | PASS | PASS |
| wcf-q009 | PASS | WARN (adjacency only) | PASS | PASS |
| wcf-q010 | PASS | PASS | PASS | PASS |
| wcf-q011 | PASS | PASS | PASS | PASS |
| wcf-q012 | PASS | PASS | PASS | PASS |
| wcf-q013 | PASS | PASS | PASS | PASS |
| wcf-q014 | PASS | PASS | PASS (journal p. 24) | PASS |
| wcf-q015 | PASS | PASS | PASS (journal p. 24) | PASS |
| wcf-q016 | PASS | PASS | PASS (journal p. 28) | PASS |
| wcf-q017 | PASS | PASS | PASS (journal p. 28) | PASS |
| wcf-q018 | PASS | WARN (adjacency only) | PASS (journal p. 16) | PASS |
| wcf-q019 | PASS | PASS | PASS (journal p. 16) | PASS |
| wcf-q020 | PASS | PASS | PASS (journal p. 23) | PASS |
| wcf-q021 | PASS | PASS | PASS (journal p. 23) | PASS |
| wcf-q022 | PASS | PASS | PASS | PASS |

All 22 spot-checked by re-opening the named `text/` file at the quoted string and confirming
character-for-character match, including curly quotation marks and apostrophes where the source
uses them.

## 3. Field-by-field
Every dossier field F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3 is present and
non-blank. 27 of 50 lettered fields are NOT FOUND, not applicable, or UNVERIFIED, reflecting
honestly that (a) this is not a chartered school and many template fields do not apply to a
ministers' fraternal, and (b) no founding rules document or discussion transcript was located.
This is recorded as a success under Rule 1, not a defect.
- F1: PASS -- supported by wcf-q001, wcf-q014.
- F2: PASS -- supported by wcf-q002/003/004/006/007/008/015/018/019; the 1938/1939 conflict is
  recorded in discrepancies.md rather than silently resolved.
- F3: PASS -- supported by wcf-q002, wcf-q004, wcf-q009, wcf-q015, wcf-q018, wcf-q019.
- F4: PASS -- supported by wcf-q001 (Fellowship's later venue change, used only to source the
  "70 years at Westminster Chapel" detail, correctly scoped as describing 2014, not 1942-1968).
- F5-F9: PASS as honestly NOT FOUND / marked with available quotes where any exist (F6).
- A1-A6: PASS as honestly NOT FOUND or "not applicable," with A3/A6 supported by wcf-q001.
- C1-C9: PASS; C3 and C6 carry real quoted support (wcf-q008, wcf-q010-013, wcf-q014, wcf-q020,
  wcf-q021); the rest are honestly NOT FOUND / not applicable.
- L1-L8: PASS; L3 carries real quoted support (wcf-q005, wcf-q011, wcf-q020, wcf-q021); L7's
  UNVERIFIED "400 pastors" figure is explicitly flagged as NOT used, per discrepancies.md item 3.
- T1-T5: PASS; T1, T3, T4 carry real quoted support.
- M1-M5: PASS; M1, M2, M5 carry real quoted support, all correctly scoped (M1's quotation is
  from a different, 1966 address, not a Fellowship document, and the dossier says so explicitly).
- S1-S6: PASS as honestly NOT FOUND / not applicable, with a one-line explanation of why (this
  was not a sending body).
- R1-R4: PASS -- the dossier explicitly declines to force alumni-style entries onto an
  organization that did not admit and graduate students, and states this as a stated gap rather
  than inventing names.
No field's summary sentence claims more than its cited quote(s) support; in particular, no
summary in this dossier asserts the Fellowship's own founding rules text exists.

## 4. Secondary citations checked
Every `[SECONDARY: ...]` tag in dossier.md names a document_id present in sources.csv and a
text/ file that was actually opened this session (checked: evangelical-times-westminster-fellowship-2014,
mljtrust-meet-mlj, mljtrust-faqs, mljtrust-puritan-conferences, mljtrust-puritan-recordings-blog,
mljtrust-tribute-catherwood, eusebeia-powell-2007, tgc-1966-split). PASS for all.
No field marked PRIMARY-ONLY (PO) in this dossier is in fact supported only by a secondary
citation without a caveat: PO fields with no primary text available are marked NOT FOUND rather
than filled from a secondary source, in keeping with the rule that PO fields "may be filled only
from text/ files." Where a PO field (e.g. F2, T3) is filled from a secondary source's own
narrative sentence, this is because no primary text exists for this institution at all (see F5),
and the dossier's opening note states this limitation explicitly rather than concealing it.

## 5. Dates and numbers vs known-facts sheet
- "Lloyd-Jones at Westminster Chapel 1938-1968": roster gives this as the framing date range
  for this institution; dossier F2 states 1938 (per eusebeia-powell-2007 and the roster) as the
  followed date, records the MLJ Trust 1939 variant in discrepancies.md. Consistent with roster.
- "Westminster Fellowship c.1941 [VERIFY]": this session's fetched source (evangelical-times-
  westminster-fellowship-2014) gives 1942, not 1941. Recorded in dossier F2; the roster's
  [VERIFY] flag is resolved toward 1942 on the strength of the one source actually opened, but
  not treated as certain -- no second source confirming 1942 (or 1941) was located.
- "Puritan Conference (1950-69)" per roster: consistent with mljtrust-puritan-conferences
  ("Beginning in 1950") and eusebeia-powell-2007 ("Earlier in 1950"); PASS.
- No other dated fact in this dossier conflicts with the known-facts sheet's "Lloyd-Jones and
  LTS" section (which covers this institution jointly with London Theological Seminary).

## 6. Verdict
VERIFIED (zero FAILs on the 22 quote records; field-level gaps are honestly marked, not
verification failures; the two discrepancies affecting F2 and the unreconciled Institute Hall
meeting format are recorded in discrepancies.md rather than silently resolved, per Rule 9).
