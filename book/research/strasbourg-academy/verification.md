STATUS: VERIFIED
TICKET: V1-strasbourg-academy
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/sturm1538-ghi.txt, text/calvin-psalms-preface.txt, text/museeprotestant-gymnase.txt, text/museeprotestant-sturm-bio.txt, sources.csv, discrepancies.md

# Verification report — strasbourg-academy (V1, dossier)

## 1. Script output
```
20/20 quote records passed; 0 failures
```
No WARNs. All 20 quote records' `text`, `before`, and `after` fields match verbatim in their
named `text/` file, and the before+text+after windows are contiguous in the source (no OCR
running-head artifact split any of these quotes, unlike two quotes in the geneva-academy
dossier).

## 2. Quote-by-quote
| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| strasbourg-academy-q001 | PASS | PASS | PASS (p. 71-72 per citation) | PASS |
| strasbourg-academy-q002 | PASS | PASS | PASS (p. 71-72) | PASS |
| strasbourg-academy-q003 | PASS | PASS | PASS (p. 73) | PASS |
| strasbourg-academy-q004 | PASS | PASS | PASS (p. 73-74) | PASS |
| strasbourg-academy-q005 | PASS | PASS | PASS (p. 74) | PASS |
| strasbourg-academy-q006 | PASS | PASS | PASS (p. 75-76) | PASS |
| strasbourg-academy-q007 | PASS | PASS | PASS (p. 76) | PASS |
| strasbourg-academy-q008 | PASS | PASS | PASS (img. 1) | PASS |
| strasbourg-academy-q009 | PASS | PASS | PASS (img. 1) | PASS |
| strasbourg-academy-q010 | PASS | PASS | PASS (img. 1) | PASS |
| strasbourg-academy-q011 | PASS | PASS | PASS (img. 1) | PASS |
| strasbourg-academy-q012 | PASS | PASS | PASS (sec. heading) | PASS |
| strasbourg-academy-q013 | PASS | PASS | PASS (sec. heading) | PASS |
| strasbourg-academy-q014 | PASS | PASS | PASS (sec. heading) | PASS |
| strasbourg-academy-q015 | PASS | PASS | PASS (sec. heading) | PASS |
| strasbourg-academy-q016 | PASS | PASS | PASS (sec. heading) | PASS |
| strasbourg-academy-q017 | PASS | PASS | PASS (p. xxxv-xxxvi) | PASS |
| strasbourg-academy-q018 | PASS | PASS | PASS (p. xlii) | PASS |
| strasbourg-academy-q019 | PASS | PASS | PASS (p. xliii) | PASS |
| strasbourg-academy-q020 | PASS | PASS | PASS (p. 74) | PASS |

Quotes q017-q019 (Calvin's Preface) contain original page-line hyphenation exactly as OCR
delivered it (e.g. "Stras-\nburg", "anewstation" for "a new station"); this is the source's own
OCR artifact, reproduced per Procedure F ("do not correct"), not an error in the quote record.
Any use of these three quotes in the chapter's running prose silently rejoins the line-end
hyphen (a typographic convention, not a wording change) and is noted as such where it occurs.

## 3. Field-by-field
Every field in the template (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3) is
present in dossier.md. Checked each summary sentence against its cited quote_id(s): no summary
claims more than its quotes support. Two fields deserve a specific note:
- F9: the summary is careful to state that Calvin's Preface does not name Sturm or the Gymnasium
  specifically, and that the secondary source (q009) is the only source connecting Calvin by name
  to this particular school. PASS — the distinction between what the primary source (Calvin's own
  words) and the secondary source each support is preserved, not blurred.
- C8: the summary is careful to date the degree-granting/university stages (1566, 1621) as later
  than, and distinct from, Sturm's 1538 foundation. PASS.

Fields marked NOT FOUND or UNVERIFIED: 31 of 51 dossier line-items (counting F1-F9, A1-A6, C1-C9,
L1-L8, T1-T5, M1-M5, S1-S6 individually, and R1-R4 as one combined item, per the template's own
grouping = 9+6+9+8+5+5+6+1 = 49 items; R1-R4 counted as 1). Of these, 31 are NOT FOUND/UNVERIFIED,
18 have at least a partial answer. This is a large fraction, consistent with the geneva-academy
precedent (26 of 50 there) and reflects that only two primary documents (one of them only a
partial excerpt of a much longer work, the other not even about this institution) were read this
session. Recorded, not treated as a verification failure (Rule 8: UNVERIFIED/NOT FOUND is a
success, not a defect, when honestly marked).

## 4. Secondary citations checked
| citation | page says what is claimed? | result |
|---|---|---|
| museeprotestant-gymnase (q008-q011) | Re-opened text/museeprotestant-gymnase.txt; each cited sentence appears exactly as summarized. | PASS |
| museeprotestant-sturm-bio (q012-q016) | Re-opened text/museeprotestant-sturm-bio.txt; each cited sentence appears exactly as summarized. | PASS |

No field marked PRIMARY-ONLY (PO) in this dossier cites a secondary source; secondary citations
are confined to fields marked PS (F1, F3 partial, F4 partial, C2, C8, L7, L8, M4, M5, T1 partial,
T4, T5, S2, S3, S5, R1-R4), consistent with the template's PO/PS labeling rule.

## 5. Dates and numbers vs known-facts sheet
| item | dossier value | facts-sheet value | result |
|---|---|---|---|
| Founding year | 1538 (F2) | "Strasbourg Gymnasium under Sturm opened 1538. [HIGH]" (Other dates) | MATCH |
| Founder | Johannes Sturm (F3) | "Johannes Sturm, with Bucer and the Strasbourg magistrates" (roster row) | PARTIAL — Sturm confirmed; Bucer's specific role as co-founder (vs. later teacher) not confirmed this session, see discrepancies.md item 1. Not a contradiction, an unconfirmed addition. |
| Calvin taught 1538-41 | Not stated as a date range in this dossier; only that Calvin was "drawn" to Strasbourg by Bucer after leaving Geneva and taught in "our small school" there (F9), and that a secondary source lists him among those who taught at the Gymnasium (F9) | "Calvin taught here 1538-41" (roster row, "why included") | CONSISTENT but narrower — this session did not locate a source stating the exact 1538-41 date range; the chapter should not state "1538-41" without a citation for those specific years, which is flagged as a gap here. |
| Motto | UNVERIFIED (M4) | not in known-facts sheet (roster does not state a motto for this institution) | no conflict; roster is silent, this dossier is honestly silent too |

## 6. Verdict
VERIFIED (zero FAILs on the 20 quote records; field-level gaps are honestly marked, not
verification failures). One item for a future revision pass, not a FAIL: confirm or drop the
"1538-41" Calvin date range and Bucer's co-founder role with a source that was not read this
session (Schmidt 1855 is the leading candidate for both, per sources.csv).
