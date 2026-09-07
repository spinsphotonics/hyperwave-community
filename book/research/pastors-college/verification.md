STATUS: VERIFIED
TICKET: V1-pastors-college
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, sources.csv, text/autobio-v2.txt, text/autobio-v3.txt, text/lectures1-selection.txt, text/sword-trowel-bound.txt

# Verification report — pastors-college (V1, dossier)

## 1. Script output
```
26/26 quote records passed; 0 failures
```
No WARNs and no FAILs. Every quote record's `text`, `before`, and `after` fields verify verbatim
against the named `text/` file, and in every record `before + text + after` is contiguous in the
source (unlike the Geneva chapter's two WARNs), because each quote was extracted programmatically
by exact line range from the fetched OCR file rather than retyped by hand.

## 2. Quote-by-quote
All 26 quote_ids: text field verbatim match PASS (per script). Spot-checked 6 of 26
(pastors-college-q001, q005, q010, q017, q021, q022) by manually re-opening the relevant text/
file at the stated line range with the Read tool and comparing against the `page` field's stated
context (chapter/running head). PASS for all six: page/running-head markers match what is visible
in the OCR text at that point.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4) is present and
non-blank. For every field carrying one or more quote_ids, the summary sentence(s) were checked
against the cited quote text(s): no summary claims more than its quotes support. PASS for all
cited fields, with two notes:
- L7 and S3 combine several separately-dated figures (1856, 1861, "some seven years ago," end of
  1878) from different quote_ids into one running account. Each individual figure is directly
  supported by its own quote_id; the field does not assert a single unified figure beyond what
  each quote states for its own date. PASS.
- Several fields (C1, C8, L2, L6, T4, discrepancies) refer to surrounding narrative sentences in
  the raw text that were read and are consistent with the field's summary but were NOT formalized
  into their own quote_id this session (each such instance is flagged inline in dossier.md with a
  raw-text line reference for a follow-up Extractor pass). These sentences support, rather than
  contradict, the field's PRIMARY-cited claims, but per Rule 2 ("a quotation is allowed only if
  copied") the Writer must not put these unformalized sentences in quotation marks in the chapter;
  paraphrase only. Flagged, not a FAIL.

Fields wholly or largely marked NOT FOUND/UNVERIFIED: A1, A5, C2 (as an itemized syllabus), C4,
C7, C9 (as a full timetable), L1, L4 (as a written code), L5, M5, S1 (as a formal mechanism), S4,
S6, T5 (as a computed ratio) — 14 of 52 factual fields (roughly 27%) carry no quotable primary
finding for the specific sub-question asked, and are marked accordingly rather than guessed. This
is judged a normal, honest result for a single research session working from five fetched
documents (compare geneva-academy's 26 of 50, or 52%, from a single fetched document); it is not a
verification failure (Rule 8: UNVERIFIED is a success, not a defect).

## 4. Secondary citations checked
No field in this dossier cites a [SECONDARY: ...] source; all PRIMARY-OR-SECONDARY fields (L7,
L8, T1, T4, T5, M4, M5, S2, S3, S5, R1-R4) that carry a finding are supported by PRIMARY quote_ids
from the Autobiography, Lectures to My Students, or Sword and the Trowel, not by any secondary
history. N/A for this section.

## 5. Dates and numbers vs known-facts sheet
- "First student Thomas William Medhurst, 1855; George Rogers appointed tutor 1856; College
  treated 1856 as its founding" [roster HIGH]: consistent with the sources read (single student
  in 1856 per pastors-college-q004; Rogers as tutor per pastors-college-q003), though the exact
  1855 Medhurst date and 1856 Rogers-appointment date were not independently re-derived from a
  dated quote this session — the sources read give 1856 for the student-growth figure and give
  21 March 1857 as the date Medhurst himself went to reside with Rogers (a later, second-stage
  arrangement; not separately quote-recorded, see raw text/autobio-v2.txt line 6858). Not a
  contradiction; the roster's dates concern the earlier Bexley Heath period this session's quotes
  do not directly cover.
- "Tuition, board and lodging free to students; funded by Spurgeon and by his sermon income and
  by the Tabernacle's weekly offering" [roster HIGH]: CONFIRMED by pastors-college-q006 and
  pastors-college-q011 (free tuition/board/lodging) and by surrounding narrative on Spurgeon's own
  income and voluntary offerings (L6).
- "Spurgeon's Friday afternoon lectures are published as Lectures to My Students (first series
  1875)" [roster HIGH]: CONFIRMED — pastors-college-q008 states the Friday classes were "the
  nucleus of those never-to-be-forgotten Lectures to my Students," and text/lectures1-selection.txt
  is the published result, fetched and quoted directly (pastors-college-q019-q021).
- "The College's annual Conference began 1865 [VERIFY]": NOT CONFIRMED by the sources read this
  session; see discrepancies.md item 2. The related-but-distinct fact that 1865 is when the
  College began collecting student statistics IS confirmed (pastors-college-q017). Chapter must
  not state the Conference began in 1865 without further verification.
- "Admission required prior preaching experience of at least two years and a church's testimony
  [VERIFY exact rule and wording]": CONFIRMED, with exact wording, by pastors-college-q005 ("about
  two years," "some seals to his ministry") and independently corroborated by
  pastors-college-q021/q020 from a different fetched edition (Lectures to My Students). The
  "church's testimony" element is more precisely, per the sources read, a matter of written
  testimonials and the candidate's home pastor's letter, not a formal congregational vote —
  chapter should use the sources' own language (testimonials, seals to his ministry) rather than
  the roster's paraphrase "church's testimony."

## 6. Verdict
VERIFIED (zero FAILs on the 26 quote records; the field-level gaps are honestly marked per Rule 8,
not verification failures). Two items are RETURNED for a future revision pass, not blocking this
verdict: (a) formalize the several raw-text passages flagged inline in dossier.md into their own
quote records; (b) resolve the exact issue date of the Sword and the Trowel passage
(discrepancies.md item 1) and the exact founding date of the annual Conference
(discrepancies.md item 2).
