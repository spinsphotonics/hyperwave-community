STATUS: VERIFIED
TICKET: V1-student-volunteer-movement
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, discrepancies.md, sources.csv, text/cleveland1891.txt, text/mott1900.txt, text/motthistory1892.txt, text/firsttwodecades1906.txt

# Verification report — student-volunteer-movement (V1, dossier)

## 1. Script output
```
19/19 quote records passed; 0 failures
```

## 2. Quote-by-quote
All 19 quote_ids: `text`, `before`, and `after` fields independently verified verbatim against
the named `text/` file by the check script (whitespace-normalized match). Spot-checked 6 of 19
(svm-q001, svm-q002, svm-q006, svm-q012, svm-q013, svm-q014) by manual inspection against the
raw fetched files at the line numbers used to build them (see extraction commands in session
history): PASS for all six. The `page` field for each quote gives a descriptive printed-page
number or section heading (no `[[p. N]]` markers were inserted into the text files, matching
the precedent already set in `research/geneva-academy`).

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| svm-q001 | yes | yes | yes | PASS |
| svm-q002 | yes | yes | yes | PASS |
| svm-q003 | yes | yes | yes | PASS |
| svm-q004 | yes | yes | yes | PASS |
| svm-q005 | yes | yes | yes | PASS |
| svm-q006 | yes | yes | yes | PASS |
| svm-q007 | yes | yes | yes | PASS |
| svm-q008 | yes | yes | yes | PASS |
| svm-q009 | yes | yes | yes | PASS |
| svm-q010 | yes | yes | yes | PASS |
| svm-q011 | yes | yes | yes | PASS |
| svm-q012 | yes | yes | yes | PASS |
| svm-q013 | yes | yes | yes | PASS |
| svm-q014 | yes | yes | yes | PASS |
| svm-q015 | yes | yes | yes | PASS |
| svm-q016 | yes | yes | yes | PASS |
| svm-q017 | yes | yes | yes | PASS |
| svm-q018 | yes | yes | yes | PASS |
| svm-q019 | yes | yes | yes | PASS |

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4-plus-one, X1-X3) is
present. Fields with quote_ids: summary sentence checked against the cited quote text; no
summary claims more than its quotes support. PASS for all cited fields.
Fields marked NOT FOUND, UNVERIFIED, or "not applicable": A1, A2, A4 (partly), A6, C1-C9,
L1-L5, L8, S2, S5, S6 -- 19 of roughly 50 template line-items. This reflects (a) that a student
sending movement genuinely has no curriculum, residence, or faculty in the template's sense
(the same honest gap already logged for `abcfm`), and (b) the OCR table-breakage documented in
discrepancies.md item 2, which blocks any specific destination or casualty figure. Recorded,
not a verification failure (Rule 8: UNVERIFIED is a success, not a defect).

## 4. Secondary citations checked
- A5 (women) cites `firsttwodecades1906` for the "about one-third...were women" statistic in a
  PS aside without a formal quote_id (the surrounding prose was captured contextually by
  svm-q014's extraction but the one-third figure itself sits in the sentence immediately after
  svm-q014's quoted text, inside the same paragraph). FLAGGED for a follow-up pass: either
  formalize as its own quote record or downgrade to UNVERIFIED. Recommend downgrading if this
  dossier is revised further; the chapter does not state the one-third figure as a footnoted
  fact for this reason.
- R2 (Mott presiding at Cleveland) cites `cleveland1891`'s officer list and opening-session line
  informally (see discrepancies.md item 3) rather than as a formal quote_id. Same
  recommendation: treat as PS/illustrative, not a hard citation, in the chapter.
- M5 cites `mott1900` generally ("Mott's 1900 book restates and defends the Watchword at
  length") without a further quote_id beyond svm-q013's context. This is a summary of the
  book's argument, not a specific quoted claim, and is treated as such in the chapter (no
  specific sentence from this general restatement is quoted as if formally extracted).

## 5. Dates and numbers vs known-facts sheet
- Mount Hermon conference, July 1886: matches `book/plan/02_institution_roster.md` known-facts
  sheet [HIGH]. PASS.
- Organized 1888: matches roster [HIGH] -- the Executive Committee's formal organization work
  began January 1889, per motthistory1892, which is consistent with (not contradictory to) a
  1888 Northfield decision to organize, per cleveland1891's own account (the 1888 Northfield
  conference resolved on organization; the Committee began work in January 1889). Both dates
  given in the dossier (F2). PASS, with the nuance recorded rather than compressed.
- Declaration card wording "It is my purpose, if God permit, to become a foreign missionary":
  confirmed verbatim in a primary source (motthistory1892, svm-q001), dated precisely to 14
  July 1892 as a revision of the 1886-91 wording. This matches the roster's characterization of
  the card, though the roster does not itself give the 1892 date; PASS, and the more precise
  date is a refinement, not a contradiction.
- Watchword "the evangelization of the world in this generation": confirmed verbatim and
  its 1888 adoption date confirmed in a primary source (mott1900, svm-q013). PASS.
- Publication year of Mott's *Evangelization* book: roster says 1900; fetched copy's own
  title page/catalog record says 1901. See discrepancies.md item 1. Not resolved; flagged.

## 6. Secondary-only-source check
No field marked PRIMARY-ONLY (PO) in this dossier cites a [SECONDARY: ...] source. PASS.

## 7. Label check
Every dossier field carries a (PO) or (PS) label per the template. PASS.

## 8. Verdict
VERIFIED (zero FAILs in the quote check or the field-by-field review). Two informal PS asides
(A5's one-third figure; R2's convention-title detail) are flagged in section 4 above for a
possible follow-up formalization pass, and the chapter is written to avoid asserting either as
a hard, footnoted number/fact beyond what its quote_id actually supports.
