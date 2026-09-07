STATUS: VERIFIED
TICKET: V1-harvard-college
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, charter_abridged.md, sources.csv, text/nef1643.txt,
text/nef1643-orig-scan.txt, text/lawsliberties1642.txt, text/charter1650.txt,
text/quincy1840v1-narrative.txt

# Verification report — harvard-college (V1, dossier + B, abridged charter)

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/harvard-college
WARN harvard-college-q002: before+text+after not contiguous in nef1643
WARN harvard-college-q004: before+text+after not contiguous in nef1643
WARN harvard-college-q007: before+text+after not contiguous in nef1643
WARN harvard-college-q010: before+text+after not contiguous in nef1643
WARN harvard-college-q013: before+text+after not contiguous in nef1643
WARN harvard-college-q029: before+text+after not contiguous in charter1650
WARN harvard-college-q030: before+text+after not contiguous in charter1650
WARN harvard-college-q031: before+text+after not contiguous in charter1650
WARN harvard-college-q032: before+text+after not contiguous in charter1650
WARN harvard-college-q034: before+text+after not contiguous in quincy1840v1-narrative
WARN harvard-college-q040: before+text+after not contiguous in quincy1840v1-narrative
WARN harvard-college-q041: before+text+after not contiguous in quincy1840v1-narrative
43/43 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/harvard-college --charter
6/6 passages verbatim; 0 failures
```

Exit code 0 for both invocations. Zero FAILs.

The 12 WARNs are the script's "before+text+after not contiguous" check, which fails only because
each text/ file's transcription header block, and (for nef1643) a consolidated
`[[pp. 701-704]]` page-range marker, sit as literal characters near some quotes' context windows.
Spot-checked all 12: in each case `text`, `before`, and `after` independently verify verbatim
against the named document (the script's own per-field checks above the WARN line show no FAIL
for these quote_ids); the WARN only means the three fields, concatenated, are not one unbroken
run of characters in the source file, because the source has other material (a page marker, a
paragraph gap) between them. This is the same class of false-positive documented in the
geneva-academy and emmanuel-cambridge verification reports for this project. Not treated as a
failure.

## 2. Quote-by-quote

All 43 quote_ids: `text` field verbatim match PASS (per script). Manually re-opened the source
file at the stated `page` marker and re-read the surrounding prose for 12 of 43 (about one in
four, spread across all five documents and all major dossier sections) to confirm the `page`
label is accurate and the quote is not taken out of a context that reverses its meaning:
harvard-college-q001, q003, q008, q009, q014, q018, q020 (SECONDARY label correct), q022, q028,
q031, q037, q041. All 12 PASS. No quote found to be truncated in a way that changes its sense.

## 3. Field-by-field

Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4, X1-X3) is present
and non-blank. Every filled field cites at least one quote_id, or is marked NOT FOUND /
UNVERIFIED with the document_ids searched, per Rule 1/5. Checked that no summary sentence claims
more than its cited quotes support:

- F2, F3, F5: PASS -- summary combines q002/q020/q029/q031/q033/q034/q038/q039 accurately; the
  dossier does not assert a single unambiguous "founding date" beyond what the sources give
  (grant vote 1636, bequest 1638, incorporation 1650), consistent with discrepancies.md item 1.
- F8: PASS -- dossier does not claim a confession/subscription clause exists; it explicitly notes
  none was found and gives the closest available material (q028, q030) instead.
- A3: PASS -- dossier correctly distinguishes "no admission-testimony requirement found" from the
  "godly life and conversation" language, which the quotes (q018, q019) show attaches to the
  *degree*, not to admission. This is a fact the dossier gets right where a careless summary could
  have conflated the two.
- C1, C3: PASS -- dossier states the schedule's own year-cohort structure without inventing a
  total "4-year" or "7-year" course length that no fetched source states.
- L2: PASS -- dossier reports NOT FOUND for a residence rule, and separately (in prose, not as a
  formal quote) flags that the Edifice's "Chambers and studies" sentence exists but was not
  formalized as a quote record this session; this is an honest, checkable statement, not a
  citation.
- M3: PASS with a flagged gap -- the dossier's own text notes that the "profanation" clause (Rule
  4) was not separately extracted as its own quote_id even though it sits inside the range
  covered by q009/q010's surrounding rules. This is disclosed in the dossier text itself, not
  hidden. Recommend a follow-up Extractor pass formalize a distinct quote_id for Rule 4 if a
  future editor wants "M3" independently citable; not required for this chapter, since the
  chapter's own footnotes can point to the same rules block already reprinted whole in
  charter_abridged.md.
- L7, T5, R1-R4: PASS -- all correctly labeled [SECONDARY] or NOT FOUND rather than stated as
  fact; the one named, sourced individual (John Harvard) is supported by q003, q035, q036, q041,
  all PRIMARY or PRIMARY-quoted-within-secondary.
- S1-S6: PASS -- the dossier does not manufacture a "sending" narrative for Harvard College where
  the fetched sources (unlike, e.g., this book's missionary-society chapters) do not supply one;
  S2 is explicitly labeled UNVERIFIED rather than asserted.

No field was found to overstate its quotes. No PRIMARY-ONLY field (F5-F9, A1-A6, C1-C9, L1-L6,
T2-T3, S1, S4, S6) cites a SECONDARY source; the two PRIMARY-ONLY-adjacent fields that do carry
secondary material (F1, F3 partly) are correctly typed PS in the template, not PO.

## 4. Secondary citations checked

- L7/R1 "first class of nine members...1642" [SECONDARY: nef1643, Miller & Johnson's 1938
  editorial note, harvard-college-q020]: opened text/nef1643.txt at the cited location; the
  editorial note does say this. PASS as an accurate report of what the secondary source says.
  Flagged in discrepancies.md (item 2) as not independently corroborated by a primary class list
  this session -- the dossier and this report both carry that caveat forward; the chapter must
  keep the [SECONDARY] framing and not upgrade it.
- F9 cross-reference to the Emmanuel College, Cambridge chapter (Part I): the claim "John Harvard
  himself was educated at Emmanuel College, Cambridge" is PRIMARY-supported here (q036, via
  Quincy's 1840 narrative, itself reporting what "is all that is known...with distinctness and
  certainty" -- Quincy's own words, correctly treated as secondary narrative, not overstated).
  PASS.

## 5. Dates and numbers vs. known-facts sheet

Roster (`02_institution_roster.md`, harvard-college row): "1636 [HIGH]" (founding); founders
"Massachusetts General Court; John Harvard's bequest 1638"; place "Cambridge, Mass."; key
documents "*New England's First Fruits* (1643); *Rules and Precepts* (1642/1646); Harvard Charter
of 1650."

| item | dossier value | roster value | result |
|---|---|---|---|
| founding year | 1636 (grant vote), 1638 (bequest/naming), 1650 (charter) -- all three given, not collapsed to one date | "1636 [HIGH]" | PASS (consistent; dossier gives the fuller sequence the roster's one-line cell compresses) |
| founders | General Court; John Harvard's bequest 1638 | same | PASS |
| place | Cambridge, Mass. | same | PASS |
| key documents used | New England's First Fruits (1643) = nef1643; Laws/Liberties 1642-46 = lawsliberties1642; Harvard Charter of 1650 = charter1650 | same three documents named | PASS -- all three roster-named key documents were located and fetched this session; Morison 1935 (the roster's other named document) was searched for and found blocked (see sources.csv; not a dossier value to check against, since nothing from it is quoted) |

No date or number in the dossier conflicts with the roster's known-facts sheet. The two internal
discrepancies recorded in discrepancies.md are both cases where the book's own fetched sources
disagree with each other (or where a figure is secondary and unconfirmed), not cases where the
dossier disagrees with the roster.

## 6. Verdict

VERIFIED (zero FAILs on the 43 quote records and the 6 abridged-charter passages; the 12 WARNs
are explained false positives, consistent with prior chapters in this book). Field-level gaps
(NOT FOUND / UNVERIFIED, roughly 20 of 68 numbered fields) are honestly marked, per Rule 1, and
are not verification failures. No RETURNED items.
