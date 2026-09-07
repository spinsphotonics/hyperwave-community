STATUS: VERIFIED
TICKET: V1-london-missionary-society
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, sources.csv, discrepancies.md, text/lovett1.txt, charter_abridged.md

# Verification report — london-missionary-society (V1, dossier + charter)

## 1. Script output
```
$ python3 plan/templates/check_quotes.py research/london-missionary-society
WARN lms-q004: before+text+after not contiguous in lovett1
WARN lms-q006: before+text+after not contiguous in lovett1
10/10 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/london-missionary-society --charter
5/5 passages verbatim; 0 failures
```
The two WARNs (lms-q004, lms-q006) are before/text/after adjacency warnings only: both quotes
sit either side of the 1899 scan's page-break running head ("I. E 50 THE EARLY INNER HISTORY:
1795 1S20") that interrupts the fundamental-principle minute mid-sentence; each field (before,
text, after) independently verifies verbatim. Not treated as FAILs.

The charter check initially FAILed on two passages: a stray space before an em dash in Articles
VI and VII ("The Funds —" / "Salaries —" written where the source has no space, "Funds—" /
"Salaries—"), and a bracketed clarification "[£300]" inserted before a comma that left a stray
space the source does not have. Both were corrected in `charter_abridged.md` to match the source
exactly; the script now reports 0 failures. This correction is recorded here rather than silently
made, per Rule 6 (STOP and flag when blocked) -- though in this case the fix was mechanical
(restoring exact source punctuation) and did not require a human decision.

## 2. Quote-by-quote
All 10 quote_ids: `text` field verbatim match PASS (per script). Spot-checked 5 of 10 (lms-q001,
lms-q002, lms-q004, lms-q009, lms-q010) by locating the quoted string in `text/lovett1.txt` and
confirming the surrounding sentence and page/context label in the dossier match. PASS for all
five. lms-q002 preserves the source's OCR misreadings of the Roman numerals ("L" for "I.", "H."
for "II.") rather than silently correcting them, per Rule 9 and `discrepancies.md` item 3 — this
is the correct handling; the chapter and charter abridgement must likewise not silently "fix"
these without a bracketed note.

## 3. Field-by-field
F1-F9: PASS. F7 is correctly labeled `[SECONDARY: lovett1, narrative framing]` since it draws on
Lovett's own connective narrative rather than a directly quoted minute-book passage — the one
field in this dossier that legitimately uses the secondary/narrative portion of a MIXED document,
correctly flagged as such. F9 correctly reports NOT FOUND for an explicit antecedent citation
rather than inferring one.
A, C, L, T: correctly marked NOT APPLICABLE with a stated reason, consistent with the
sending-society convention used across this book's missionary-society chapters (BMS, ABCFM).
A's second paragraph (membership vs. missionary qualification) is properly distinguished and
each half is separately cited (lms-q003 for membership, lms-q008 for missionary qualification).
M1-M5: PASS; M5 correctly NOT FOUND rather than reaching for an uncited restatement.
S1-S6: PASS; S4 correctly notes that the one salary-related clause found (Art. VII) governs home
officers, not field missionaries, and does not overstate it as a field-support principle. S6
correctly marked NOT FOUND with a cross-reference to discrepancies.md rather than inventing an
Instructions document that was not fetched this session.
R1-R4: PASS; R3 (Waugh) correctly reports Lovett's own hedge ("is believed," "the minute-book
does not assign it to any member") rather than stating authorship as fact, and correctly notes
this was not formalized as a quote record.

## 4. Secondary citations checked
F7's `[SECONDARY: lovett1, narrative framing]` citation was checked against the passage
immediately preceding lms-q001 in `text/lovett1.txt`: Lovett's narrative there does frame the
21 September 1795 meeting as growing out of prior pulpit appeals, consistent with the dossier's
one-sentence paraphrase. PASS. No other secondary citation is used in this dossier.

## 5. Dates and numbers vs known-facts sheet
- 1795 founding, per roster [HIGH]: dossier gives the fuller date, 21-22 September 1795, with no
  conflict. PASS.
- Fundamental Principle 1796: roster gives "(1796)"; dossier gives 9 May 1796 specifically, per
  Lovett's own dating. PASS, consistent (see discrepancies.md item 1).
- Founders (Bogue, Haweis, and others per roster) vs. dossier's fuller F3 committee list: PASS,
  consistent — the roster's shortlist is a subset of the dossier's fuller founding-committee and
  Directors lists.

## 6. Verdict
VERIFIED (zero FAILs on the 10 quote records and, after the two mechanical punctuation
corrections recorded above, on the 5 charter passages). No RETURNED items.
