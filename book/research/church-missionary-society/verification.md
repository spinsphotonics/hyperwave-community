STATUS: VERIFIED
TICKET: V1-church-missionary-society
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, sources.csv, discrepancies.md, text/stock1.txt, charter_abridged.md

# Verification report — church-missionary-society (V1, dossier + charter)

## 1. Script output
```
$ python3 plan/templates/check_quotes.py research/church-missionary-society
12/12 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/church-missionary-society --charter
8/8 passages verbatim; 0 failures
```
No WARNs on the quote-record check (unlike the other two Part III dossiers this session, whose
context windows crossed OCR page-break artifacts). The charter check initially FAILed twice: the
abridgement's bracketed section headings ("[Resolution 2, on the ground left unreached...]")
were placed where Stock's own connecting narrative sentences had been cut, but without an
accompanying `[...]` cut marker, so the script tried to match Resolution 1's text running
straight into Resolution 2's as one contiguous string, which the source does not support (Stock's
narrative sits between them). Added `[...]` at each such transition; the script now reports 0
failures. Recorded here per Rule 6 rather than silently fixed; the correction restores, not
alters, what the source contains.

## 2. Quote-by-quote
All 12 quote_ids: `text` field verbatim match PASS (per script). Spot-checked 6 of 12 (cms-q001,
cms-q002, cms-q003, cms-q007, cms-q008, cms-q010) against `text/stock1.txt`. PASS for all six.
cms-q001 and cms-q008 preserve the OCR's systematic R-to-E misreading ("Eev." for "Rev.," "Eector"
for "Rector," "Ghurch-iyrinciple" for "Church-principle") rather than silently correcting it, per
Rule 9 and `discrepancies.md` item 2 -- correct handling; the chapter's own prose must use the
standard spellings outside quotation marks, which it does.

## 3. Field-by-field
F1-F9: PASS. F5 correctly documents that the 1799 pamphlet named in the roster was not located
this session and substitutes the four Resolutions (genuinely quoted by Stock as reproductions of
"the original Minutes") as the working charter text, cross-referenced to discrepancies.md item 1
rather than treating the substitution as if it were the roster's named document. F8 correctly
distinguishes the absence of a separate creedal clause from the Society's actual doctrinal
footing (the "Church-principle"), and does not overstate one as the other.
A, C, L, T: PASS, and distinct from the other two Part III dossiers in this batch — because CMS
did found a residential college (Islington, 1825), these fields correctly carry real content
(admission numbers, examination subjects, named Principal) rather than a blanket NOT APPLICABLE,
while still flagging that this content pertains to the College, not the Society's own 1799
founding. C correctly stops at NOT FOUND for the College's length of course and set texts rather
than guessing from the examination-subject list what a "curriculum" beyond that must have been.
M1-M5: PASS; M3 and M5 correctly NOT FOUND rather than reaching for a paraphrase not directly
quoted this session.
S1-S6: PASS; S3 correctly offers the College's enrollment growth (12 to 26 students) as the only
directly attested numeric trend, without extrapolating it into a missionary-sending total, which
the sources do not support.
R1-R4: PASS; R2 (Pratt) is correctly qualified as "not independently extracted as a quote record
this session" even though it appears in the dossier's prose, distinguishing narrative color from
a cited fact.

## 4. Secondary citations checked
No `[SECONDARY: ...]` citation is used in this dossier for a fact stated outright; Stock's
narrative is drawn on directly (not via a secondary citation marker) for the Islington College
passages, which sources.csv explicitly treats as an eyewitness institutional record and therefore
primary, and for Venn's summarized principles, likewise treated primary per sources.csv's own
edition notes (quoted directly from "the original Minutes," per Stock). This dossier-level
judgment call — treating Stock's summary of Venn's address as primary rather than secondary — is
noted here for the human reviewer's awareness; it does not affect the verdict below, since no
fact in the dossier is mis-labeled as PRIMARY-ONLY where only a secondary source supports it.

## 5. Dates and numbers vs known-facts sheet
- 1799 founding; college 1825, per roster [HIGH]: dossier gives 12 April 1799 (public meeting)
  and 31 January 1825 (Islington inauguration) specifically. PASS, consistent.
- Founders (Simeon, Venn, Wilberforce, the Eclectic Society, per roster) vs. dossier: Venn chairs
  the founding meeting and authors the Rules and Account per Stock (F3, R1); Wilberforce declines
  the presidency and becomes a Vice-President (F3, narrative, not independently quoted this
  session); Simeon is named by Stock as absent from the first meeting (footnote text within
  cms-q003/q004's surrounding passage), not as a founder present on 12 April 1799 -- no conflict
  with the roster, which lists him among the wider Eclectic Society circle, not specifically as
  present at the 12 April meeting.

## 6. Verdict
VERIFIED (zero FAILs on the 12 quote records and, after the two `[...]` corrections recorded
above, on the 8 charter passages). No RETURNED items.
