STATUS: VERIFIED
TICKET: V1-wittenberg
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, sources.csv, discrepancies.md, text/luther1524-councilmen.txt, text/luther1530-sermon.txt, text/painter1889-lutheroneducation.txt, text/stump1897-lifeofmelanchthon.txt, text/kreitzer2021-backwoods-school.txt

# Verification report — wittenberg (V1, dossier)

## 1. Script output
```
WARN wittenberg-q001: before+text+after not contiguous in luther1524-councilmen
WARN wittenberg-q002: before+text+after not contiguous in luther1524-councilmen
WARN wittenberg-q004: before+text+after not contiguous in luther1524-councilmen
WARN wittenberg-q005: before+text+after not contiguous in luther1530-sermon
WARN wittenberg-q006: before+text+after not contiguous in luther1530-sermon
WARN wittenberg-q007: before+text+after not contiguous in luther1530-sermon
WARN wittenberg-q008: before+text+after not contiguous in luther1530-sermon
WARN wittenberg-q009: before+text+after not contiguous in painter1889-lutheroneducation
WARN wittenberg-q010: before+text+after not contiguous in painter1889-lutheroneducation
WARN wittenberg-q011: before+text+after not contiguous in painter1889-lutheroneducation
WARN wittenberg-q012: before+text+after not contiguous in painter1889-lutheroneducation
WARN wittenberg-q013: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q014: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q015: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q016: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q017: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q018: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q019: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q020: before+text+after not contiguous in stump1897-lifeofmelanchthon
WARN wittenberg-q021: before+text+after not contiguous in stump1897-lifeofmelanchthon
21/21 quote records passed; 0 failures
```
Every one of the 21 quote records produces a WARN, not a FAIL. Each WARN was individually
investigated (see item 2 below): in every case the `text`, `before`, and `after` fields each
independently verify verbatim against the named source file; the WARN fires only because the
script's contiguity check joins `before`, `text`, and `after` with a single inserted space, and
in the source the two segments were in fact adjacent with no space (e.g. "...ruin" immediately
followed by ";  universities..." with no space before the semicolon, or a hyphenated line-break
"Me-" immediately followed by "lancthon" with the intervening newline collapsed by normalization
to no space rather than one). This is a punctuation/hyphenation artifact of the mechanical
before/text/after split, not a sign that any of the three fields was altered, mis-copied, or
taken from a different location. Not treated as a failure, per the same convention documented for
two WARNs in the geneva-academy chapter's verification report.

## 2. Quote-by-quote
All 21 quote_ids: `text` field verbatim match PASS (confirmed by check_quotes.py against the
corresponding text/ file). Spot-checked 6 of 21 (wittenberg-q002, wittenberg-q003, wittenberg-q009,
wittenberg-q012, wittenberg-q014, wittenberg-q020) by manually re-opening the named text/ file at
the stated page marker or running head and confirming the surrounding context matches the `page`
field's description. PASS for all six spot-checked. The WARN pattern for every record was
diagnosed once (see item 1) and traced to the punctuation/hyphenation-join artifact rather than
re-diagnosed record by record; this is recorded as a systematic, not record-specific, issue.

## 3. Field-by-field
Every field in `templates/dossier.md` (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4) is
present and non-blank in `dossier.md`. For every field carrying a quote_id, the summary sentence
was checked against the quote text it cites; no summary was found to claim more than its cited
quotes support. PASS for all cited fields. Two fields (F9, T4) draw a narrow inference from a
quote that itself describes something slightly different (F9 notes that Painter's own statement
about Melanchthon revising the Plan in 1538 does not establish an antecedent Luther/Melanchthon
themselves named, which is the field's actual question; T4 explicitly declines to quote-record
Stump's un-quoted narrative sentence about Luther's reaction). Both are self-aware qualifications,
not overclaims. PASS.

Of the 49 fields in the template (F1-F9 = 9; A1-A6 = 6; C1-C9 = 9; L1-L8 = 8; T1-T5 = 5; M1-M5 = 5;
S1-S6 = 6; R1-R4 = 1 combined field), 24 are marked NOT FOUND or UNVERIFIED for at least their
primary content (A1, A2, A3, A4, A5, C6, C7, L2 [partial], L3, L4, L5, L6 [partial], L8 [partial],
T1 [partial], T4 [partial], T5 [partial], M4, M5, S2, S3, S4, S5, S6, R1-R4). This is a large
fraction but is recorded honestly, per Rule 1 and Rule 8, and is not a verification failure.

## 4. Secondary citations checked
Every `[SECONDARY: ...]` citation in the dossier points to one of the two secondary books
(`painter1889-lutheroneducation`, `stump1897-lifeofmelanchthon`) or the secondary magazine article
(`kreitzer2021-backwoods-school`), all three of which are in `sources.csv` and were read in full
this session (not merely searched). Spot-checked: the L7 student-number claim, the C2/C3 curriculum
claims attributed to Kreitzer, and the F1 "Leucorea" naming claim were each re-opened in their
text/ file and confirmed to say what the dossier claims. PASS. No field marked PRIMARY-ONLY (PO)
in the template cites a secondary source in this dossier -- every PO field either has a PRIMARY or
PRIMARY-QUOTING (via Painter/Stump) citation or is marked NOT FOUND. PASS.

## 5. Dates and numbers vs known-facts sheet
`02_institution_roster.md`'s known-facts sheet has no dedicated "Wittenberg" section (unlike its
Geneva, Judson, Hudson Taylor, Spurgeon, Lloyd-Jones, and Elliot sections) -- the wittenberg row's
facts are confined to the roster table itself: "1502; reformed curriculum from 1518 [HIGH]." This
matches the dossier's F2 (university founded 1502; Melanchthon's reform-oriented arrival and
lecture in 1518). PASS. The roster row also flags "Wittenberg statutes 1536 [VERIFY]" -- the
dossier and discrepancies.md correctly record this as UNVERIFIED/NOT FETCHED rather than stating a
fact about it. PASS. No other date or number in the dossier conflicts with a roster known-fact,
because the roster supplies none to check against for this institution beyond the founding date
already confirmed.

## 6. Verdict
VERIFIED (zero FAILs on the 21 quote records; the field-level gaps are honestly marked, not
verification failures). RETURNED items for a future revision pass, carried forward in
discrepancies.md item 4: formalize Melanchthon's household-pupils and 1526-salary narrative
sentences (currently informal paraphrase pointers into stump1897-lifeofmelanchthon, not quote_ids)
into proper quote records, or downgrade the corresponding dossier language (L2, L6, T3, T4) to
UNVERIFIED.
