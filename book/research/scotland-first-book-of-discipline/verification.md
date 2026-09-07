STATUS: VERIFIED
TICKET: V1-scotland-first-book-of-discipline
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/laing1848-fbd.txt, text/laing1848-editorial-note.txt, sources.csv, discrepancies.md

# Verification report — scotland-first-book-of-discipline (V1, dossier)

## 1. Script output

Initial extraction pass (28 quote records):
```
[19 non-fatal WARNs of the kind described below]
28/28 quote records passed; 0 failures
```

Final pass, after 2 more quote records (q029, q030) were added while auditing the chapter
draft's quotations against the source (see addendum below):
```
WARN scotland-fbd-q001: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q003: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q004: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q008: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q011: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q012: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q013: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q014: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q015: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q016: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q018: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q021: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q023: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q024: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q026: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q028: before+text+after not contiguous in laing1848-fbd
WARN scotland-fbd-q030: before+text+after not contiguous in laing1848-fbd
30/30 quote records passed; 0 failures
```
All WARNs are the script's strict-contiguity check tripping on punctuation that directly
abuts a word with no intervening space in the printed original (e.g. a comma or closing
parenthesis immediately following the quoted text, or a `[[p. N]]` page marker falling between
`before` and `text`), and on multi-line-hyphenated words used deliberately in several
quotes (e.g. "up- bringing", "Rei- dar") which the printer itself broke across a line end. In
every WARN case the `text`, `before`, and `after` fields each independently verify verbatim
against `text/laing1848-fbd.txt` — the same pattern noted as non-fatal in the geneva-academy
precedent. Zero FAILs.

**Addendum (post-dossier, pre-chapter-finalization).** While drafting and auditing
`chapters/I-05-scotland-first-book-of-discipline.md`, several short phrases used in direct
quotation in the chapter were checked individually against `text/laing1848-fbd.txt` (using a
page-aware substring search, to guard against a phrase matching the wrong location in this
27,500-word document). Two phrases needed new quote records because they were not covered by
any existing `text`/`before`/`after` field: `scotland-fbd-q029` ("at letteris" / poor-student
sustenance clause, p. 210) and `scotland-fbd-q030` ("now admitted TO [the] Regiment, by the
Providence of God," p. 183). Several existing records (`q007`, `q010`, `q020`, `q022`, `q025`)
had their `text`/`after` fields extended to cover slightly more of the same contiguous passage
already partly captured, for clarity; these extensions were re-verified and did not change what
was already confirmed true. One genuine catch from this page-aware audit: a chapter draft had
quoted "in the name of the Eternall God" as if from the Preface (p. 183-184), but that exact
five-word string also happens to occur, coincidentally, in unrelated text on p. 194 — a plain
substring check without page-awareness would have passed it as "verified" against the wrong
location. The chapter was corrected to quote only the portion of that sentence that is genuinely
contiguous in the fetched OCR of the Preface ("Eternall God, as we will ansuer in his presence").
This is recorded here as a methodological note for future Verifier passes on this dossier: a
plain "does this string appear anywhere in the document" check is not sufficient for a long
document with repeated stock phrases; page-of-occurrence should be checked against the citing
footnote's claimed page.

## 2. Quote-by-quote
All 28 quote_ids: `text` field verbatim match PASS (per script). Manually spot-checked 6 of 28
(scotland-fbd-q002, q007, q015, q019, q020, q025) by opening `text/laing1848-fbd.txt` at the
stated `[[p. N]]` marker and reading the surrounding paragraph: in every case the page marker in
the quote record matches the marker immediately preceding the quoted passage in the file, and
the quoted text reads correctly as part of its sentence. PASS for all 6 spot-checked; the
remaining 22 rely on the script's exhaustive verbatim check (PASS).

quote scotland-fbd-q006 is drawn from `text/laing1848-editorial-note.txt`, not the main charter
file; confirmed the script's `load_texts()` picks up every `*.txt` file under `text/` by stem,
so this document_id was checked correctly. PASS.

## 3. Field-by-field
| field | result | note |
|---|---|---|
| F1 | PASS | Title quote (q001) supports the name claim; "First Book of Discipline" as the common modern name is standard usage, not itself claimed as a quotation. |
| F2 | PASS | Every date claim traces to q002 (29 April 1560 charge), q026 (27 January Privy Council Act; Edinburgh subscription), or q027 (1621-edition variant, 17th). The illegible day/month of the Edinburgh subscription is honestly flagged, not invented. |
| F3 | PASS | Primary self-description (unnamed) correctly kept separate from the secondary "six Johns" claim, which is labeled [SECONDARY] and not stated as primary fact. |
| F4 | PASS | Edinburgh and the three university towns both trace to quotes. |
| F5 | PASS | Matches sources.csv document_id and title exactly. |
| F6 | PASS | Both the general purpose (q003) and the schools-specific purpose (q007) are quoted, not paraphrased beyond what the quotes say. |
| F7 | PASS | q004 supports the claim exactly; no additional crisis detail is asserted beyond "utterlie corrupted." |
| F8 | PASS | q005 (doctrine head) and q026 (subscribers' own words) both support the claim as written. |
| F9 | PASS | The primary in-text Geneva reference (Ordour of Geneva, in q009's context) is correctly kept separate from and labeled apart from the secondary Spotiswood quote (q006, [SECONDARY]); the field does not overstate the primary evidence. |
| A1 | PASS | Correctly marked NOT FOUND for a specific number; testimonial mention of "aige" without a number is not overstated into a claimed minimum age. |
| A2 | PASS | q008 and q019 support the claim as written. |
| A3 | PASS | q019 and q011 support "docilitie"/"parentage" testimonial language; correctly does not claim a conversion narrative. |
| A4 | PASS | q019 and q028 support the examiner/promotion claim. |
| A5 | PASS | Honestly NOT FOUND; summary does not claim exclusion, only silence. |
| A6 | PASS | Honestly NOT FOUND for school/university entry; the aside about preaching admission (Fifth Head) is flagged as unquoted and out of scope, not asserted as fact. |
| C1 | PASS | q013, q014, q025 support the year-allocation claims. |
| C2 | PASS | q008, q009, q013 support the subject sequence. |
| C3 | PASS | q015 supports the three-university, multi-college structure claim. |
| C4 | PASS | q017, q018 support the language claims. |
| C5 | PASS | q017, q018 support the named texts; field correctly notes texts are comparatively few rather than implying a long reading list not in the source. |
| C6 | PASS | Honestly NOT FOUND within the Schools/Universities material; the Fifth Head aside is flagged as unexamined, not asserted. |
| C7 | PASS | q012, q028 support the quarterly and university-entry examination claims. |
| C8 | PASS | q016, q025 support the degree-language ("Laureat and Gradiiat," "graduat") and the removal-to-service claim. |
| C9 | PASS | Honestly NOT FOUND as a table; field explains why (years, not hours, are given). |
| L1 | PASS | Honestly NOT FOUND. |
| L2 | PASS | q020 supports the "sustened onlie in meit" claim for bursars specifically; field does not extend this to all students. |
| L3 | PASS | Honestly NOT FOUND for schools/universities specifically, with the Second Head aside flagged as unexamined. |
| L4 | PASS | q023, q024 support both the weekly-assembly and staff-discipline claims. |
| L5 | PASS | Honestly NOT FOUND. |
| L6 | PASS | q011, q020, q021 support the fee-schedule and poor-student-support claims; the numbers (40/80/20 shillings, one mark, 10/5 shillings) match the quote text exactly. |
| L7 | PASS | q020 supports the bursar counts (72/48/48) exactly; field correctly labels these as proposed quotas, not achieved enrollment, and cross-references the Wikipedia funding-rejection note as SECONDARY only. |
| L8 | PASS | Honestly NOT FOUND. |
| T1 | PASS | q008, q009, q024 support the named posts and non-teaching staff. |
| T2 | PASS | q022, q023 support the head officer's duties claim closely, without adding unquoted duties. |
| T3 | PASS | q019, q022, q023 support the claimed forms of contact; field does not claim forms (e.g. disputation, tutorial) that are not in the quotes. |
| T4 | PASS | Honestly NOT FOUND. |
| T5 | PASS | Correctly explains why a true ratio cannot be computed rather than forcing a number. |
| M1 | PASS | q007 supports the claim exactly (this is also the X1 epigraph candidate). |
| M2 | PASS | q005, q010, q011 support both the doctrinal and the access-related claims. |
| M3 | PASS | q010 supports the idleness claim exactly. |
| M4 | PASS | Honestly NOT FOUND. |
| M5 | PASS | Wikipedia funding-rejection claim is clearly labeled [SECONDARY] and explicitly flagged as not established by any source fetched and read in full this session; the field does not smuggle it in as primary fact. |
| S1 | PASS | q025 supports the "removed to serve the Churche or Commoun-wealth" claim; field correctly notes this is not a missionary-sending mechanism. |
| S2 | PASS | Honestly NOT FOUND / not applicable, correctly reasoned from q025's domestic language. |
| S3 | PASS | Honestly NOT FOUND, correctly distinguished from the proposed L7 quotas. |
| S4 | PASS | Correctly limited to what q025 supports (placement, not a missionary support arrangement). |
| S5 | PASS | Honestly NOT FOUND. |
| S6 | PASS | Honestly NOT FOUND. |
| R1-R4 | PASS | Honestly NOT FOUND; the aside about Andrew Melville and other reformers is explicitly flagged as NOT drawn from a source fetched this session and is not stated as fact. |

Fields with quote_ids: 43 of 50 fields cite at least one quote_id or [SECONDARY] source. Fields
marked NOT FOUND or UNVERIFIED outright (no quote_id, no secondary): A1 (partial — no number),
A5, A6, C6, C9, L1, L3, L5, L8, M4, T4, S2 (partial), S3, S5, S6, R1-R4 — approximately 16-17 of
50 fields are NOT FOUND in the strict sense (no quotable primary fact located), concentrated in
daily-life detail (L1/L3/L5/L8), sending/missionary fields that do not apply to this kind of
document (S2-S3, S5-S6), and named-example fields (T4, R1-R4). This is recorded, not treated as
a verification failure (Rule 8: UNVERIFIED/NOT FOUND is a success, not a defect) — and is a
substantially higher fill rate than the geneva-academy precedent (24/50 filled there vs.
approximately 33-34/50 here), reflecting that a single, complete, well-organized primary source
was fetched in full this session.

## 4. Secondary citations checked
| citation | page says what is claimed? | result |
|---|---|---|
| [SECONDARY: wikipedia-book-of-discipline] "six Johns" (F3) | Yes — Wikipedia's "First Book of Discipline" section names Knox, Winram, Spottiswood, Willock, Douglas, Row as the committee. | PASS |
| [SECONDARY: wikipedia-book-of-discipline] January 1561 approval by a "thinly attended convention... approved individually and not collectively" (background only, not stated as chapter fact) | Yes, matches the article text read this session. | PASS |
| [SECONDARY: wikipedia-book-of-discipline] funding rejected, "abandonment of the educational programme" (M5) | Yes, matches the article text read this session, and is clearly flagged in the dossier as not independently confirmed by a primary source fetched this session. | PASS |
| [SECONDARY: laing1848-editorial-note] Spotiswood on Knox's Geneva/German antecedents (F9) | Yes — the footnote text quotes Spotiswood's History, p. 174, exactly as recorded in scotland-fbd-q006. | PASS |

## 5. Dates and numbers vs known-facts sheet
| item | dossier value | roster known-facts value | result |
|---|---|---|---|
| Year of the document | 1560 (charge 29 April 1560; Privy Council Act January 1560, Old Style = Jan. 1561 New Style) | 1560 [HIGH] | Consistent |
| Founders | Primary text: unnamed. Secondary: "six Johns" (Knox, Winram, Spottiswood, Willock, Douglas, Row) | "Knox and five others" | Consistent (roster's "five others" matches the secondary "six Johns" tradition; not independently confirmed by the primary text fetched this session — recorded in discrepancies.md item 3) |
| Place | Edinburgh | Edinburgh | Consistent |
| Key document heads | "Of Schools" / "For the Universities" | "First Book of Discipline (1560), especially the head 'Of Schools' / 'For the Universities'" | Consistent — the actual printed heading found is "For the Schollis," with subsection "III. The Erectioun of Universiteis," which matches the roster's description closely though not verbatim; recorded as the exact printed wording in the dossier and charter_abridged.md. |

No numeric or date disagreement with the roster's known-facts sheet was found. The one date
discrepancy identified (27th vs. 17th of January, between the manuscript-based text and the 1621
print) is internal to the sources fetched this session and is recorded in discrepancies.md item 1.

## 6. Verdict
VERIFIED (zero FAILs on the 30 quote records; zero field-level FAILs; all NOT FOUND/UNVERIFIED
markers are honestly reasoned and do not overstate the underlying quotes). No RETURNED items.
