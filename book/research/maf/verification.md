STATUS: VERIFIED
TICKET: V1-maf
ROLE: Verifier
INPUTS READ: research/maf/dossier.md; research/maf/quotes.jsonl; research/maf/charter_abridged.md; research/maf/discrepancies.md; research/maf/sources.csv; research/maf/text/*.txt; book/plan/02_institution_roster.md (roster row, maf)

# Verification report — maf

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/maf
WARN maf-q016, maf-q018, maf-q034, maf-q035, maf-q039, maf-q040, maf-q041: before+text+after
not contiguous (markdown link boundaries / inline citation brackets / archival-folder citation
brackets sitting between the quote and its stored context window)
50/50 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/maf --charter
1/1 passages verbatim; 0 failures
```

All WARN lines are non-contiguity, not text-accuracy, failures — caused by Wikipedia inline
citation markers, markdown link boundaries, or (for the Wheaton Archives blog quotations) the
archival folder-citation brackets the archivist herself inserted immediately after each
quotation (e.g., "[Letter from Greene, June 4, 1946. MAF Records, Folder 1-9]"). All 50 quote
records were built by a script that locates the `text` field as an exact substring of the
corresponding `text/<document_id>.txt` file and computes `before`/`after` programmatically
(never retyped from memory), so verbatim matching is true by construction and independently
re-confirmed by the script run above. The single charter passage (MAF-US's current, seven-
point Statement of Faith, printed whole) was extracted directly from
`text/maf-statement-of-faith.txt` by exact line range (`sed`), guaranteeing character-for-
character fidelity including curly apostrophes.

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| maf-q001–maf-q050 (all 50 present) | PASS (script) | PASS (script; WARN-only for 7 ids) | PASS (all single-page web sources, "p. 1" marker) | PASS |

Spot-checked by hand against the fetched text: maf-q006, maf-q007, maf-q015, maf-q027,
maf-q035, maf-q036, maf-q039, maf-q049.

## 3. Field-by-field

| field | result | note |
|---|---|---|
| F1–F4 | PASS | Summaries state the 1945 sequence and explicitly carry the three-name/two-date discrepancies (discrepancies.md #1–2) into the dossier rather than silently resolving them. |
| F5 | PASS | Correctly marked NOT FOUND; directs to the flagged current-substitute Statement of Faith and Vision/Mission text in charter_abridged.md. |
| F6–F7 | PASS | F6 correctly notes no single founder is quoted directly for the "servant of missions" self-description; F7 supported by two independently worded but agreeing sources. |
| F8 | PASS | Correctly frames the current Statement of Faith as undated-to-1945. |
| F9 | PASS | Correctly notes the direct operational link to the Wycliffe/SIL institution covered elsewhere in this book (Cameron Townsend's own flight), distinct from a doctrinal antecedent claim. |
| A1–A4, A6 | PASS | Correctly marked NOT FOUND with searched-scope noted. |
| A5 | PASS | The fullest-supported field in the dossier: Greene's own words on MAF's reluctance to recruit women pilots are quoted directly and not softened or omitted, alongside her own record-setting service. |
| C1–C9 | PASS | Each is either supported by a specific quote_id or correctly marked NOT FOUND/not applicable, correctly distinguishing MAF's character as an operating service agency from the seminary/institute model of this book's other chapters. |
| L1–L6 | PASS | Correctly marked NOT FOUND or supported only by loosely analogous facts (office donation, Shell Mera house), not overstated as formal policy. |
| L7 | PASS | Uses 2010 fleet/passenger figures as an organizational-size proxy, explicitly flagged as "not applicable in the school sense." |
| L8 | PASS | Supported directly by Greene's own 1946 letter on carrying a firearm. |
| T1–T5 | PASS | Correctly marked NOT FOUND in the classroom sense that this book's other chapters use, with T3 substituting the correspondence-based mentorship actually attested. |
| M1–M5 | PASS | Each quotation is attributed to its actual speaker or source; M5 correctly frames the post-1956 reconciliation account as a later restatement, not a founding-era fact. |
| S1–S6 | PASS | S5 is the most extensively supported field in the dossier: eleven separate casualty events (1951 Hartwig; 1956 Ecuador five; six further wiki-maf-listed crashes through 2020) each trace to a specific quote_id, none invented or embellished beyond what the cited text states. |
| R1–R4 (four named figures) | PASS | Greene, Saint, Hartwig, and Lin each carry specific quote_ids. |
| X1–X3 | PASS | Epigraph and Emphasis picks trace to real quote_ids; X3 correctly explains the Statement of Faith's short length justifies printing it whole. |

No field states a fact its cited quotes do not support.

## 4. Secondary citations checked

| citation | page says what is claimed? | result |
|---|---|---|
| maf-history, maf-betty-greene, maf-vision-mission, maf-statement-of-faith, maf-nate-saint, maf-5-martyred (MAF-US's own website) | Yes — verbatim, script-confirmed. | PASS |
| wheaton-gal-plane-dream (Wheaton College Archives & Special Collections, "From the Vault" blog) | Yes — verbatim, script-confirmed; the archivist's own folder citations for each archival quotation are reproduced accurately in the footnotes, and this session did not claim to have independently opened the underlying archival letters themselves — only the blog's own transcription of them. | PASS |
| wiki-maf, wiki-bettygreene, wiki-natesaint (Wikipedia) | Yes — verbatim, script-confirmed; the "Accidents" section's own footnotes (to sources such as aviation-safety databases, not independently opened this session) are not claimed as independently verified beyond the Wikipedia text itself. | PASS |
| mafint-history, mafau-80years (MAF International and MAF Australia websites) | Yes — verbatim, script-confirmed; correctly used to surface, not paper over, the US/UK founding-date and founder-list discrepancies. | PASS |

## 5. Dates and numbers vs known-facts sheet

| item | dossier value | roster value | result |
|---|---|---|---|
| MAF (US) founding | 20 May 1945, Los Angeles | "1945 [HIGH]" | MATCH |
| Founders | Betty Greene, Jim Truxton, Grady Parrott (roster); this session also finds Charlie Mellis independently corroborated by two sources | "Betty Greene, Jim Truxton, Grady Parrott" | MATCH; the additional Mellis reading is recorded, not substituted for the roster's three names |
| Place | Los Angeles | "Los Angeles" | MATCH |
| Nate Saint / Auca connection | Confirmed: MAF pilot, killed 8 January 1956 with four other missionaries (Elliot, McCully, Youderian, Fleming) attempting to reach the Waodani | "Nate Saint of the Auca five was MAF" | MATCH |

## 6. Verdict

VERIFIED (zero FAILs). Coverage is limited — no founding-era constitution or statement of
purpose was located this session, Russell Hitt's *Jungle Pilot* was BLOCKED by Internet
Archive's controlled digital lending across all four editions checked, and the Billy Graham
Center Archives' MAF finding aid was BLOCKED by a Cloudflare gate. These gaps are disclosed,
not papered over, in `dossier.md`'s coverage note, `charter_abridged.md`'s coverage note, and
`sources.csv`'s BLOCKED rows with a `.REQUEST.md` follow-up file. No dossier field asserts a
fact beyond what its cited quotations support, and the founding-date/founder-list
discrepancies found this session are reported as unresolved rather than silently settled.
