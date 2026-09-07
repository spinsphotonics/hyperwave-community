STATUS: VERIFIED
TICKET: V1-new-tribes-mission
ROLE: Verifier
INPUTS READ: research/new-tribes-mission/dossier.md; research/new-tribes-mission/quotes.jsonl; research/new-tribes-mission/charter_abridged.md; research/new-tribes-mission/discrepancies.md; research/new-tribes-mission/sources.csv; research/new-tribes-mission/text/*.txt; book/plan/02_institution_roster.md (roster row, new-tribes-mission)

# Verification report — new-tribes-mission

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/new-tribes-mission
WARN ntm-q003, ntm-q013, ntm-q016, ntm-q026, ntm-q028, ntm-q030, ntm-q031, ntm-q032, ntm-q033,
ntm-q043, ntm-q050: before+text+after not contiguous (inline citation markers / markdown link
boundaries / paragraph breaks between quote and stored context)
51/51 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/new-tribes-mission --charter
3/3 passages verbatim; 0 failures
```

All WARN lines are non-contiguity caused by markdown link boundaries, blog-post paragraph
breaks, or inline citation punctuation sitting between the quoted text and its stored
before/after context window — not a text-accuracy failure. All 51 quote records were built by
a script that locates the `text` field as an exact substring of the corresponding
`text/<document_id>.txt` file and computes `before`/`after` programmatically (never retyped
from memory), so verbatim matching is true by construction and independently re-confirmed by
the script run above. The three charter passages (Ethnos360's current "About" page, with the
Training/Partnerships sections cut and two pairs of decorative image lines cut as `[...]`)
were extracted directly from `text/ethnos360-about.txt` by exact line range (`sed`),
guaranteeing character-for-character fidelity.

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| ntm-q001–ntm-q054 (51 present; ntm-q010, ntm-q020, ntm-q021 dropped before quotes.jsonl was written — see discrepancies.md note and scout_notes.md) | PASS (script) | PASS (script; WARN-only for 11 ids) | PASS (all single-page web sources, "p. 1" marker) | PASS |

Spot-checked by hand against the fetched text: ntm-q007, ntm-q008, ntm-q018, ntm-q027,
ntm-q028, ntm-q032, ntm-q036, ntm-q049.

## 3. Field-by-field

| field | result | note |
|---|---|---|
| F1–F4 | PASS | Summaries state the 1942/1943/1944-45 sequence and correctly flag the unresolved Fredonia, Wisconsin claim (discrepancies.md #2) rather than asserting it. |
| F5 | PASS | Correctly marked NOT FOUND as a complete text; directs to the flagged 1942 Pledge excerpt and current-substitute in charter_abridged.md. |
| F6–F7 | PASS | F6 quotes both the Pledge and Fleming's separate statement; F7 is explicitly framed as Fleming's personal call narrative, not an institutional crisis diagnosis, since no source states the latter. |
| F8 | PASS | Correctly frames the current Statement of Faith as undated-to-1942; notes the Pledge itself is a purpose-pledge rather than a doctrinal confession. |
| F9 | PASS | Supported by two independent quote_ids on Fleming's pre-founding Malaya service. |
| A1–A4, A6 | PASS | Correctly marked NOT FOUND with searched-scope noted; A6 correctly distinguishes Dye's own prior pastoral experience from a stated rule for others. |
| A5 | PASS | Names three women (Jean Dye Johnson, Audrey Bacon, Dorothy Dye) without asserting a formal admission rule not found in the sources. |
| C1–C9 | PASS | Each is either supported by a specific quote_id or correctly marked NOT FOUND, with C1/C3/C8 correctly distinguishing the founding-era "boot camp" (undocumented in detail) from the later, separately founded Ethnos360 Bible Institute (1955) and Canadian program (1968). |
| L1, L3, L4, L5 | PASS | Correctly marked NOT FOUND. |
| L2, L6, L7 | PASS | Supported by specific quote_ids (Chicago nightclub, Fouts Springs, Durham property, 1968 seven students); L6 correctly frames the founders'-own-poverty quotation as a fact about the founders rather than a formal missionary-support policy. |
| L8 | PASS | Supported by Jean Dye Johnson's own words on the years-long uncertainty. |
| T1–T5 | PASS | Correctly marked NOT FOUND or partially supported (T4 uses Cecil Dye's letter to his own men as the closest available named interaction, clearly distinct from a classroom teacher-student relationship). |
| M1–M5 | PASS | Each quotation is attributed to its actual speaker (Fleming, Dye, Walker) and dated where a date is known (Brown Gold, May 1943; the 2017 restatements). |
| S1–S6 | PASS | S3 explicitly carries the unresolved 2,000-vs-3,000 figure into the chapter rather than picking one (discrepancies.md #1); S5's five-death, two-plane-crash, and Rattlesnake-Fire figures each trace to independently footnoted Wikipedia sentences, cross-checked against the ethnos360-wont-come-back narrative for the Bolivia deaths. |
| R1–R4 (four named figures) | PASS | Fleming, Dye, Jean Dye Johnson, and Audrey Bacon each carry specific quote_ids. |
| X1–X3 | PASS | Epigraph and Emphasis picks trace to real quote_ids; X3 correctly identifies the 62-word Pledge excerpt as the only genuinely founding-era text available and notes it supplements rather than replaces the current-document substitute. |

No field states a fact its cited quotes do not support.

## 4. Secondary citations checked

| citation | page says what is claimed? | result |
|---|---|---|
| e360-founding, e360-founded-1942, ethnosca-heritage, ntmuk-heritage, ethnos360-about, ethnosca-whatwebelieve, ethnos360-namechange, ethnos360-wont-come-back, housley-founding-fathers (Ethnos360/NTM-family websites) | Yes — verbatim, script-confirmed for all quoted material. | PASS |
| wiki-ntm (Wikipedia, Ethnos360/New Tribes Mission article) | Yes — verbatim, script-confirmed; each casualty-record sentence used is independently footnoted on the live Wikipedia page (to Time magazine and other sources not independently opened this session, appropriately not claimed as directly verified beyond the Wikipedia text itself). | PASS |
| riches-item520 (RICHES digital exhibit, Univ. of Central Florida) | Yes — verbatim; correctly used only for the 1977 headquarters-building fact, not conflated with founding-era (1942) material. | PASS |
| housley-founding-fathers (staff personal blog, quoting the 1942 Pledge and Covenant and Cecil Dye's letters) | Yes — verbatim; the page's own disclaimer (personal ministry content, not necessarily official Ethnos360 doctrine) is noted in sources.csv and not glossed over. | PASS |

## 5. Dates and numbers vs known-facts sheet

| item | dossier value | roster value | result |
|---|---|---|---|
| NTM founding | 1942, spring; first group sent to Bolivia November 1942 | "1942 [HIGH]" | MATCH |
| Founder(s) | Paul Fleming and Cecil A. Dye named as the founding pair by two sources; a third source additionally names Lance B. Latham and M. Robert Williams; discrepancy recorded rather than resolved | "Paul Fleming" | MATCH on Fleming; the roster names only Fleming, and the dossier does not silently drop the additional-founders reading found independently, per Rule 9 |
| Location | Chicago (founding); Chico, California and Fouts Springs, California (1944/45 boot camp); Fredonia, Wisconsin NOT corroborated | "Chicago; Fredonia, Wis. boot camp" | PARTIAL — Chicago confirmed; Fredonia, Wis. could not be corroborated by any source fetched this session and is recorded as an open discrepancy (discrepancies.md #2) rather than silently affirmed or silently dropped |
| "Boot camp" training model | Confirmed as a named missionary training facility (Fouts Springs, California, per Wikipedia); founding-era manual/curriculum text NOT located | "its own 'boot camp'" [roster's "why included"] | MATCH on the boot-camp concept; the underlying manual is honestly marked NOT FOUND rather than fabricated |

## 6. Verdict

VERIFIED (zero FAILs). Coverage is limited — no complete 1940s training manual or doctrinal
statement was located this session, and the roster's Fredonia, Wisconsin claim could not be
corroborated. These gaps are disclosed, not papered over, in `dossier.md`'s coverage note,
`charter_abridged.md`'s coverage note, and `sources.csv`'s NOT FOUND/NOT FETCHED rows. No
dossier field asserts a fact beyond what its cited quotations support, and the founder-count
and membership-figure discrepancies inherited or discovered this session are reported as
unresolved rather than silently settled.
