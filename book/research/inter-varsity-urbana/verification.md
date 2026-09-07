STATUS: VERIFIED
TICKET: V1-inter-varsity-urbana
ROLE: Verifier
INPUTS READ: research/inter-varsity-urbana/dossier.md; research/inter-varsity-urbana/quotes.jsonl; research/inter-varsity-urbana/charter_abridged.md; research/inter-varsity-urbana/discrepancies.md; research/inter-varsity-urbana/sources.csv; research/inter-varsity-urbana/text/*.txt; book/plan/02_institution_roster.md (roster row, inter-varsity-urbana)

# Verification report — inter-varsity-urbana

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/inter-varsity-urbana
WARN ivu-q021: before+text+after not contiguous in wiki-urbana
WARN ivu-q023: before+text+after not contiguous in wiki-urbana
46/46 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/inter-varsity-urbana --charter
1/1 passages verbatim; 0 failures
```

The two WARN lines are non-contiguity caused by inline Wikipedia footnote markers
(`[6]`, `[8]`) and adjacent markdown links sitting between the quoted text and its stored
before/after context — not a text-accuracy failure. All 46 quote records were built by a
script that locates the `text` field as an exact substring of the corresponding
`text/<document_id>.txt` file and computes `before`/`after` programmatically (never retyped
from memory), so verbatim matching is true by construction and independently re-confirmed by
the script run above. The single charter passage (the current Statement of Agreement,
IVCF/USA's Basis of Faith and Purpose) was extracted directly from `text/ivcf-statement-agreement.txt`
by exact line range (`sed`), guaranteeing character-for-character fidelity including curly
apostrophes and markdown blockquote markers.

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| ivu-q001–ivu-q047 (all 46 present; ivu-q022 dropped before quotes.jsonl was written, no record exists for it) | PASS (script) | PASS (script; WARN-only for ivu-q021, ivu-q023) | PASS (all single-page web sources, "p. 1" marker) | PASS |

Spot-checked by hand against the fetched text: ivu-q001, ivu-q010, ivu-q019, ivu-q023,
ivu-q032, ivu-q038, ivu-q040, ivu-q045.

## 3. Field-by-field

| field | result | note |
|---|---|---|
| F1–F4 | PASS | Summaries state the 1877/1928/1938/1941 sequence and the founding-date discrepancy (see discrepancies.md #1–2) rather than asserting a single date as settled fact. |
| F5 | PASS | Correctly marked NOT FOUND; directs to the flagged present-day substitute in charter_abridged.md rather than overstating coverage. |
| F6 | PASS | Correctly notes the 1947 IFES purpose statement is for the international federation, not IVCF/USA specifically, and is the only founder-generation purpose statement found. |
| F7 | PASS | Correctly marked NOT FOUND as a founders' own statement; distinguishes this from the secondary narrative background offered instead. |
| F8 | PASS | Correctly frames the current Statement of Agreement as undated-to-1941; does not claim continuity of wording without a citation. |
| F9 | PASS | Supported by seven independent quote_ids on the Cambridge/Guinness/Woods/SVM lineage. |
| A1–A4 | PASS | Correctly marked NOT FOUND with searched-scope noted; A1 correctly distinguishes a 2012 policy statement from a founding-era rule. |
| A5 | PASS | Summary states named women's roles without asserting a formal admission rule not found in any source. |
| A6 | PASS | Correctly distinguishes Finley's 1945 staff hire from a general admission rule. |
| C1–C9 | PASS | Each is either supported by a specific quote_id or correctly marked NOT FOUND/not applicable, with the C1 "not applicable" framing matching this institution's actual character (a fellowship, not a school with a course of study). |
| L1, L4, L5, L6 | PASS | Correctly marked NOT FOUND. |
| L2, L3 | PASS | Correctly distinguish later-decade facts (1950s camps; Urbana 76 small-group schedule) from an asserted 1941 rule. |
| L7 | PASS | All six numbers (18 staff/277 campuses 1946; 35 staff/499 chapters 1950; 46 opening campuses; 500+ chapters by early 1950s; 151 colleges at Toronto 1946; 1,300 students/154 campuses at Urbana 1948) trace to specific quote_ids. |
| L8 | PASS | Correctly cross-references R2/S5 instead of asserting a founding-era hardship claim not in the sources. |
| T1–T5 | PASS | Numbers and named examples (Howard, Boyd) trace to quote_ids; T2 correctly marked NOT FOUND given no charter was located. |
| M1–M5 | PASS | Each quotation is attributed to its actual speaker/source (Bakht Singh, MacLeod quoting Woods's biography, the 1947 IFES statement) without conflating one speaker's words with another's. |
| S1–S6 | PASS | S2/S4/S6 correctly marked NOT FOUND or not applicable given IVCF/USA's role as a convening/connecting body rather than a sending agency in its own right; S5 explicitly cross-references the unresolved Jim Elliot/Urbana-1948 discrepancy rather than silently accepting the secondary claim. |
| R1 (five named figures) | PASS | Howard, Boyd, Keller, Elliot, and Henn each carry specific quote_ids or in-text citations; Elliot's entry explicitly notes the discrepancy rather than asserting Urbana attendance as settled. |
| X1–X3 | PASS | Epigraph and Emphasis picks trace to real quote_ids; X3 correctly explains why "reprinted whole" applies loosely here (no founding document was found). |

No field states a fact its cited quotes do not support.

## 4. Secondary citations checked

| citation | page says what is claimed? | result |
|---|---|---|
| ivcf-ifes-history, ivcf-growing-love-1941, ivcf-urbana-decades, ivcf-stacey-woods, ivcf-studying-doctrinal-basis (InterVarsity.org) | Yes — verbatim, script-confirmed. | PASS |
| ivcf-statement-agreement (InterVarsity.org, treated as PRIMARY per F5/F8) | Yes — verbatim, script-confirmed for both the quotes.jsonl entries and the full charter_abridged.md extraction. | PASS |
| urbana-story (Urbana.org) | Yes — verbatim, script-confirmed. | PASS |
| wiki-ivcf, wiki-urbana (Wikipedia) | Yes — verbatim, script-confirmed; dossier and chapter correctly note wiki-urbana's own "needs updating" and "relies too closely on subject-affiliated sources" maintenance tags. | PASS |
| encyclopedia-ivcf (Encyclopedia.com / Gale, Larry Eskridge) | Yes — verbatim, script-confirmed; correctly used as an independent scholarly cross-check that produces the F2/F3/F1 discrepancies. | PASS |
| eef-journals-summer1948-pt1 (Elisabeth Elliot Foundation, quoting Jim Elliot's 1978-published journal) | Yes — verbatim; correctly flagged [RIGHTS: copyrighted, short quotation only] and logged in rights/permissions_log.csv. | PASS |

## 5. Dates and numbers vs known-facts sheet

| item | dossier value | roster value | result |
|---|---|---|---|
| IVCF USA founding | 1941, with three differing month/year readings (Nov 1941 / Sept 1941 / 1939-1940) all recorded | "IVCF-USA 1941" [HIGH] | MATCH on the year; the roster's [HIGH] label covers the year only, and the dossier does not silently drop the two conflicting month/earlier-year readings found independently, per Rule 9 |
| First convention | Toronto, 1946 | "first convention Toronto 1946" [HIGH] | MATCH |
| Urbana name from | 1948, University of Illinois at Urbana-Champaign | "Urbana from 1948" [HIGH] | MATCH |
| Founder | C. Stacey Woods | "C. Stacey Woods" | MATCH |
| Jim Elliot at Urbana 1948 | Reported as a secondary-source claim (Wikipedia, citing Sheppard 2010); the primary source fetched this session (Elliot's own 1948 journal) attests a related but distinct summer 1948 IVCF-affiliated gospel-team trip, not convention attendance specifically — recorded as unresolved | "Jim Elliot attended Urbana 1948 [VERIFY]" | Roster's own [VERIFY] flag is honored: this session neither confirms nor silently accepts the claim; both the secondary claim and the primary journal's own (different) content are recorded in discrepancies.md #4 |

## 6. Verdict

VERIFIED (zero FAILs). Coverage is limited — no founding-era constitution, doctrinal basis,
or convention program was located this session, and the Billy Graham Center Archives' IVCF
finding aid was BLOCKED by a Cloudflare gate. These gaps are disclosed, not papered over, in
`dossier.md`'s coverage note, `charter_abridged.md`'s coverage note, and `sources.csv`'s
BLOCKED/NOT FETCHED rows with a `.REQUEST.md` follow-up file. No dossier field asserts a fact
beyond what its cited quotations support, and the Jim Elliot/Urbana-1948 claim inherited from
the roster is reported as unresolved rather than silently confirmed.
