STATUS: VERIFIED
TICKET: V1-nyack-mti | V2-nyack-mti
ROLE: Verifier
INPUTS READ: research/nyack-mti/dossier.md; research/nyack-mti/quotes.jsonl; research/nyack-mti/charter_abridged.md; research/nyack-mti/discrepancies.md; research/nyack-mti/sources.csv; research/nyack-mti/text/thompson1920-lifeofsimpson.txt; research/nyack-mti/text/caw-1889-12-06.txt; chapters/IV-03-nyack-mti.md; plan/02_institution_roster.md (nyack-mti row)

# Verification report — nyack-mti

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/nyack-mti
38/38 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/nyack-mti --charter
4/4 passages verbatim; 0 failures
```

## 2. Quote-by-quote

All 38 quote records in `quotes.jsonl` were built directly from the fetched `text/` files using
a script that (a) extracts the exact printed-page/line range named in the record, (b) locates the
quoted text as a contiguous, whitespace-normalized word span inside that range, and (c) derives
the `before`/`after` context fields from the words immediately flanking that span in the same
source range — so `text`, `before`, and `after` are all guaranteed byte-for-byte derivable from
the named `document_id`. `check_quotes.py` independently re-confirms this for every record
against the full `text/` file (not just the narrow extraction range) and against the
before+text+after contiguity check; all 38 passed with no FAIL and no WARN (contiguity) lines.
Page markers were assigned by locating the nearest actual printed running-head page number
before and after each passage in the source OCR (Thompson 1920 carries legible alternating
running heads, e.g. "215", "216 LIFE OF A. B. SIMPSON"); a handful of quotes span two pages and
are marked "pp. N-M" accordingly. For the 1889 periodical (`caw-1889-12-06`), no legible running
page number was recoverable near the quoted items in this scan's OCR; those three records
(`nyack-mti-q036`-`q038`) say so explicitly in their `page` field rather than guessing a number.
Result: **38/38 PASS** (text matches, context matches, page-field honestly qualified where
uncertain).

## 3. Field-by-field (V1)

Every field in `dossier.md` was checked against its cited quote_id(s): the one-to-three-sentence
summary does not assert anything the quoted text does not support. Fields with no located
evidence are marked `NOT FOUND (searched: ...)` and no narrative claim is made from them in the
chapter (spot-checked: chapter's Admission and Common Life sections explicitly say when the
sources are silent, per `01_book_design.md` section 8, rather than omitting the gap silently).

| field group | result | note |
|---|---|---|
| F1-F9 (Founding) | PASS | F5 is qualified as "no single original document located"; F9 correctly limited to the two comparisons Thompson's text actually states (East London Institute; China Inland Mission). |
| A1-A6 (Admission) | PASS | A1, A2, A4, A6 correctly marked NOT FOUND; A3, A5 supported by quotes. |
| C1-C9 (Curriculum) | PASS | C5, C7 correctly marked NOT FOUND; C8 states "no degree...is described" rather than asserting one exists or does not exist beyond what the silence supports. |
| L1-L8 (Common life) | PASS | L2-L5 correctly marked NOT FOUND or partial; L7 fully supported by two independent dated figures (1883 and 1889). |
| T1-T5 (Teachers) | PASS | T2, T4 correctly marked NOT FOUND; T5 correctly states the ratio is "not directly computable" rather than computing one from mismatched figures. |
| M1-M5 (Emphasis) | PASS | M4 is labeled [SECONDARY] (Glover's narrative attribution of the "regions beyond" motto, not an independently tagged quote this session) and the dossier says so. |
| S1-S6 (Sending) | PASS | S6 correctly marked NOT FOUND (no instructions document located); the Boxer Rising casualty figure is explicitly flagged in `discrepancies.md` as not confirmed to be Institute-graduate-specific and is correspondingly kept OUT of the S5 chapter narrative (the chapter's Sending section does not repeat that figure). |
| R1-R4 (Fruit) | PASS | Four named individuals, each with a cited quote_id; dossier and chapter both state explicitly that no further named alumni were found this session. |

## 4. Secondary citations checked

Two dossier fields cite `[SECONDARY: ...]` without a quote_id (F7's narrative framing is PO via
quotes, not secondary; the actual [SECONDARY] citations are at F9... — on rereading, the dossier
uses [SECONDARY] tags at M4, M5, S2, and R4's byline attribution). Each was checked against the
named page region of `text/thompson1920-lifeofsimpson.txt` and found to say what is claimed
(Glover's and Turnbull's chapter bylines and narrative framing, and the 16-field list in Glover's
1920 chapter). Result: PASS for all four.

## 5. Dates and numbers vs known-facts sheet

Roster row (`02_institution_roster.md`, Part IV): "1882 (institute); 1887 (Alliance) [HIGH]."

| item | dossier value | roster value | result |
|---|---|---|---|
| Institute founding | 1882 (first informal class); formally organized October 1883 | 1882 | Consistent — roster's single year covers the informal 1882 start; the dossier adds the 1883 formalization date found in the source, which does not contradict 1882 and is recorded as an additional, more precise data point. |
| Alliance founding | 1887 (both the Christian Alliance and the Evangelical Missionary Alliance, Old Orchard convention) | 1887 | Consistent. |
| Founder | Albert B. Simpson | Albert Benjamin Simpson | Consistent (same person, roster gives full name). |
| Place | New York City (Institute); South Nyack from 1897 | New York; Nyack from 1897 | Consistent. |

No disagreement with the roster's known-facts sheet was found. All other discrepancies
identified this session are between the roster's list of "key primary documents" (which were not
locatable as independently digitized full texts) and what was actually recoverable; these are
recorded in `discrepancies.md` and are not disagreements about dates or numbers between two
fetched sources.

## 6. Check: no PRIMARY-ONLY field cites a secondary source

Confirmed by inspection: every field in `dossier.md` marked (PO) cites only quote_ids drawn from
`text/thompson1920-lifeofsimpson.txt` or `text/caw-1889-12-06.txt` (both fetched primary-treated
sources), never a `[SECONDARY: ...]` tag. `[SECONDARY: ...]` tags appear only on fields marked
(PS). PASS.

## 7. Check: nothing stated without a label

Confirmed by inspection: every dossier field carries either supporting quote_ids, a
`[SECONDARY: ...]` citation, `NOT FOUND (searched: ...)`, or `UNVERIFIED`. PASS.

## 8. Stage V2 — chapter verification

Every sentence in `chapters/IV-03-nyack-mti.md` that states a specific fact, date, name, or
number was checked against a dossier field or a footnoted quote_id. All 33 footnotes resolve to
either a `quote_id` present in `quotes.jsonl` (30 footnotes) or an explicit `[SECONDARY]` /
narrative citation matching a page region confirmed present in `text/thompson1920-lifeofsimpson.txt`
(3 footnotes: [^2], [^3], [^33]). No sentence in the chapter states a fact marked `UNVERIFIED` or
`NOT FOUND` in the dossier as though it were established; the Admission, Curriculum, Common Life,
and Sending sections each contain an explicit sentence naming what the sources do not establish,
matching the dossier's own NOT FOUND markers. The chapter's coverage note at the top discloses
the session's source limitations rather than concealing them. Result: PASS, no FAIL items.

## 9. Verdict

**VERIFIED** (zero FAILs across quote-record checks, charter-passage checks, field-by-field
checks, secondary-citation checks, date/number cross-checks, and chapter sentence-by-sentence
checks). Coverage gaps (NOT FOUND / UNVERIFIED fields, and undigitized primary documents named in
the roster but not located) are real and are documented in `dossier.md`, `discrepancies.md`, and
the chapter's coverage note — they are not verification failures, since nothing false or
unsupported was asserted.
