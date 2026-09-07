STATUS: VERIFIED
TICKET: V1-biola
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; sources.csv; text/*.txt

# Verification report — biola

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/biola
6/6 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/biola --charter
2/2 passages verbatim; 0 failures
```

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| biola-q001 | PASS | PASS | PASS (back-cover advertisement) | PASS |
| biola-q002 | PASS | PASS | PASS (back-cover advertisement, officer list) | PASS |
| biola-q003 | PASS | PASS | PASS (SECONDARY, labeled) | PASS |
| biola-q004 | PASS | PASS | PASS (SECONDARY, labeled) | PASS |
| biola-q005 | PASS | PASS | PASS (SECONDARY, labeled) | PASS |
| biola-q006 | PASS | PASS | PASS (SECONDARY, labeled) | PASS |

## 3. Field-by-field (V1)

Every dossier field carries a supporting quote_id, a `[SECONDARY: ...]` label, or a
`NOT FOUND (searched: ...)` marker; none is stated bare. Spot-checked fields:

- F5: PASS — states plainly that no 1908 Articles of Incorporation text was located, names the
  substitute used, and cross-references discrepancies.md item 2 (the confluence.biola.edu TLS
  failure) rather than silently omitting the gap.
- F8: PASS — correctly marked NOT FOUND for the doctrinal statement's actual text, while noting
  (via SECONDARY citation) that such a document is known to have existed.
- C1-C4, C7-C9: PASS — all correctly marked NOT FOUND rather than inferred from the Hunan
  satellite institute's curriculum, which is kept clearly distinguished (labeled as describing a
  different institution in China) rather than conflated with the Los Angeles campus.
- A1-A6: PASS — all six admission fields are honestly marked NOT FOUND; no invented age,
  examination, or testimony requirement appears anywhere in the dossier or chapter.
- Quotations sourced through the secondary `biola-hope-street-1985`: PASS — each is labeled
  SECONDARY in its dossier field and its page/citation field, following the same convention this
  book's moody-bible-institute dossier uses for biography-embedded founder quotations; no
  PRIMARY-ONLY field cites this secondary source for anything beyond a quoted utterance.

## 4. Secondary citations checked

`biola-hope-street-1985` (70 Years on Hope Street, 1985) is used for: the 1908 founding date and
officer list (F2, F3, T1), the 1908 stated-purpose quotation (F6), Lyman Stewart's 1913
cornerstone address (M1), Horton's own words on the founding need (F7, M1), the first student
body's size (L7), and the 1928 doctrinal-statement controversy (M3, M5). In each case the page
region was re-opened in `text/biola-hope-street-1985.txt` and the quoted or paraphrased material
confirmed present; result PASS in every instance. The source's heavy OCR corruption in captioned
and dual-column pages (noted in sources.csv) was worked around by using only clean, contiguous
passages, confirmed by the automated verbatim check for the six quoted passages.

## 5. Dates and numbers vs known-facts sheet

The known-facts sheet (`02_institution_roster.md` section 3) does not carry a dedicated entry for
`biola` beyond the roster table row itself (founded 1908, Lyman Stewart and T. C. Horton, `[HIGH]`).

| item | dossier value | roster value | result |
|---|---|---|---|
| Founding year | 1908 | 1908 [HIGH] | MATCH |
| Founders | Lyman Stewart, T. C. Horton | Lyman Stewart, T. C. Horton | MATCH |
| Place | Los Angeles | Los Angeles | MATCH |

## 6. Verdict

VERIFIED (zero FAILs). Coverage is thin relative to other Part IV chapters — six quote records
against, e.g., aim-sim's twenty-six — reflecting genuinely limited digitized primary material
located this session (see discrepancies.md items 2-4) rather than a shortcut in extraction.
