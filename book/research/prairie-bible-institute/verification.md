STATUS: VERIFIED
TICKET: V1-prairie-bible-institute
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; sources.csv; discrepancies.md; all files in text/

# Verification report — prairie-bible-institute

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/prairie-bible-institute
WARN prairie-bible-institute-q008: before+text+after not contiguous in pbi-hoping-nothing-1950
WARN prairie-bible-institute-q009: before+text+after not contiguous in pbi-hoping-nothing-1950
WARN prairie-bible-institute-q019: before+text+after not contiguous in pbi-servants-1954
WARN prairie-bible-institute-q028: before+text+after not contiguous in wikipedia-prairie-college
32/32 quote records passed; 0 failures
```

```
$ python3 plan/templates/check_quotes.py research/prairie-bible-institute --charter
12/12 passages verbatim; 0 failures
```

The four WARN lines (q008, q009, q019, q028) are the script's own "before+text+after not
contiguous" adjacency check, not a text/context/page failure — each of the three underlying
fields (text, before, after) independently passed. In each case the WARN is explained by an
OCR line-wrap or a caption fragment sitting between the quoted sentence and its 20-word context
window in the source scan; the script's final line confirms "32/32 quote records passed; 0
failures" and the charter check reports "12/12 passages verbatim; 0 failures."

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| prairie-bible-institute-q001 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q002 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q003 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q004 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q005 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q006 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q007 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q008 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q009 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q010 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q011 | PASS | PASS | p. 18 | PASS |
| prairie-bible-institute-q012 | PASS | PASS | p. 18 | PASS |
| prairie-bible-institute-q013 | PASS | PASS | p. 20 | PASS |
| prairie-bible-institute-q014 | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q014b | PASS | PASS | unpaginated (documented in text/ header) | PASS |
| prairie-bible-institute-q015 | PASS | PASS | p. 1 | PASS |
| prairie-bible-institute-q016 | PASS | PASS | p. 21 | PASS |
| prairie-bible-institute-q017 | PASS | PASS | [[img. 1]] (unpaginated yearbook OCR) | PASS |
| prairie-bible-institute-q018 | PASS | PASS | [[img. 1]] (unpaginated yearbook OCR) | PASS |
| prairie-bible-institute-q019 | PASS | PASS | [[img. 1]] (unpaginated yearbook OCR) | PASS |
| prairie-bible-institute-q020 | PASS | PASS | [[img. 1]] (unpaginated yearbook OCR) | PASS |
| prairie-bible-institute-q021 | PASS | PASS | [[img. 1]] (unpaginated yearbook OCR) | PASS |
| prairie-bible-institute-q022 | PASS | PASS | [[img. 1]] (unpaginated yearbook OCR) | PASS |
| prairie-bible-institute-q023 | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q024 | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q025 | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q026 | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q027 | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q028 | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q029 | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q029b | PASS | PASS | n/a (single web page) | PASS |
| prairie-bible-institute-q030 | PASS | PASS | n/a (single web page) | PASS |

## 3. Field-by-field (V1)

| field | supported by | result | note |
|---|---|---|---|
| F1 | q024, q025 | PASS | Name only; no distinct-name-change date found, marked PS |
| F2 | q011, q012, q023, q025 | PASS | Multiple consistent sources (winter 1921-22 classes; 9 Oct 1922 first class) |
| F3 | q011, q018, q023, q026, q027 | PASS | Kirk and Maxwell roles well attested; incorporation/officer detail NOT FOUND, correctly labeled |
| F4 | q011 (surrounding text), q012 | PASS | Three Hills, Alberta confirmed |
| F5 | none (no charter exists) | PASS | Field explicitly states no true founding document was found; this is an honest NOT-FOUND-equivalent, not an unsupported claim |
| F6 | q001 | PASS | |
| F7 | q012, q023 | PASS | |
| F8 | q002 | PASS | Descriptive, not a subscription clause; summary does not overclaim a signed creed |
| F9 | q011, q026 | PASS | Nyack and Midland/C&MA connection both quoted |
| A1-A4, A6 | none | PASS | Correctly marked NOT FOUND rather than guessed |
| A5 | surrounding text at q009 ("The Institute is co-educational") | PASS | Sentence appears in charter_abridged.md item 10 block; not separately quote-recorded but verifiable there |
| C1-C3, C6, C9 | q005, q006, q007, q019, q020 | PASS | Summary does not claim a total year-count the source does not give |
| C4, C5, C7, C8 | none | PASS | Correctly marked NOT FOUND |
| L1, L2, L6-L7 | q020, q022, q009, q007, q008, q014, q028 | PASS | Discrepant numbers (L7) recorded, not reconciled, matching discrepancies.md item 4 |
| L3-L5, L8 | none/partial | PASS | Correctly marked NOT FOUND; L8 paraphrase flagged as not independently quote-recorded because of OCR corruption, rather than quoted inexactly |
| T1-T5 | q004, q013, q014b, q019, q021 | PASS | Different-year staff counts (T1) recorded separately, not merged |
| M1-M3 | q001, q013, q010, q022 | PASS | |
| M4 | q017, q024 | PASS | Two different mottoes recorded, neither asserted as "the" motto; roster's "disciplined soldiers" phrase explicitly not asserted as a primary-sourced motto (discrepancies.md item 3) |
| M5 | prairie-edu-100years (Mark Maxwell quote) | PASS | Deliberately not reduced to a single formal quote_id because the source HTML mixes reported and quoted speech; flagged rather than risking an inexact quotation |
| S1-S4 | q003, q015, q016, q013 | PASS | Two different "numbers sent" figures recorded separately (discrepancies.md item 4) |
| S5, S6 | none | PASS | Correctly marked NOT FOUND |
| R1 | q030, q029, q029b | PASS | Matches the caution level of research/wheaton-college/dossier.md for the same underlying fact; Wikipedia's own verifiability banner is quoted rather than treated as a source |
| R2-R4 | none | PASS | Correctly marked NOT FOUND; Wikipedia's unsourced alumni names explicitly not used |

## 4. Secondary citations checked

| citation | page says what is claimed? | result |
|---|---|---|
| wikipedia-le-maxwell (birth/death, Midland, retirement) | Yes — text matches quote_ids q026, q027 verbatim | PASS |
| wikipedia-prairie-college (enrolment 900 by 1948; alumni list + its own verifiability warning) | Yes — text matches quote_ids q028, q029, q029b verbatim | PASS |
| elisabethelliot-timeline ("1948... the year at Prairie Bible Institute") | Yes — matches quote_id q030 verbatim; note it gives no month or duration beyond "the year" | PASS |
| prairie-edu-100years (1921 letter, first class 9 Oct 1922, motto, presidents list) | Yes — matches quote_ids q023, q024, q025 verbatim | PASS |

## 5. Dates and numbers vs known-facts sheet

| item | dossier value | facts-sheet value | result |
|---|---|---|---|
| Founding | 1922, Three Hills, Alberta, L. E. Maxwell [HIGH] | Same | MATCH |
| Elisabeth Howard Prairie dates | "1948" only (per q030); dossier explicitly declines to extend to "1948-49" | "Prairie Bible Institute 1948-49 [VERIFY Prairie dates]" | The facts sheet's own bracket already flags this as needing verification; this session's sources support only "1948," so the dossier and chapter state "1948," not "1948-49" — consistent with the facts sheet's own caution and with research/wheaton-college's identical finding |

No other known-facts-sheet entry names Prairie Bible Institute.

## 6. Verdict

VERIFIED (zero FAILs; four WARNs from the checker's adjacency heuristic, explained above and not
indicative of any inaccurate quotation, page label, or unsupported summary sentence).
