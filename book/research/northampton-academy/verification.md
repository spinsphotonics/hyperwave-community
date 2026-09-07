STATUS: VERIFIED
TICKET: V1-northampton-academy
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/boyd1860.txt, sources.csv, discrepancies.md, 02_institution_roster.md (northampton-academy row and known-facts sheet)

# Verification report — northampton-academy

## 1. Script output
```
9/9 quote records passed; 0 failures
```

## 2. Quote-by-quote
| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| northampton-academy-q001 | PASS | PASS | PASS (p. 269-270) | PASS |
| northampton-academy-q002 | PASS | PASS | PASS (p. 271) | PASS |
| northampton-academy-q003 | PASS | PASS | PASS (p. 281) | PASS |
| northampton-academy-q004 | PASS | PASS | PASS (p. 281) | PASS |
| northampton-academy-q005 | PASS | PASS | PASS (p. 281-282) | PASS |
| northampton-academy-q006 | PASS | PASS | PASS (p. 282) | PASS |
| northampton-academy-q007 | PASS | PASS | PASS (p. 282-283) | PASS |
| northampton-academy-q008 | PASS | PASS | PASS (p. 284) | PASS |
| northampton-academy-q009 | PASS | PASS | PASS (p. 286) | PASS |

## 3. Field-by-field
| field | supported by | result | note |
|---|---|---|---|
| F1 | q001, q002 | PASS | |
| F2 | q001 | PASS | move to Northampton correctly stated as inferred from dateline, not an independently quoted fact |
| F3 | q007 | PASS | correctly notes assistant is unnamed |
| F4 | q001, q002 | PASS | |
| F5 | q002 | PASS | correctly explains why no charter exists and names the letter as the substitute document used |
| F6 | q009 | PASS | correctly labeled thin/NOT FOUND as a full purpose statement |
| F7 | q001, q002 | PASS | |
| F8 | none quoted | PASS (NOT FOUND correctly labeled) | |
| F9 | none quoted (Jennings passage located but flagged, not misattributed) | PASS | correctly distinguishes Jennings's academy from Doddridge's own per discrepancies.md item 2 |
| A1-A5 | none | PASS (all correctly NOT FOUND) | |
| A6 | q009 | PASS | |
| C1 | none directly, inferred | PASS (correctly hedged as not a stated total) | |
| C2 | q008 | PASS | |
| C3 | q007 | PASS | |
| C4 | q005 | PASS | |
| C5 | q008 | PASS | |
| C6 | located, not separately quoted (public exercises paragraph) | PASS (summary matches the read passage; flagged that no discrete quote_id was made for this specific paragraph — minor softness, not a FAIL, since the paragraph was read and the summary does not overclaim beyond it) | |
| C7 | located, not separately quoted (weekly examination paragraph) | PASS (same note as C6) | |
| C8 | none | PASS (NOT FOUND correctly labeled) | |
| C9 | q007 | PASS | |
| L1 | q004, q005 | PASS | |
| L2 | q003, q006 | PASS | |
| L3 | q004, q005 | PASS | |
| L4 | q006, q003 | PASS | correctly distinguishes Boyd's own characterization from a quoted rule |
| L5, L7-L8 | none | PASS (NOT FOUND correctly labeled) | |
| L6 | context of q002, q008 | PASS | correctly hedged, not a formal fee schedule |
| T1 | q007 | PASS | |
| T2 | none | PASS (NOT FOUND correctly labeled) | |
| T3 | q003, q007 | PASS | |
| T4 | located, not separately quoted | PASS (summary does not overclaim; no named student example was actually available) | |
| T5 | q007 | PASS (correctly hedged as qualitative, not numeric) | |
| M1, M4-M5 | none | PASS (all correctly NOT FOUND) | |
| M2 | q008, q005 | PASS | |
| M3 | q002 | PASS | |
| S1-S6 | none | PASS (all correctly NOT FOUND) | |
| R1 | q001, q008 | PASS | |
| R2 | q003 | PASS | |

Note on C6/C7/T4: three dossier fields draw on paragraphs of text/boyd1860.txt (the "public
exercises" paragraph and the "examined them upon the contents" paragraph, both read in full
this session as shown in the transcript above) without a dedicated quote_id isolating them.
This is a minor completeness gap, not a FAIL — the summaries do not state anything beyond what
those paragraphs say, and the paragraphs themselves are part of the same continuously-read
Orton-derived passage already covered by quote records q003, q007, and q008 on either side of
them. Flagged for a follow-up Extractor pass to add discrete quote records for these two
paragraphs before any future expansion of the Curriculum section.

## 4. Secondary citations checked
| citation | page says what is claimed? | result |
|---|---|---|
| Boyd's introduction to the Orton-derived "method of education" account | Yes — matches discrepancies.md item 1 | PASS |

## 5. Dates and numbers vs known-facts sheet
| item | dossier value | facts-sheet value | result |
|---|---|---|---|
| Founding date | 1729, Harborough | roster: "1729 [HIGH]" | PASS |

## 6. Verdict
VERIFIED (zero FAILs). Coverage is thin relative to Yale, honestly: no admission rule (A1-A5),
doctrinal-subscription clause (F8), sending mechanism (S1-S6), fee schedule (L6), or named
alumni beyond Doddridge and Orton were located in the one volume fetched this session. Three
further primary/near-primary sources named in sources.csv (the 1776 *Course of Lectures*, the
1807 *Lectures on Preaching*, and Humphreys's 1829 multi-volume *Correspondence and Diary*)
were located but not fetched this session for lack of time, and are flagged there as follow-up
work rather than silently omitted. No FAIL requires correction before the chapter is written;
the Writer must not state anything from the NOT FOUND fields.
