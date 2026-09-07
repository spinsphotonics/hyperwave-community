STATUS: VERIFIED
TICKET: V1-trevecca
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/seymour1840-v1.txt, sources.csv, discrepancies.md, 02_institution_roster.md (trevecca row and known-facts sheet)

# Verification report — trevecca

## 1. Script output
```
7/7 quote records passed; 0 failures
```

## 2. Quote-by-quote
| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| trevecca-q001 | PASS | PASS | PASS (p. 309) | PASS |
| trevecca-q002 | PASS | PASS | PASS (p. 388-389) | PASS |
| trevecca-q003 | PASS | PASS | PASS (p. 85) | PASS |
| trevecca-q004 | PASS | PASS | PASS (p. 484) | PASS |
| trevecca-q005 | PASS | PASS | PASS (p. 305-306) | PASS |
| trevecca-q006 | PASS | PASS | PASS (p. 306) | PASS |
| trevecca-q007 | PASS | PASS | PASS (p. 308) | PASS |

## 3. Field-by-field
| field | supported by | result | note |
|---|---|---|---|
| F1 | q001 | PASS | |
| F2 | none quoted directly | PASS (correctly hedged as inferred from surrounding narrative, not independently quoted) | |
| F3 | q002 | PASS | correctly marks Fletcher's title UNVERIFIED per discrepancies.md item 1 |
| F4 | q003, q004 | PASS | |
| F5 | none | PASS (correctly NOT FOUND, with the substitution explained) | |
| F6 | q001 | PASS | |
| F7 | q001 | PASS | |
| F8 | none | PASS (correctly NOT FOUND) | |
| F9 | q004 | PASS | correctly distinguishes the settlement's founder from the College's founder |
| A1, A4-A5 | none | PASS (all correctly NOT FOUND) | |
| A2 | q005 | PASS | |
| A3 | located, not separately quoted | PASS (summary matches the read passage; flagged in discrepancies.md item 2 for a follow-up formal quote) | |
| A6 | none | PASS (correctly NOT FOUND, with reasoning given) | |
| C1 | q006 | PASS | correctly hedged as illustrative of one case, not a stated rule |
| C2, C4 | q005 | PASS | |
| C3, C5, C7, C9 | none | PASS (all correctly NOT FOUND) | |
| C6 | q007, located-not-quoted Hull passage | PASS | |
| C8 | context (Tyler's Cambridge/ordination path) | PASS | correctly distinguishes this from a Trevecca-granted credential |
| L1-L8 | none | PASS (all eight correctly NOT FOUND) | |
| T1 | q002 (context) | PASS | correctly hedged as UNVERIFIED for Fletcher's actual presence/role |
| T2-T5 | none | PASS (all correctly NOT FOUND) | |
| M1 | q001 | PASS | |
| M2 | q001, q007 | PASS | |
| M3, M5 | none | PASS (correctly NOT FOUND) | |
| M4 | q001 | PASS (correctly hedged as informal, not a stated motto) | |
| S1 | located, not separately quoted (Riddell/Hull request) | PASS (flagged in discrepancies.md item 2) | |
| S2 | q007 | PASS | |
| S3 | none as a total | PASS (correctly hedged, cites the source's own "names... irrecoverably lost" statement) | |
| S4-S6 | none | PASS (all correctly NOT FOUND) | |
| R1 | q005, q006 | PASS | |
| R2, R3 | located, not separately quoted (Oxford expulsion passage) | PASS (flagged in discrepancies.md item 2 as follow-up) | |

## 4. Secondary citations checked
| citation | page says what is claimed? | result |
|---|---|---|
| Seymour's table-of-contents heading "Mr. Fletcher appointed Vicar of Madely" | Yes — matches discrepancies.md item 1 | PASS |

## 5. Dates and numbers vs known-facts sheet
| item | dossier value | facts-sheet value | result |
|---|---|---|---|
| Founding date | not independently confirmed this session; inferred window 1767-1768 from surrounding narrative | roster: "1768 [HIGH]" | PASS (consistent; dossier does not contradict, and does not claim more precision than the source supports) |
| Fletcher's role | UNVERIFIED (title not found in material read this session) | roster: "Fletcher of Madeley as president [HIGH]" | Flagged, not a FAIL — the roster's confidence label is not itself a citable source (per 02_institution_roster.md's own instruction: "cite these from a source in the dossier; do not cite this sheet"), and no primary confirmation was located this session. See discrepancies.md item 1. |

## 6. Verdict
VERIFIED (zero FAILs). Coverage is markedly thinner than Yale's and thinner than
Northampton's: no admission rule, curriculum syllabus, daily-life detail (all of L1-L8), or
teacher roster was located in the one volume fetched this session, and Fletcher's specific role
at the College — the roster's own headline fact — could not be confirmed from a primary
quotation this session. This is reported honestly rather than papered over. No FAIL requires
correction before the chapter is written; the Writer must not state Fletcher's title as
"president" without a citation, and must not state anything from the NOT FOUND fields.
