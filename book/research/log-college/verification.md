STATUS: VERIFIED
TICKET: V1-log-college
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; discrepancies.md; charter_abridged.md; text/whitefield-journal5-1739.txt; text/alexander1845.txt; sources.csv; book/plan/02_institution_roster.md (known-facts sheet has no log-college-specific entries beyond the roster row itself)

# Verification report — log-college

## 1. Script output

```
$ python3 book/plan/templates/check_quotes.py book/research/log-college
24/24 quote records passed; 0 failures
```

## 2. Quote-by-quote

All 24 quote records were re-opened at their named `text/` file and page marker and
compared against `quotes.jsonl`.

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| log-college-q001 | PASS | PASS | PASS (p.43-44, per note in text file) | PASS |
| log-college-q002 | PASS | PASS | PASS (p.44) | PASS |
| log-college-q003 | PASS | PASS | PASS (p.44) | PASS |
| log-college-q004 | PASS | PASS | PASS (p.44-45, spans page break) | PASS |
| log-college-q005 | PASS | PASS | PASS (p.45) | PASS |
| log-college-q006 | PASS | PASS | PASS (p.44) | PASS |
| log-college-q007 | PASS | PASS | PASS (p.8) | PASS |
| log-college-q008 | PASS | PASS | PASS (p.9) | PASS |
| log-college-q009 | PASS | PASS | PASS (p.10) | PASS |
| log-college-q010 | PASS | PASS | PASS (p.10-11) | PASS |
| log-college-q011 | PASS | PASS | PASS (p.18) | PASS |
| log-college-q012 | PASS | PASS | PASS (p.18-19) | PASS |
| log-college-q013 | PASS | PASS | PASS (p.19) | PASS |
| log-college-q014 | PASS | PASS | PASS (p.14) | PASS |
| log-college-q015 | PASS | PASS | PASS (p.22) | PASS |
| log-college-q016 | PASS | PASS | PASS (p.43) | PASS |
| log-college-q017 | PASS | PASS | PASS (p.44) | PASS |
| log-college-q018 | PASS | PASS | PASS (p.82-83) | PASS |
| log-college-q019 | PASS | PASS | PASS (p.84) | PASS |
| log-college-q020 | PASS | PASS | PASS (p.14) | PASS |
| log-college-q021 | PASS | PASS | PASS (p.44) | PASS |
| log-college-q022 | PASS | PASS | PASS (p.43) | PASS |
| log-college-q023 | PASS | PASS | PASS (p.43) | PASS |
| log-college-q024 | PASS | PASS | PASS (p.44) | PASS |

## 3. Field-by-field

| field | supported by | result | note |
|---|---|---|---|
| F1 | q003, q007, q010 | PASS | summary claims only the naming ("the College"/"the log college") that the quotes state |
| F2 | q001, q005, q012, q014, q020 | PASS | 1726 call to Neshaminy and 22 Nov 1739 visit both directly quoted; discrepancy on arrival-year (1716/1718) recorded, not resolved, per Rule 9 |
| F3 | q011, q016, q023 | PASS | founder + four sons + sole-instructor claim all directly quoted |
| F4 | q001, q008, q020 | PASS | location and building-siting both directly quoted |
| F5 | (none -- NOT FOUND) | PASS | correctly marked NOT FOUND with the ticket's own descriptions-in-lieu-of-a-charter note |
| F6 | q002, q003, q007, q010 | PASS | purpose statements attributed only to Whitefield/Alexander's own words, not invented |
| F7 | q009, q016, q022 | PASS | "need" framed only via Alexander's narrative, correctly labeled PS not PO |
| F8 | (none -- NOT FOUND) | PASS | no doctrinal-subscription text exists in either source; correctly NOT FOUND |
| F9 | q009, q018, q019 | PASS | antecedents correctly marked NOT FOUND for the founder's own citation; successors (College of NJ etc.) fully supported |
| A1, A3, A4 (partial), A5 | (none -- NOT FOUND) | PASS | correctly marked NOT FOUND |
| A2 | q009, q016, q022 | PASS | |
| A4 | q024 | PASS | correctly distinguishes an external post-graduation Synod exam from a Log College entrance exam |
| A6 | q016, q022 | PASS | |
| C1 | q016, q021 | PASS | "does not appear" language preserved, not overstated |
| C2 | q017 | PASS | |
| C3, C4, C5, C6, C9 | (none -- NOT FOUND) | PASS | correctly marked NOT FOUND; C4 explicitly flags that "classics" is not stretched into an asserted language list |
| C7 | q024 | PASS | |
| C8 | q017, q022 | PASS | |
| L1, L3, L4, L5, L6 | (none -- NOT FOUND) | PASS | correctly marked NOT FOUND |
| L2 | q006, q020 | PASS | summary explicitly notes what is NOT established (whether pupils boarded there) rather than assuming it |
| L7 | q004 | PASS | correctly distinguishes "sent out" count from an enrollment count |
| L8 | q015 | PASS | founder's death correctly distinguished from a student casualty |
| T1 | q011, q023 | PASS | |
| T2 | (none -- NOT FOUND) | PASS | |
| T3 | q006 | PASS | summary explicitly notes this is hospitality, not a described lecture/tutorial format |
| T4, T5 | (none -- NOT FOUND) | PASS | |
| M1 | q003, q010 | PASS | correctly labeled as outsiders' words, not Tennent's own |
| M2 | q002 | PASS | |
| M3 | (Synod-critics passage, same page as q024) | PASS | correctly attributed to critics, not the founder |
| M4, M5 | (none -- NOT FOUND) | PASS | |
| S1 | q002, q013, q024 | PASS | |
| S2 | q018, q019 | PASS | correctly distinguished as lineage rather than a stated sending destination |
| S3 | q004 | PASS | |
| S4, S5, S6 | (none -- NOT FOUND) | PASS | |
| R1-R4 | q011, q022, q004, q018 | PASS | four named/counted items, each cited |

## 4. Secondary citations checked

No dossier field in this session's log-college dossier cites a `[SECONDARY: ...]` source
(both `alexander1845` and `whitefield-journal5-1739` are treated as PRIMARY per the
roster's own "Key primary documents" cell for this slug, and the ticket note authorizing
this). Nothing in section 4 of the verification template applies; recorded here as
checked with zero secondary citations to verify.

## 5. Dates and numbers vs. known-facts sheet

`02_institution_roster.md`'s "Known-facts sheet" (section 3) has no `log-college`-specific
subsection (unlike Geneva, Judson, Hudson Taylor, Spurgeon, Lloyd-Jones, and the Elliots,
each of which gets one). The only relevant facts sheet is the roster row itself, checked
below.

| item | dossier value | roster row value | result |
|---|---|---|---|
| Founded | 1726 (call to Neshaminy; building erected same period) | "c.1726-27 [HIGH]" | consistent -- dossier's 1726 falls within the roster's stated range and is not a disagreement |
| Founder | William Tennent Sr. | "William Tennent Sr." | match |
| Place | Neshaminy, Pa. | "Neshaminy, Pa." | match |
| Type | Household academy for ministers | "Household academy for ministers" | match |

## 6. Verdict

VERIFIED (zero FAILs).
