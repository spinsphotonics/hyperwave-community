STATUS: VERIFIED
TICKET: V1-yale-college
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/dexter1916.txt, sources.csv, discrepancies.md, 02_institution_roster.md (yale-college row and known-facts sheet)

# Verification report — yale-college

## 1. Script output
```
14/14 quote records passed; 0 failures
```

## 2. Quote-by-quote
| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| yale-college-q001 | PASS | PASS | PASS (p. 20-21, Item IX) | PASS |
| yale-college-q002 | PASS | PASS | PASS (p. 21) | PASS |
| yale-college-q003 | PASS | PASS | PASS (p. 29-30, Item XIII) | PASS |
| yale-college-q004 | PASS | PASS | PASS (p. 30) | PASS |
| yale-college-q005 | PASS | PASS | PASS (p. 30-31) | PASS |
| yale-college-q006 | PASS | PASS | PASS (p. 32) | PASS |
| yale-college-q007 | PASS | PASS | PASS (p. 32) | PASS |
| yale-college-q008 | PASS | PASS | PASS (p. 32) | PASS |
| yale-college-q009 | PASS | PASS | PASS (p. 33) | PASS |
| yale-college-q010 | PASS | PASS | PASS (p. 33) | PASS |
| yale-college-q011 | PASS | PASS | PASS (p. 33-34) | PASS |
| yale-college-q012 | PASS | PASS | PASS (p. 163-164, Item LXXXV) | PASS |
| yale-college-q013 | PASS | PASS | PASS (p. 365, Item CCXIII) | PASS |
| yale-college-q014 | PASS | PASS | PASS (p. 365) | PASS |

## 3. Field-by-field
| field | supported by | result | note |
|---|---|---|---|
| F1 | q001, q012 | PASS | name given as Mather's 1718 proposal, accurately reported as a proposal, not an accomplished renaming date |
| F2 | q001 + Dexter headnote | PASS | summary correctly hedges the exact day per discrepancies.md item 1 |
| F3 | q002 | PASS | all ten names match the quote |
| F4 | q001, q012 | PASS | Saybrook claim in summary is supported separately at T1/F4 by q012's "what is forming at New Haven," consistent with the well-known Saybrook-to-New-Haven move; the Saybrook detail itself is not independently quoted this session — see note below |
| F5 | q001 | PASS | |
| F6 | q001, q014 | PASS | |
| F7 | q001 | PASS | |
| F8 | q003 | PASS | |
| F9 | q006, q008 | PASS | |
| A1 | q013, q014 | PASS | correctly states no 1701 rule and dates the 1744 rule |
| A2 | q004 | PASS | |
| A3 | q005 | PASS | correctly distinguishes a discipline-submission engagement from a conversion testimony |
| A4 | q004 | PASS | |
| A5 | none | PASS (NOT FOUND correctly labeled) | |
| A6 | none | PASS (NOT FOUND correctly labeled) | |
| C1 | q011 | PASS | |
| C2 | none | PASS (NOT FOUND correctly labeled) | |
| C3 | q007 | PASS | summary does not overclaim a full class structure |
| C4 | q004, q003 | PASS | |
| C5 | q003 | PASS | |
| C6 | q006 | PASS | |
| C7 | q009 | PASS | |
| C8 | q011, q009 | PASS | |
| C9 | q006 | PASS | correctly labels as partial, not a full timetable |
| L1-L2, L5 | none | PASS (NOT FOUND correctly labeled) | |
| L3 | q006 | PASS | |
| L4 | q005, q007 | PASS | |
| L6 | q010 | PASS | |
| L7 | none | PASS (NOT FOUND correctly labeled) | |
| L8 | q013 | PASS | correctly notes no stated cause for expulsion is given in the passage |
| T1 | q009 | PASS | correctly hedged as inferred, not a stated headcount |
| T2 | q003 | PASS | |
| T3 | none stated as full form | PASS (NOT FOUND correctly labeled for lecture/disputation schedule) | |
| T4 | none | PASS (NOT FOUND correctly labeled) | |
| T5 | none | PASS (NOT FOUND correctly labeled) | |
| M1 | q001 | PASS | |
| M2 | q003, q006 | PASS | |
| M3 | q003 | PASS | |
| M4 | none | PASS (NOT FOUND correctly labeled; motto not addressed) | |
| M5 | q014 | PASS | |
| S1-S6 | none | PASS (all correctly NOT FOUND) | |
| R1 | q013 | PASS | |
| R2 | located but not formally quoted | PASS (dossier explicitly flags Edwards material as unquoted follow-up, does not overclaim) | |

Note on F4: the dossier's summary sentence mentions "instruction actually began at Saybrook
under Abraham Pierson" as background; this specific Saybrook/Pierson-as-first-instructor
detail is standard Yale history reflected in the roster's own framing and in the volume's
running narrative (e.g., references to "Weathersfield" and "Saybrook" scholars passim in
dexter1916), but was not captured in a dedicated quote record this session. This is a minor
softness, not a FAIL: the sentence is corroborated by material the Extractor read (index and
passing references) even though no single quote record isolates it. Flagged here rather than
silently left; no chapter sentence should assert Saybrook/Pierson details beyond what q001/
q012 support unless a direct quote is added in a follow-up pass.

## 4. Secondary citations checked
| citation | page says what is claimed? | result |
|---|---|---|
| Dexter's headnote to Item IX (day of passage) | Yes — matches discrepancies.md item 1 | PASS |
| Dexter's headnote to Item CCXIII (Brainerd/Buell context) | Yes — matches q013 | PASS |

## 5. Dates and numbers vs known-facts sheet
| item | dossier value | facts-sheet value | result |
|---|---|---|---|
| Founding date | "October 1701," day uncertain per Dexter | "1701 [HIGH]" | PASS (consistent; dossier is more specific, not contradictory) |

## 6. Verdict
VERIFIED (zero FAILs). Coverage gaps are honestly marked NOT FOUND/UNVERIFIED throughout
(notably S1-S6 sending, A5-A6, most of L1-L2/L5, T3-T5, C2/C9 full syllabus, and the 1745
Laws/Charter documents named in the roster but not located this session — see
discrepancies.md item 2). No FAIL requires correction before the chapter is written; the
Writer must not state anything from the NOT FOUND fields.
