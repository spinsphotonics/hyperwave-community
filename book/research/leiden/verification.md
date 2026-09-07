STATUS: VERIFIED
TICKET: V1-leiden
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; sources.csv; discrepancies.md; all text/*.txt

# Verification report — leiden

## 1. Script output
```
$ python3 templates/check_quotes.py research/leiden
28/28 quote records passed; 0 failures

$ python3 templates/check_quotes.py research/leiden --charter
1/1 passages verbatim; 0 failures
```

## 2. Quote-by-quote
All 28 quote_ids (leiden-q001 through leiden-q028) were generated programmatically by locating
the exact substring in the corresponding text/*.txt file and computing the 20-word before/after
context from that same file, then re-checked by check_quotes.py. Result: PASS for all 28 on
text match, context match, and page marker (each page marker reads "img. 1 (single web page, no
pagination)" and is correct — every source this session is a single unpaginated web page).

| quote_id range | text matches | context matches | page correct | result |
|---|---|---|---|---|
| leiden-q001–q028 | yes (28/28) | yes (28/28) | yes (28/28) | PASS |

## 3. Field-by-field
- F1: PASS — summary states two names (University; Statencollege) plus the Seminarium Indicum, all directly supported by leiden-q024/q027.
- F2: PASS — dates supported by leiden-q001, leiden-q018, leiden-q024, leiden-q027.
- F3: PASS — supported by leiden-q002, q003, q026, q027; note Waleus/Walaeus spelling flagged in discrepancies.md rather than silently resolved.
- F4: PASS as SECONDARY-labeled, unquoted background (Statencollege location, Seminarium Indicum location) — correctly marked as not independently quoted this session rather than presented as a formal quote-backed fact.
- F5: PASS — correctly states the charter/statutes text itself was not obtained (BLOCKED), does not overclaim.
- F6: PASS — supported by leiden-q006, q008, q009.
- F7: PASS — supported by leiden-q006, q007, q014, q015, q025.
- F8, F9: PASS — correctly marked NOT FOUND rather than invented.
- A1–A6: PASS — mostly NOT FOUND, correctly marked; A1/A6 correctly note the Seminarium Indicum restriction is a training-time rule, not an admission requirement, avoiding overstatement.
- C1–C9: PASS — all claims trace to leiden-q010–q021 or explicitly marked NOT FOUND/SECONDARY-unquoted.
- L1–L8: PASS — L2, L4, L5, L7 supported by quotes; L1, L3, L8 correctly NOT FOUND; L6 correctly left unquoted/NOT FOUND for a formal quote_id rather than stated as fact.
- T1–T5: PASS — supported by leiden-q012, q018, q019, q027, q028; T2 correctly NOT FOUND.
- M1–M5: PASS — supported by leiden-q022, q023, q016, q019; M5 correctly NOT FOUND.
- S1–S6: PASS — supported by leiden-q014, q015, q020, q028; S4, S5 correctly NOT FOUND.
- R1: PASS — supported by leiden-q018, q022, q027. R2 correctly flagged as SECONDARY-unquoted follow-up rather than stated as a formal fact. R3–R4 correctly NOT FOUND.
- No field claims more than its cited quotes support.

## 4. Secondary citations checked
| citation | page says what is claimed? | result |
|---|---|---|
| statencollege-nlwiki (leiden-q024, q025, q026) | yes — text matches verbatim per script | PASS |
| indisch-seminarium-nlwiki (leiden-q027, q028) | yes — text matches verbatim per script | PASS |
| walaeus-christianstudylibrary (leiden-q014–q023) | yes — text matches verbatim per script | PASS |
| leiden-foundation-documents (leiden-q001–q005) | yes | PASS |
| leiden-founding-royal-family (leiden-q006–q009) | yes | PASS |
| leiden-1592-curriculum-blog (leiden-q010–q013) | yes | PASS |

## 5. Dates and numbers vs known-facts sheet
Roster row (02_institution_roster.md, Part I table): "1575; Staten College 1592 [HIGH]." Both
dates confirmed: University charter dated 6 Jan 1575 / inaugurated 8 Feb 1575 (leiden-q001);
Statencollege founded 1592 (leiden-q024). The roster's description "the Dutch Reformed seminary
that trained ministers for the Indies" does not match a single institution in the sources read
this session; see discrepancies.md item 1. This is recorded as a discrepancy, not silently
corrected.

## 6. Verdict
VERIFIED (zero FAILs). Coverage is thin: no primary charter/statute/Ordinances text was
obtained (BLOCKED, see sources.csv and discrepancies.md item 5); roughly half of dossier fields
(A1, A3–A5, C1, C8, F8, F9, L1, L3, L6, L8, M5, S4, S5, R2–R4) are NOT FOUND or flagged
SECONDARY-unquoted rather than fully sourced. This is disclosed in the dossier's sourcing note
and is expected to produce a chapter shorter than the 3,500–5,000-word Tier A budget, per the
Tier B allowance in the task brief.
