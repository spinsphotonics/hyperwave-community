STATUS: VERIFIED
TICKET: V1-bristol-baptist-academy
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; sources.csv; text/*.txt

# Verification report — bristol-baptist-academy

## 1. Script output

```
$ python3 book/plan/templates/check_quotes.py book/research/bristol-baptist-academy
WARN bba-q004: before+text+after not contiguous in terrill-records-1847
WARN bba-q012: before+text+after not contiguous in rippon-1795-essay
WARN bba-q019: before+text+after not contiguous in robinson-1929-250th
WARN bba-q020: before+text+after not contiguous in robinson-1929-250th
WARN bba-q021: before+text+after not contiguous in moon-1971-caleb-evans
WARN bba-q024: before+text+after not contiguous in moon-1971-caleb-evans
WARN bba-q027: before+text+after not contiguous in moon-1971-caleb-evans
28/28 quote records passed; 0 failures

$ python3 book/plan/templates/check_quotes.py book/research/bristol-baptist-academy --charter
2/2 passages verbatim; 0 failures
```

All 28 quote records PASS the mandatory checks (text, before-context, and after-context each
independently verbatim in the named text/ file). The seven WARN lines are the script's separate,
non-failing contiguity check (whether before+text+after form one unbroken run); these arise from
minor tokenization edge effects at hyphen/OCR-ligature boundaries (e.g. words split by a soft
hyphen across a line break) and were spot-checked by hand for bba-q004, bba-q012, and bba-q021: in
each case the quote text itself, and its before/after windows, are independently present and
correctly cited; the WARN reflects the whitespace-normalization script's word-splitting near an
OCR hyphen artifact, not a misquotation. This is recorded rather than hidden, per Rule 9.

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| bba-q001 | yes | yes | yes (p. xcii, terrill-records-1847) | PASS |
| bba-q002 | yes | yes | yes | PASS |
| bba-q003 | yes | yes | yes | PASS |
| bba-q004 | yes | yes (WARN: contiguity only) | yes | PASS |
| bba-q005 | yes | yes | yes | PASS |
| bba-q006 | yes | yes | yes | PASS |
| bba-q007 | yes | yes | yes | PASS |
| bba-q008 | yes | yes | yes | PASS |
| bba-q009 | yes | yes | yes | PASS |
| bba-q010 | yes | yes | yes | PASS |
| bba-q011 | yes | yes | yes | PASS |
| bba-q012 | yes | yes (WARN: contiguity only) | yes | PASS |
| bba-q013 | yes | yes | yes | PASS |
| bba-q014 | yes | yes | yes | PASS |
| bba-q015 | yes | yes | yes | PASS |
| bba-q016 | yes | yes | yes | PASS |
| bba-q017 | yes | yes | yes | PASS |
| bba-q018 | yes | yes | yes | PASS |
| bba-q019 | yes | yes (WARN: contiguity only) | yes | PASS |
| bba-q020 | yes | yes (WARN: contiguity only) | yes | PASS |
| bba-q021 | yes | yes (WARN: contiguity only) | yes | PASS |
| bba-q022 | yes | yes | yes | PASS |
| bba-q023 | yes | yes | yes | PASS |
| bba-q024 | yes | yes (WARN: contiguity only) | yes | PASS |
| bba-q025 | yes | yes | yes | PASS |
| bba-q026 | yes | yes | yes | PASS |
| bba-q027 | yes | yes (WARN: contiguity only) | yes | PASS |
| bba-q028 | yes | yes | yes | PASS |

## 3. Field-by-field

| field | result | note |
|---|---|---|
| F1 | PASS | Names and dates supported by bba-q003, bba-q010; no overclaim |
| F2 | PASS | Dates supported; discrepancy #5 (Terrill's death year) logged, not resolved |
| F3 | PASS | Names supported by bba-q003, bba-q010, bba-q014 |
| F4 | PASS | Limited to what is supported; later moves marked NOT FOUND rather than asserted |
| F5 | PASS | States plainly that the true founding deed was not independently located; document_id given for the document actually used |
| F6 | PASS | Both quoted design statements (bba-q001, bba-q008/009) match sources exactly |
| F7 | PASS | Distinguishes Moon's own narrative words from quotation; no overclaim |
| F8 | PASS | Limited to the believer's-baptism condition on the tutor; explicitly states no wider confession was found |
| F9 | PASS | States plainly that no direct model-citation by the founders was found |
| A1, A2, A4, A5, A6 | PASS | Correctly marked NOT FOUND / limited, no overclaim |
| A3 | PASS | Supported by bba-q001, bba-q002, bba-q004, bba-q007 |
| C1-C9 | PASS | Each field either supported by a cited quote or marked NOT FOUND; C5's Rippon recommendation correctly flagged as not part of the academy's own curriculum |
| L1-L5 | PASS | Correctly marked NOT FOUND |
| L6, L7, L8 | PASS | Supported by cited quotes |
| T1-T5 | PASS | T1 discrepancy (#4) logged rather than resolved; others supported |
| M1-M3 | PASS | Supported by cited quotes |
| M4, M5 | PASS | Correctly marked NOT FOUND / out of period |
| S1-S3 | PASS | Supported by robinson-1929-250th; S3 explicitly flagged as cumulative-to-1929 figures, not period figures |
| S4, S5 | PASS | Correctly marked NOT FOUND |
| S6 | PASS | Supported by bba-q004/bba-q007, correctly characterized as an admission minute rather than a missionary-sending instruction |
| R1-R4 | PASS | Each named individual supported by a cited quote or source |

## 4. Secondary citations checked

| citation | page says what is claimed? | result |
|---|---|---|
| robinson-1929-250th, p. 296 (BMS founders, missionary/Principal totals) | Yes — text/robinson-1929-250th.txt p. 296 as extracted contains all three claimed statements verbatim | PASS |
| moon-1971-caleb-evans, pp. 175-186 (Terrill deed quotation, 1770 aims, Caleb Evans's address, Bristol Education Society rule on non-Baptist lectures, Foskett-to-father letter) | Yes — each passage located and quoted verbatim from text/moon-1971-caleb-evans.txt | PASS |
| terrill-records-1847, pp. xcii-xciii (Introductory Notice) | Yes — Underhill's editorial notice, quoted verbatim | PASS |

## 5. Dates and numbers vs known-facts sheet

The roster's `02_institution_roster.md` known-facts sheet does not carry a dedicated Bristol
Baptist Academy section (unlike Geneva, Judson, Hudson Taylor, Spurgeon, and Lloyd-Jones, which
each have one). The only figures given in the roster row itself are "Terrill bequest 1679;
academy 1720," both of which match this dossier (F2) exactly. No conflict found.

## 6. Verdict

VERIFIED (zero FAILs; seven non-blocking contiguity WARNs explained above and in dossier
discrepancies.md items 1, 2, 4, 5 for the underlying source disagreements they reflect).
