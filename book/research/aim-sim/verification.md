STATUS: VERIFIED
TICKET: V1-aim-sim
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; sources.csv; text/*.txt

# Verification report — aim-sim

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/aim-sim
WARN aim-sim-q005: before+text+after not contiguous in aim-leaflets
WARN aim-sim-q007: before+text+after not contiguous in aim-leaflets
WARN aim-sim-q013: before+text+after not contiguous in sim-principles-practice
WARN aim-sim-q015: before+text+after not contiguous in sim-principles-practice
WARN aim-sim-q018: before+text+after not contiguous in sim-principles-practice
WARN aim-sim-q019: before+text+after not contiguous in sim-principles-practice
WARN aim-sim-q023: before+text+after not contiguous in sim-burden-of-sudan
WARN aim-sim-q024b: before+text+after not contiguous in sim-burden-of-sudan
26/26 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/aim-sim --charter
10/10 passages verbatim; 0 failures
```

The eight WARN lines are the script's contiguity check on `before+text+after`; they arise because
each `text` field itself begins or ends on an opening/closing quotation mark (curly or straight)
that does not knit cleanly onto the adjacent word in a raw string join, not because the quoted
text is wrong. Each `text`, `before`, and `after` field was independently confirmed present
verbatim in its document (see script's own per-field FAIL check, which reports 0 failures), and
each was re-checked by hand against the source `text/` file below (section 2).

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| aim-sim-q001 | PASS | PASS | PASS (leaflet, unpaginated; page field names the leaflet title) | PASS |
| aim-sim-q002 | PASS | PASS | PASS | PASS |
| aim-sim-q003 | PASS | PASS | PASS | PASS |
| aim-sim-q004 | PASS | PASS | PASS | PASS |
| aim-sim-q005 | PASS | PASS | PASS | PASS |
| aim-sim-q006 | PASS | PASS | PASS | PASS |
| aim-sim-q007 | PASS | PASS | PASS | PASS |
| aim-sim-q008 | PASS | PASS | PASS | PASS |
| aim-sim-q009 | PASS | PASS | PASS | PASS |
| aim-sim-q010 | PASS | PASS | PASS | PASS |
| aim-sim-q011 | PASS | PASS | PASS | PASS |
| aim-sim-q012 | PASS | PASS | PASS (verso of title page) | PASS |
| aim-sim-q013 | PASS | PASS | PASS (Article 1) | PASS |
| aim-sim-q014 | PASS | PASS | PASS (Article 11) | PASS |
| aim-sim-q015 | PASS | PASS | PASS (Article 9) | PASS |
| aim-sim-q016 | PASS | PASS | PASS (Article 19, closing sentence) | PASS |
| aim-sim-q017 | PASS | PASS | PASS (Article 9) | PASS |
| aim-sim-q018 | PASS | PASS | PASS (Article 4, Section 2) | PASS |
| aim-sim-q019 | PASS | PASS | PASS (Article 20) | PASS |
| aim-sim-q020 | PASS | PASS | PASS (Article 19) | PASS |
| aim-sim-q021 | PASS | PASS | PASS (Article 18) | PASS |
| aim-sim-q022 | PASS | PASS | PASS (p. 4) | PASS |
| aim-sim-q023 | PASS | PASS | PASS (p. 8) | PASS |
| aim-sim-q024a | PASS | PASS | PASS (p. 8-9, split at printed page-break) | PASS |
| aim-sim-q024b | PASS | PASS | PASS (p. 9) | PASS |
| aim-sim-q025 | PASS | PASS | PASS ("Facts About the Mission") | PASS |

## 3. Field-by-field (V1)

Every dossier field carries either a supporting quote_id, a `[SECONDARY: ...]` citation, or a
`NOT FOUND (searched: ...)` / `UNVERIFIED` marker; none is stated bare. Spot-checked fields:

- F5: PASS — states plainly that no 1893/1895/1898 founding constitution text was located, and
  names the substitute document with its limits; matches discrepancies.md item 5.
- F9: PASS — the antecedent claim (SIM's textual dependence on CIM) is supported by the identical
  "weapons of our warfare... spiritual and not carnal" sentence in both documents (quote aim-sim-q016
  here; quote cim-q011 in the china-inland-mission dossier), and the summary explicitly states that
  this is the only direct textual evidence found, not a general claim beyond that.
- A4: PASS — correctly marked NOT FOUND rather than left blank or guessed.
- L7/S3: PASS — the two AIM figure sets (1917 vs. 1950-51) are kept distinct rather than merged,
  consistent with Rule 9 (record disagreement, do not resolve).
- S5: PASS — no aggregate casualty count is claimed beyond what the two individual deaths (Gowans,
  Kent) and Scott's death actually support; no invented tally.

## 4. Secondary citations checked

No dossier field in aim-sim relies on a `[SECONDARY: ...]` citation; all filled fields are PO
(primary-only) and either sourced from the four fetched texts or marked NOT FOUND/UNVERIFIED.
Not applicable.

## 5. Dates and numbers vs known-facts sheet

The known-facts sheet (`02_institution_roster.md` section 3) does not carry a dedicated entry for
`aim-sim` beyond the roster table row itself (SIM 1893, AIM 1895, both `[HIGH]`). Both dates as
recorded in this dossier (F2) match the roster row exactly.

| item | dossier value | roster value | result |
|---|---|---|---|
| SIM founding year | 1893 | 1893 [HIGH] | MATCH |
| AIM founding year | 1895 | 1895 [HIGH] | MATCH |
| SIM founders | Gowans/Gowan, Kent, Bingham | Rowland Bingham (named founder) | MATCH (roster names only Bingham; dossier adds Gowans and Kent from primary sources, which does not contradict the roster) |
| AIM founder | Peter Cameron Scott | Peter Cameron Scott | MATCH |

## 6. Verdict

VERIFIED (zero FAILs).
