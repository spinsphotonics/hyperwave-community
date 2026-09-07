STATUS: VERIFIED
TICKET: V1-wheaton-college
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, text/*.txt (all six documents), sources.csv, discrepancies.md, charter_abridged.md

# Verification report — wheaton-college (V1, dossier)

## 1. Script output
```
WARN wheaton-college-q012: before+text+after not contiguous in fromthevault-humble-beginnings
WARN wheaton-college-q013: before+text+after not contiguous in elisabethelliot-journals-part1
25/25 quote records passed; 0 failures
```
```
10/10 passages verbatim; 0 failures   (--charter check against wheaton-catalog-1904)
```
The two WARNs are non-contiguous before/text/after window checks only (each of `text`, `before`,
`after` independently verifies verbatim); consistent with the geneva-academy precedent, not
treated as failures.

## 2. Quote-by-quote
All 25 quote_ids: `text` field verbatim match PASS. Manually spot-checked 8 of 25
(q001, q004, q007, q008, q013, q017, q020, q024) by opening the named text/ file and confirming
the "page" field's description matches the surrounding content. PASS for all 8 spot-checked.

## 3. Field-by-field
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4) is present. Every
summary sentence checked against its cited quote_id(s); none claims more than the quotes support.
PASS for all cited fields. NOT FOUND fields (roughly half the template) are correctly labeled
rather than filled with plausible-sounding invented detail — most conspicuously, no specific
Statement of Faith wording or date is stated anywhere in the dossier, and Elisabeth Howard's
Prairie Bible Institute span is stated as "1948" only, not silently extended to match the
roster's own "[VERIFY]"-flagged "1948-49." PASS (Rule 8: honest NOT FOUND is a success).

## 4. Secondary citations checked
`wheaton-catalog-1904` fields are labeled (PO) — is_primary = yes in sources.csv, a genuine
public-domain published catalog. `fromthevault-*`, `elisabethelliot-journals-part1`, and
`elisabethelliot-timeline` fields are labeled (PS) or noted as quoted-at-second-hand, matching
their is_primary = no rows, EXCEPT where a specific primary document is quoted within them (the
Blanchard 1859 letter, Jim Elliot's own 1948 journal text) — those passages are treated as
primary-at-second-hand, on the same logic the geneva-academy dossier applied to Borgeaud's quoted
archival passages. Checked: no field marked PRIMARY-ONLY (PO) rests solely on an is_primary=no
document without this second-hand-primary distinction being stated. PASS.

## 5. Dates and numbers vs known-facts sheet
- Jim Elliot born 8 October 1927, Portland, Oregon: matches known-facts sheet [HIGH]; independently
  confirmed by wheaton-college-q016 (Elisabeth Elliot Foundation timeline), not merely asserted
  from the roster.
- Jim Elliot Wheaton BA 1949: consistent with wheaton-college-q013/q014 (entered fall 1945) and the
  "Junior Year at Wheaton College, 1948" title of the Journals excerpt; matches known-facts sheet
  [HIGH]. The exact degree (A.B. vs other) is not independently confirmed this session (C8).
- Elisabeth Howard Wheaton BA 1948: this session's fetched timeline states she "enters Wheaton
  College" in 1944 [wheaton-college-q017] but does not itself state a 1948 graduation date in the
  material fetched; consistent with, but not independently re-derived from, the known-facts
  sheet's "Wheaton BA 1948 (Greek)" [VERIFY years].
- Elisabeth Howard Prairie Bible Institute: known-facts sheet gives "1948-49 [VERIFY Prairie
  dates]"; this session's fetched source gives only "1948" [wheaton-college-q019] — logged as
  discrepancy 1, not silently reconciled.
- Jim and Elisabeth married 8 October 1953, Quito: matches known-facts sheet [HIGH]; consistent
  with (though not independently re-quoted from) the fetched timeline's entry of the same date.
- Five missionaries killed January 1956: matches known-facts sheet [HIGH] "8 Jan 1956"; this
  session's own fetched source states "January 1956" [wheaton-college-q022] without independently
  re-confirming the specific day of the month; not a contradiction, but the day-level precision in
  the chapter rests on the roster's [HIGH]-labeled known-facts sheet, not on a primary document
  this session itself opened and read down to the day.
- Elisabeth Elliot's spelling ("Elisabeth," not "Elizabeth"; not to be confused with John Eliot):
  checked against every quote and every dossier sentence in this file — PASS, spelled correctly
  throughout, and no sentence conflates her with John Eliot the Puritan missionary.

## 6. Verdict
VERIFIED (zero FAILs on the 25 quote records and the 10 charter passages; NOT FOUND fields and the
three logged discrepancies are honestly and correctly marked per Rule 8/Rule 9, not verification
failures).
