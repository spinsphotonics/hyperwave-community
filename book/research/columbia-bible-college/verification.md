STATUS: VERIFIED
TICKET: V1-columbia-bible-college
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; sources.csv; text/*.txt

# Verification report — columbia-bible-college

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/columbia-bible-college
WARN cbc-q017: before+text+after not contiguous in cbc-finial-1952
17/17 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/columbia-bible-college --charter
2/2 passages verbatim; 0 failures
```

The single WARN is the script's contiguity check on `before+text+after` for cbc-q017, whose
`text` field begins mid-quotation (immediately after an OCR ligature artifact rendering the
opening quotation mark as "ff", which is excluded from the quoted text itself); the `text` field
was independently confirmed present verbatim (0 FAILs reported).

## 2. Quote-by-quote

| quote_id | text matches | context matches | page correct | result |
|---|---|---|---|---|
| cbc-q001 | PASS | PASS | PASS | PASS |
| cbc-q002 | PASS | PASS | PASS | PASS |
| cbc-q003 | PASS | PASS | PASS | PASS |
| cbc-q004 | PASS | PASS | PASS | PASS |
| cbc-q005 | PASS | PASS | PASS | PASS |
| cbc-q006 | PASS | PASS | PASS | PASS |
| cbc-q007 | PASS | PASS | PASS (Page Six) | PASS |
| cbc-q008 | PASS | PASS | PASS (Page Six) | PASS |
| cbc-q009 | PASS | PASS | PASS (Page Six) | PASS |
| cbc-q010 | PASS | PASS | PASS (Page Ninety-two) | PASS |
| cbc-q011 | PASS | PASS | PASS (Page Ninety-two) | PASS |
| cbc-q012 | PASS | PASS | PASS | PASS |
| cbc-q013 | PASS | PASS | PASS | PASS |
| cbc-q014 | PASS | PASS | PASS | PASS |
| cbc-q015 | PASS | PASS | PASS | PASS |
| cbc-q016 | PASS | PASS | PASS | PASS |
| cbc-q017 | PASS | PASS (WARN on strict contiguity, see above) | PASS | PASS |

## 3. Field-by-field (V1)

Every dossier field carries a supporting quote_id, a paraphrase citation, or a
`NOT FOUND (searched: ...)` marker; none is stated bare. Spot-checked fields:

- F5: PASS — states plainly that no 1923-1930 catalog was located and names the 1948
  retrospective substitute used, cross-referencing discrepancies.md item 1.
- F3: PASS — the "Mother and Daddy McQuilkin" identification question (discrepancies.md item 2)
  is flagged rather than silently resolved.
- M2: PASS — the Ben Lippen "Victorious Life" connection is stated as McQuilkin's personal,
  separately-run ministry, not folded into Columbia Bible College's own stated purpose without
  support; the summary explicitly preserves that distinction.
- C1, C3, C8: PASS — the four-year/two-year/Graduate School structure is stated only as far as
  the section headings and enrollment figures actually show, with content differences among the
  tracks correctly marked NOT FOUND rather than guessed.

## 4. Secondary citations checked

No dossier field in columbia-bible-college relies on a `[SECONDARY: ...]` citation; all three
fetched documents (the 1939, 1948, and 1952 Finial yearbooks) are treated as primary institutional
publications. Not applicable.

## 5. Dates and numbers vs known-facts sheet

The known-facts sheet (`02_institution_roster.md` section 3) carries no dedicated entry for
`columbia-bible-college` beyond the roster table row itself (founded 1923, Robert C. McQuilkin,
`[HIGH]`).

| item | dossier value | roster value | result |
|---|---|---|---|
| Founding year | 1923 (confirmed indirectly via the 1948 25th-anniversary framing) | 1923 [HIGH] | MATCH |
| Founder | Robert C. McQuilkin | Robert C. McQuilkin | MATCH |
| Place | Columbia, S.C. | Columbia, S.C. | MATCH |
| Motto | "To Know Him and to Make Him Known" | "'To know Him and to make Him known'" (roster's own "why included" cell) | MATCH |

## 6. Verdict

VERIFIED (zero FAILs).
