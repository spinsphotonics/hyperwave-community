STATUS: RETURNED
TICKET: V1-glasgow-bti
ROLE: Verifier
INPUTS READ: dossier.md; quotes.jsonl; charter_abridged.md; discrepancies.md; sources.csv

# Verification report — glasgow-bti

## 1. Script output

```
$ python3 plan/templates/check_quotes.py research/glasgow-bti
0/0 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/glasgow-bti --charter
FAIL: charter_abridged.md does not name a document_id in backticks
```

The quote-record check passes trivially: there are zero quote records because no source text was
located this session (see discrepancies.md item 2), and an empty ledger contains no unverified
claims. The charter check reports one FAIL, by design: `charter_abridged.md` explicitly declines
to name a `document_id`, because no founding document or substitute primary text exists to
abridge. This FAIL is not a data-integrity defect — no fabricated or misquoted passage is
present — it is the mechanical signature of a genuine, documented gap.

## 2. Quote-by-quote

Not applicable; `quotes.jsonl` is empty.

## 3. Field-by-field (V1)

Every PO (primary-only) field in the dossier is marked `NOT FOUND`; every PS
(primary-or-secondary) field that is filled cites `[SECONDARY: tertiary-wikipedia-icc]` and
states only bare facts (dates, names, addresses) with no interpretation added beyond what that
source states. No field is left blank, and no field states a claim without a label.

| field | result | note |
|---|---|---|
| F1-F2, F4 | PASS | dates/place drawn from tertiary-wikipedia-icc, labeled, no PO field affected |
| F3 | PASS | correctly flagged UNRESOLVED given the roster/source conflict over Anderson's role; not silently resolved either way |
| F5 | PASS | states plainly that no document was located; does not substitute an unlabeled guess |
| F6-F9, all A/C/L(PO)/T(PO/PS beyond names)/M/S/R | PASS | uniformly NOT FOUND, no invented content |

## 4. Secondary citations checked

The single background source cited, `tertiary-wikipedia-icc` (Wikipedia, "International Christian
College"), was re-opened and each cited fact (1892 opening; Bothwell Street; 1898 move; the
Moody/Sankey connection; the Principal list with dates) confirmed present in the fetched HTML.
Result: PASS for accurate transcription of what the source states. This is a tertiary source and
its own reliability is not independently verified by this session; it is used only for
background, per Rule 3, and nothing in the dossier's PO fields or in the (absent) Charter section
relies on it.

## 5. Dates and numbers vs known-facts sheet

The known-facts sheet (`02_institution_roster.md` section 3) carries no dedicated entry for
`glasgow-bti` beyond the roster table row itself (founded 1892, John Anderson, `[HIGH]`). The
roster's `[HIGH]` label for the 1892 date is not contradicted by the one background source found
(which agrees on 1892); the roster's naming of John Anderson as founder is not confirmed by that
source, which instead dates his principalship from 1898 — flagged in discrepancies.md item 1
rather than silently adopted.

## 6. Verdict

RETURNED (one FAIL, in the charter-abridgement check only, by design and fully explained in
sections 1 and above). The dossier itself has zero FAILs: every field is honestly labeled, and no
invented fact appears anywhere in this research folder. A human reviewer's sign-off should treat
the charter FAIL as an accepted, documented gap rather than a defect to fix, unless and until a
primary source is located in a future session.
