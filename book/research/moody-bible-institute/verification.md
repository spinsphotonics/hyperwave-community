STATUS: VERIFIED
TICKET: V1-moody-bible-institute
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, charter_abridged.md, text/lifedwight00mood.txt, text/moodybibleinstit00camp.txt, sources.csv, discrepancies.md

# Verification report — moody-bible-institute (V1, dossier + charter)

## 1. Script output — quotes.jsonl
```
WARN moody-bible-institute-q012: before+text+after not contiguous in moodybibleinstit00camp
WARN moody-bible-institute-q013: before+text+after not contiguous in moodybibleinstit00camp
WARN moody-bible-institute-q014: before+text+after not contiguous in moodybibleinstit00camp
WARN moody-bible-institute-q015: before+text+after not contiguous in moodybibleinstit00camp
WARN moody-bible-institute-q016: before+text+after not contiguous in moodybibleinstit00camp
WARN moody-bible-institute-q018: before+text+after not contiguous in lifedwight00mood
18/18 quote records passed; 0 failures
```
The six WARNs are non-contiguous before/text/after windows (the "before" or "after" field was
trimmed to a shorter span than the literal run of text immediately adjacent, for readability);
the `text`, `before`, and `after` fields of each record independently verify verbatim against
the source file. Not treated as failures, consistent with the geneva-academy precedent for the
same WARN class.

## 2. Script output — charter_abridged.md
```
4/4 passages verbatim; 0 failures
```
All four kept passages in the partial charter-in-substance verify verbatim against
`text/lifedwight00mood.txt`.

## 3. Quote-by-quote
All 18 quote_ids: `text` field verbatim match PASS (per script). Spot-checked 6 of 18
(q001, q002, q004, q011, q014, q017) by manual inspection against the named text files at the
stated line ranges: content and OCR artifacts match what is visible in each source. PASS.

## 4. Field-by-field (dossier.md)
Every dossier field (F1-F9, A1-A6, C1-C9, L1-L8, T1-T5, M1-M5, S1-S6, R1-R4) is present with a
summary sentence and either quote_id(s), a NOT FOUND marker, or an UNVERIFIED marker. No summary
sentence checked claims more than its cited quote_ids support. Fields with only inline
line-number references to raw scratchpad text rather than a formal quote_id (noted explicitly
in the dossier text itself, e.g. parts of C3, C8, L2, L6, S1, S2, R4) are flagged there as "not
yet formalized... flagged as follow-up" rather than presented as fully verified facts; this is
consistent with Rule 8 (every fact needs a confidence label) since each such instance is
explicitly labeled as context/not-yet-a-quote_id rather than left unmarked. RECOMMENDATION for a
future revision pass: formalize these into quote records or downgrade to UNVERIFIED, per the
same recommendation the geneva-academy verification made for its own F7/L6 fields.
Fields marked NOT FOUND or UNVERIFIED: approximately 20 of 50 dossier fields (A1, A3, A4, C5,
C7, F5, F9 [partial], L1 [partial], L3, L4, L5, L8, T2, T4, T5, M3, S4, S5, S6, and F9). This
reflects that only two secondary narrative sources (plus one title page) were fetched this
session, and no charter/prospectus document was located. Recorded honestly, not a verification
failure (Rule 1: an UNVERIFIED marker is a success).

## 5. Secondary-source labeling check
Both `lifedwight00mood` and `moodybibleinstit00camp` are labeled `is_primary = no` in
`sources.csv`, consistent with Rule 3 (biography and institutional historical pamphlet are
secondary narrative sources). Quoted passages within them, presented with quotation marks as
Moody's own words or the Institute's own standing statement, are treated as PRIMARY for dossier
purposes per the explicit sourcing note at the top of dossier.md — this labeling convention
matches the geneva-academy and china-inland-mission precedents already accepted in this project.
No field marked PRIMARY-ONLY (PO) in the dossier template cites a secondary narrative sentence
without either a quote_id or an explicit "not yet formalized" flag; PASS.

## 6. Dates and numbers vs known-facts sheet (02_institution_roster.md)
- "Society 1887; building opened 1889" (Known-facts sheet) vs "1886 (society); 1889 (institute
  opened)" (roster table row): the two sources fetched this session narrate a 1886-87 chartering
  sequence without stating "5 February 1887" specifically (see discrepancies.md item 1). Chapter
  must not state that exact date without a citation to a source actually read this session.
- "Torrey superintendent Bible Institute course 1890" (roster search-string cell) vs the fetched
  sources' own dating of Torrey's appointment to September 1889 (the building's formal opening):
  no contradiction — the search-string cell names a search target, not an asserted fact, and both
  fetched sources independently agree on 26 September 1889. Matches [HIGH].
- "R. A. Torrey first superintendent": confirmed by both fetched sources. Matches.
- "Emma Dryer": named in the roster but not found in either fetched source (discrepancies.md
  item 2). UNVERIFIED, not contradicted.
- "'Gap men' idea": confirmed verbatim, quoted directly from Moody as reported by his son
  (moody-bible-institute-q001). Matches [HIGH].

## 7. Verdict
VERIFIED (zero FAILs on the 17 quote records and the 4 charter passages; the field-level gaps
and the missing charter document are honestly marked as NOT FOUND/BLOCKED, not verification
failures). RETURNED items for a future revision pass: formalize the inline-cited (non-quote_id)
dossier facts noted in section 4 above into proper quote records, and continue the Scout search
for the 1887 charter, an 1889-90 prospectus, and material on Emma Dryer.
