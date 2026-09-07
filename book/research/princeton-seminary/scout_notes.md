STATUS: DONE
TICKET: S-princeton-seminary
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (row: princeton-seminary); book/plan/05_repositories_and_search_strings.md (section: princeton-seminary)

# Scout notes — Princeton Theological Seminary (`princeton-seminary`)

## Target documents (from roster "Key primary documents" cell)
1. *The Plan of the Theological Seminary of the Presbyterian Church* (1811)
2. Archibald Alexander's inaugural address (12 Aug 1812)
3. Miller's charge [roster's phrase; see finding below]
4. Catalogues

## Searches run and results
- `"Plan of the Theological Seminary" Presbyterian Church 1811 archive.org full text` (WebSearch): found archive.org identifier `planoftheologica00pres_0` (the original 1811 Philadelphia printing by Jane Aitken, digitized from a copy presented to the Princeton Theological Seminary Library). Fetched full OCR text. Also surfaced a second, later printing at `planoftheologica00prin` (located in Princeton, undated cover title) and a Library of Congress record for a 3rd printing (Elizabeth-Town, I.A. Kollock, 1816) at loc.gov/item/ltf90014661 -- logged in sources.csv row `princeton-plan-loc` but not fetched this session (the 1811 original was used as the primary text of record).
- `Archibald Alexander inaugural address 1812 seminary` (WebSearch): found archive.org identifier `sermondelivered00millgoog` (Google Books scan of the University of Michigan copy of the full published volume). Fetched full OCR text. This single bound volume contains three separate pieces: (a) Samuel Miller's sermon "The Duty of the Church to Take Measures for Providing an Able and Faithful Ministry" (2 Timothy 2:2), delivered immediately before the inauguration; (b) Archibald Alexander's own Inaugural Discourse (John 5:39, "Search the Scriptures"); (c) Philip Milledoler's "Charge to the Professor and Students of Divinity."
- `Miller charge inauguration Alexander 1812` (WebSearch, folded into the search above): **Finding** -- the roster cell's "Miller's charge" appears to conflate two different pieces within the same 1812 volume. Samuel Miller delivered the opening *sermon*, not the *charge*; the charge to Alexander and the students was delivered by Philip Milledoler. Both are in the same fetched volume (document_id `inaug1812`). Recorded here rather than silently corrected, per Rule 9 (discrepancies are recorded, not resolved) -- see discrepancies.md.
- `Princeton theological seminary catalogue 1815` (WebSearch): found archive.org identifier `biographicalcata00prin_2`, *Biographical Catalogue of Princeton Theological Seminary, 1815-1932*, compiled by the Seminary's own Registrar and published by its Trustees (1933). Fetched full OCR text. Front matter gives a dated founding chronology; faculty roster gives Alexander's and Miller's professorship dates; alumni entries begin with the class admitted 1812 / completing 1815.

## Secondary sources (roster "Secondary" cell)
- Calhoun, *Princeton Seminary* vols. 1-2 (1994-96): in-copyright, not digitized as full text on Internet Archive; not located this session. Logged as NOT FETCHED in sources.csv (row `calhoun-princeton-seminary`).
- Moorhead: roster gives only a surname, no title; not identified or searched further this session. Logged as NOT FETCHED (row `moorhead-secondary`).

## Bibliography mining
Not performed this session (time-boxed); the two fetched primary documents (the Plan itself, and the inauguration volume) plus the institutional biographical catalogue supply direct coverage of Founding, Admission, Curriculum, Piety/Common Life, and Emphasis without needing to mine a secondary history's footnotes. Flagged as a follow-up if Calhoun or Moorhead are later obtained.

## Institution-specific note (risk register, Part III/07_risk_register)
Per `07_risk_register_and_qa.md` section B: this chapter concerns Princeton **Theological Seminary** (founded 1812, Plan adopted 1811), a separate institution from the **College of New Jersey** (1746, Princeton University's precursor, covered elsewhere as slug `college-of-new-jersey`). The Plan's own text (Art. and closing minutes) records that the 1811 General Assembly appointed a committee to confer with a committee of "the Trustees of New Jersey College" about possibly locating the Seminary at the College's site and framing a joint constitutional relationship -- this is a documented institutional *link* between the two bodies, not an identity between them; the two remained, and are treated here as, distinct institutions.

## Done-when checklist
- [x] Every target document has at least one sources.csv row (hit or NONE).
- [x] Every row has all columns filled.
- [x] scout_notes.md lists targets, new targets found, and institution-specific findings (Miller/Milledoler charge discrepancy; College of New Jersey distinction).
- [x] No content from any document has been summarized here beyond locating it (content extraction happens in dossier.md).
