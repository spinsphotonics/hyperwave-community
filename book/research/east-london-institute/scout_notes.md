STATUS: DONE
TICKET: S-east-london-institute
ROLE: Scout (folded into a single-session Scout+Fetcher+Extractor pass; see 00_README.md rule 4 note below)
INPUTS READ: book/plan/02_institution_roster.md (row `east-london-institute`); book/plan/05_repositories_and_search_strings.md (`east-london-institute` section)

NOTE ON PROCESS: Given the single-session nature of this task, Scout and Fetcher steps were combined: each target document was searched, and where a full-text hit was found on Internet Archive it was fetched immediately rather than logged separately and fetched in a later ticket. All hits (including NONE results) are recorded in `sources.csv`.

## Target documents (from roster's "Key primary documents" cell)

1. Institute annual reports -- FOUND: `wideworldourwork00guin` (1886), `notuntousarecord00guinuoft` (1893), `ourhelpersatwork00guin` ([1894]); also found but not usable, `SomeAreFallenAsleep...` (404, see sources.csv).
2. *The Regions Beyond* (magazine) -- NOT FOUND as digitized full text; microfilm-only holding located at UW-Madison (reels 27-46). See sources.csv row `eli-regionsbeyond-not-fetched`.
3. Guinness's *Prospectus* [VERIFY] -- NOT FOUND as a separately titled document. See sources.csv row `eli-prospectus-not-found`.
4. Fanny Guinness, *The New World of Central Africa* -- FOUND and fetched: `newworldofcentra00guin` (1890). Does not name the Institute itself in its text (see sources.csv note); not cited.

## Secondary sources (roster's "Secondary" cell)

- Fiedler, *The Story of Faith Missions* (1994) -- not located as a free digitized full text this session (recent, likely still in copyright); not fetched.

## Additional primary documents found and fetched (not on the roster's original list, surfaced by search)

- `centralafricacon00guin` (1884), a short reprinted address by Fanny Guinness on the Congo mission.
- `drharryguinnessl0000unse` ([1932]), C. W. Mackintosh's biography of the founders' son and successor Harry Grattan Guinness -- treated SECONDARY (see dossier and sources.csv), but published by the Regions Beyond Missionary Union itself from family papers, and used for corroborating dates, the Hudson Taylor connection, and named alumni.

## Institution-specific notes

No institution-specific Scout step is listed for `east-london-institute` in `03_research_protocol.md` step 5 (the special per-institution steps there name only `geneva-academy`, `china-inland-mission`, `pastors-college`, `andover-seminary`/`abcfm`, `london-theological-seminary`, and the Elliot-related institutions). None applies here.

## Founding-date discrepancy flagged for the Extractor

Two of the Institute's own publications (`wideworldourwork00guin`, 1886, and `ourhelpersatwork00guin`, [1894]) state the Institute was founded/opened in **1872**. Two other sources -- `notuntousarecord00guinuoft` (1893, also Institute-published) and the secondary Mackintosh biography (`drharryguinnessl0000unse`, [1932]) -- state **1873**. The roster's known-facts sheet gives 1873 [HIGH]. Recorded in `discrepancies.md`; not resolved here per Rule 9.
