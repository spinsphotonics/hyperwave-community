STATUS: DONE
TICKET: S-abcfm
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (row: abcfm); book/plan/05_repositories_and_search_strings.md (section: abcfm)

# Scout notes — American Board of Commissioners for Foreign Missions (`abcfm`)

## Target documents (from roster "Key primary documents" cell)
1. Constitution (1810/1812)
2. *Instructions of the Prudential Committee to the Rev. Messrs. Judson, Nott, Newell and Hall* (Salem, 1812)
3. Haystack/Brethren records
4. Judson's letters

## Searches run and results
- `"American Board of Commissioners for Foreign Missions" constitution 1810` (WebSearch): general catalog/Wikipedia hits; no standalone free digitized full text of the Massachusetts Act of Incorporation located independently this session. See sources.csv row `abcfm-1810-1812-constitution-standalone`.
- `Anderson "Memorial Volume" American Board fifty years 1861 archive.org` (WebSearch): found archive.org identifier `memorialvolumeof00andeiala`. Fetched. Contains, in quotation marks, the Board's charter purpose clause ("for propagating the gospel in heathen lands by supporting missionaries and diffusing a knowledge of the Holy Scriptures"), the 1812 General Assembly reply (quoted verbatim), and first-hand reminiscence letters written for this 1861 volume by three surviving 1810-1812 participants (Samuel Nott, John Keep, Noah Porter) describing the Bradford memorial and the Salem ordination. Treated PRIMARY where directly quoted; Anderson's narrative connective prose SECONDARY.
- `"Instructions of the Prudential Committee" Judson Nott Newell Hall 1812 Salem` (WebSearch): did not directly surface the full text; led to a follow-up search for "First Ten Annual Reports of the American Board" (per the task instructions' suggested search).
- `"First Ten Annual Reports of the American Board" archive.org full text` (WebSearch): found archive.org identifier `firsttenannualre00amerrich` (1834 reprint volume). Fetched. **This volume prints the full text of Target 2** verbatim (pp. 38-42 of the printed book): "INSTRUCTIONS GIVEN BY THE PRUDENTIAL COMMITTEE OF THE AMERICAN BOARD OF COMMISSIONERS FOR FOREIGN MISSIONS, TO THE MISSIONARIES TO THE EAST, FEBRUARY 7, 1812," addressed by name to Judson, Nott, Newell, Hall, and Rice, closing "A true copy from the Records of the Prudential Committee, Attest, SAMUEL WORCESTER, Salem, Feb. 7, 1812." This satisfies Target 2 as the chapter's charter document per the E-abcfm ticket instruction.
- Target 3 (Haystack/Brethren records): Anderson 1861 names the "Brethren" society (formed at Williams College 1808, transferred to Andover) and quotes its stated object, but its own constitutional text was not located as a standalone document this session -- see sources.csv row `abcfm-brethren-society` and cross-reference in `andover-seminary/discrepancies.md` item 2.
- Target 4 (Judson's letters): Judson's 1812-1813 letters on his change of Baptist views are documented and quoted at length in Wayland 1853 (fetched under the `judson-triennial-convention` folder, since that is the institution most directly concerned with that change of views per the roster); cross-referenced here for context but not re-quoted separately in this folder to avoid duplicating citations across two dossiers without a shared source.

## Secondary sources
- Anderson 1861 also used as the source of the Board's own founding narrative (F1-F4), treated PS per field type.

## Bibliography mining
Not separately performed this session (time-boxed).

## Done-when checklist
- [x] Every target document has at least one sources.csv row (hit or NONE).
- [x] Every row has all columns filled.
- [x] scout_notes.md lists targets, new targets found (firstten1834, located via the task's suggested "First Ten Annual Reports" search), and institution-specific findings.
- [x] No content from any document has been summarized here beyond locating it and stating what it contains at the document level.
