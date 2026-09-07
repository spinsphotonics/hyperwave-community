STATUS: DONE
TICKET: S-andover-seminary
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (row: andover-seminary); book/plan/05_repositories_and_search_strings.md (section: andover-seminary)

# Scout notes — Andover Theological Seminary (`andover-seminary`)

## Target documents (from roster "Key primary documents" cell)
1. *Constitution and Associate Statutes of the Theological Seminary in Andover* (1808)
2. *Associate Creed* (part of the Additional Statutes, 1808)
3. *Laws of the Theological Institution* [VERIFY separate document]
4. Constitution of the Society of the Brethren (1808) [VERIFY]

## Searches run and results
- `"Constitution and Associate Statutes of the Theological Seminary in Andover"` (WebSearch, general web): found catalog/reprint listings (HathiTrust, Amazon/AbeBooks classic reprints); no standalone free digitized full text located independently of a history that reprints it. See sources.csv row `andover-associate-creed-1808-standalone`.
- `Woods "History of the Andover Theological Seminary"` (WebSearch): found archive.org identifier `historyofandover00woodrich` (1885). Fetched. Its documentary appendix (pp. 232-253 of the printed book, per running heads in the OCR) reprints the full text of Target 1 and Target 2 verbatim, under the headings "CONSTITUTION OF THE THEOLOGICAL SEMINARY" and "ADDITIONAL STATUTES" (the latter containing the Associate Creed itself, beginning "I believe that there is one and but one living and true GOD..."). This satisfies Targets 1 and 2 as a verbatim documentary reprint (treated PRIMARY; see text/woods1885.txt header note).
- `Woods "History of the Andover Theological Seminary"` also surfaced archive.org identifier `historyofandover00rowe` (Rowe, 1933, a later institutional history). Fetched for cross-check; not used for any quote this session (see sources.csv).
- `"Laws of the Theological Institution in Andover"` (Target 3): a chapter of Woods 1885 is titled "LAWS OF THE THEOLOGICAL INSTITUTION" (running heads at approx. lines 11301, 11957 of the fetched OCR) -- this appears to be Woods's own narrative account of the Laws rather than (or in addition to) a documentary reprint; not separately fetched as a standalone pamphlet this session. Left for Extractor to determine from woods1885.txt whether it is a verbatim reprint.
- `"Society of the Brethren" Williams Andover constitution` (Target 4): searched the fetched text of both woods1885 and rowe1933 for the string "Society of the Brethren" and "BRETHREN" -- zero hits in either file. NOT FOUND this session. See sources.csv row `andover-brethren-constitution` and discrepancies.md.

## Secondary sources
- Rowe 1933 (`rowe1933`) logged as secondary; fetched but not quoted this session.
- Anderson, *Memorial Volume of the First Fifty Years of the ABCFM* (1861) also discusses the Andover "Brethren" and the 1810 Bradford memorial in some detail; that document is logged and used under the `abcfm` research folder (its primary subject), and is cross-cited here in dossier field F9 as [SECONDARY-ADJACENT, quoted primary reminiscence] where it bears directly on Andover's own founders/antecedents.

## Bibliography mining
Woods 1885's own footnotes were not separately mined for further primary citations this session (time-boxed); flagged as follow-up.

## Done-when checklist
- [x] Every target document has at least one sources.csv row (hit or NONE).
- [x] Every row has all columns filled.
- [x] scout_notes.md lists targets, new targets found, and institution-specific findings (Society of the Brethren search).
- [x] No content from any document has been summarized here beyond locating it (content extraction happens in dossier.md).
