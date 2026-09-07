STATUS: DONE
TICKET: S-westminster-seminary
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (row for westminster-seminary); book/plan/05_repositories_and_search_strings.md (westminster-seminary section)

# Scout notes — Westminster Theological Seminary (`westminster-seminary`)

## Target documents (from roster)
1. J. Gresham Machen, *Westminster Theological Seminary: Its Purpose and Plan* (opening address, 25 Sept 1929; published *The Presbyterian* 99, October 10, 1929, pp. 6-9).
2. Westminster Theological Seminary, first catalogue (1929-30).
3. The Plan of the Seminary (referenced by the roster as a possible separate constitutional document).

## Searches run
- `"Westminster Theological Seminary" "Its Purpose and Plan" Machen 1929 full text` (web) — found readmachen.com bibliography record (metadata only, no full text) and several secondary sites quoting the address at length.
- `Machen "Westminster Theological Seminary" opening address September 1929 archive.org` (web) — found opc.org's *The Presbyterian Conflict* ch. 4 (Rian, 1940) and *Fighting the Good Fight* pt. 1 (Hart & Muether), both quoting the address; no archive.org item for the address itself.
- `Ned Stonehouse "J. Gresham Machen: A Biographical Memoir" archive.org full text` (web) — book confirmed to exist (1954, Eerdmans); no free full text located; OPC lists an ePub but this was not fetched.
- `"Westminster Theological Seminary" first catalogue 1929 1930 archive.org` (web) — no digitized 1929-30 catalogue found; Montgomery Library (WTS's own archive) is the likely holder but its finding aids were not searched item-by-item this session.
- `archive.org "The Presbyterian" 1929 volume 99 weekly religious newspaper` — no digitized run of the periodical found.
- `"J. Gresham Machen" "Selected Shorter Writings" 2004 archive.org` — found archive.org identifier `selectedshorterw0000mach`; item is controlled-digital-lending only (confirmed by a 401 Authorization Required response when the `_djvu.txt` OCR endpoint was requested directly).
- Internet Archive advanced search (`title:(Westminster Theological Seminary)`, `Machen Westminster Theological Seminary`, `title:(Seeking a Better Country)`) — found several relevant book records, all lending-only or off-topic (Collected Writings of John Murray, etc.); none gave open full text of the founding documents.
- `"Westminster Theological Seminary" founded 1929 Machen` (Wikipedia) — background facts and one quotation.

## Findings
- The complete text of Machen's 1929 address was NOT located as an open digitized primary source this session. Three separate secondary/tertiary web pages (PCA Historical Center's "This Day in Presbyterian History," the Orthodox Presbyterian Church's "Today in OPC History," and Westminster Seminary California's own website) each independently reproduce different, non-overlapping verbatim excerpts of the address, citing the original periodical (*The Presbyterian* 99, Oct. 10, 1929, pp. 6-9). These three excerpts, together, are the fullest primary text recovered (see `text/machen-pcahistory.txt`, `text/machen-opctoday.txt`, `text/machen-wscal.txt`, and the composite `text/machen1929-address.txt`).
- The 1929-30 catalogue was NOT located. BLOCKED — print/archive-only as far as searched; Westminster's own Montgomery Library is the named holder (wts.edu/library) but was not queried at the item level this session.
- Edwin H. Rian's *The Presbyterian Conflict* (1940), ch. 4, gives a full secondary narrative of the founding (meetings, dates, founders, faculty, funding, doctrinal basis, and a quoted paragraph from the Seminary's own 1937-38 catalogue) and is used extensively for background facts. Rian was a founding trustee.
- D. G. Hart and John R. Muether, *Fighting the Good Fight* (1995), corroborates the founding narrative and quotes the same "though Princeton is dead" passage with wording that differs from the Rian and OPC versions — see `discrepancies.md`.
- No pre-1900 or period-specific "school of death" type nickname search was applicable to this institution (that special step in Procedure S applies only to `geneva-academy`).

## New targets found via bibliography mining
- Ned B. Stonehouse, ed., *What Is Christianity? and Other Addresses* (Eerdmans, 1951) — reprints the 1929 address; NEW: not in the original roster cell by this exact title; not located as digitized full text.
- D. G. Hart, ed., *J. Gresham Machen: Selected Shorter Writings* (P&R, 2004), pp. 187-194 — reprints the address; archive.org copy is lending-only.

## Done-when checklist
- [x] Every target document has at least one row in sources.csv (hit or NONE/BLOCKED).
- [x] Every sources.csv row has all columns filled.
- [x] This file lists targets, new targets, and findings.
- [x] No content was invented; all quoted material in dossier.md and quotes.jsonl is copied from fetched text.
