STATUS: DONE
TICKET: S-maf
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (maf row); book/plan/05_repositories_and_search_strings.md (general repositories)

# Scout notes — maf

**Note on process:** As with the other Part V Tier B institutions this session, Scout,
Fetcher, and Extractor work were combined rather than run as strictly separate tickets.

## Target documents (from roster row)

1. MAF statement of purpose (1945) [VERIFY]. NOT located as a digitized period document; see
   `sources.csv` row `maf-statement-of-purpose-1945` (NONE). Current substitutes found and
   used in `charter_abridged.md`: `maf-vision-mission` and `maf-statement-of-faith`.

## Special step

MAF is not one of the four institutions named in Procedure S step 5's special list, but the
task instructions for this session specifically named Russell Hitt's *Jungle Pilot* (1959) as
a priority in-copyright source to check. Four Internet Archive identifiers for the book were
located (`junglepilotstory0000hitt`, `junglepilotgripp0000hitt`, `junglepilotlifea00hitt`,
`junglepilotlifew0000hitt`); all four were attempted via the specified
`https://archive.org/download/<id>/<id>_djvu.txt` URL pattern and all four returned HTTP 401
(controlled digital lending), matching the same block the `wycliffe-sil` chapter's research
session recorded for a different book on the same archive. BLOCKED — see `sources.csv` row
`hitt-jungle-pilot-1959`.

The Billy Graham Center Archives' own MAF finding aid (Collection 176, Wheaton College) was
attempted and BLOCKED by the same Cloudflare Turnstile human-check gate already logged by
three earlier chapters' research sessions this book — see
`text/wheaton-maf-collection176.REQUEST.md`. A close substitute was found, however: a Wheaton
Archives staff-authored blog post ("A Gal, A Plane & A Dream") that directly quotes, with
folder citations, several documents from the same underlying collection (there cited as
"Collection 136"). This is used extensively in the dossier and chapter as the richest
available primary-adjacent source on MAF's founding and on Betty Greene's role within it.

## Secondary sources (from roster "Secondary" cell)

- Hitt, *Jungle Pilot* (1959) — see above, BLOCKED.

## Bibliography mining

`maf-5-martyred` and `maf-nate-saint` (MAF's own institutional pages) independently name the
other four missionaries killed alongside Nate Saint (Jim Elliot, Ed McCully, Roger Youderian,
Pete Fleming) and their respective sending agencies (Gospel Missionary Union for Youderian;
Plymouth Brethren for Elliot, McCully, and Fleming) — material also relevant to this book's
`cmml-brethren` chapter, cross-referenced rather than duplicated in full here.

## Institution-specific findings

Two independently fetched sources give different 1945 founding dates/framings: MAF-US's own
site and the Wheaton Archives blog both date incorporation to 20 May 1945 in Los Angeles,
while MAF International's site instead describes "the registration of MAF in London in 1945"
without a specific date — recorded in `discrepancies.md` #1 as a difference between the US and
UK/international branches' own founding moments rather than a contradiction requiring
resolution. The organization's own early self-description ("Christian Airman's Missionary
Service" per the Wheaton Archives blog, vs. "Christian Airmen's Missionary Fellowship" per
MAF-US's own current page) is recorded in `discrepancies.md` #2. No pre-1900 nickname-origin
step applies to this institution (that special step is unique to `geneva-academy`).
