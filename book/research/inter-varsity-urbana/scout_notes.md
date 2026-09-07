STATUS: DONE
TICKET: S-inter-varsity-urbana
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (inter-varsity-urbana row); book/plan/05_repositories_and_search_strings.md (general repositories)

# Scout notes — inter-varsity-urbana

**Note on process:** As with `wycliffe-sil`, this session combined Scout, Fetcher, and
Extractor work rather than running them as strictly separate tickets. Search strings run
before content was read for extraction; the resulting search log lives directly in
`sources.csv`.

## Target documents (from roster row)

1. IVCF doctrinal basis (1941). NOT located as a digitized period document; see `sources.csv`
   row `ivcf-doctrinal-basis-1941` (NONE). Current substitute found and used in
   `charter_abridged.md`: `ivcf-statement-agreement` (adopted by the Board of Trustees,
   20 October 2000), explicitly flagged as not the founding-era text.
2. First convention program (1946). NOT located; see `sources.csv` row
   `ivcf-first-convention-program-1946` (NONE).

## Special step (per 03_research_protocol.md Procedure S step 5)

This institution is not one of the four named in the protocol's special step
(`wheaton-college`, `cmml-brethren`, `prairie-bible-institute`, `wycliffe-sil`), but the
Jim Elliot/Urbana 1948 connection named in the roster ("Jim Elliot attended Urbana 1948
[VERIFY]") warranted the same kind of check. The Billy Graham Center Archives' own IVCF
finding aid (Collection 300, at Wheaton College) was attempted and BLOCKED by the same
Cloudflare Turnstile human-check gate already logged by the `wycliffe-sil` session — see
`text/wheaton-ivcf-collection300.REQUEST.md`.

## Secondary sources (from roster "Secondary" cell)

- Hunt and Hunt, *For Christ and the University* (1991/1992) — not located as a digitized
  open text this session. Cited extensively at second hand via `wiki-urbana`, which lists it
  as its principal source (pp. 127-129, 172-175, 217-218, 303-305, 356-358). See `sources.csv`
  row `hunt-for-christ-university`.

## Bibliography mining

`wiki-urbana`'s own references list points to A. Donald MacLeod's 2007 biography of Stacey
Woods (announced in `ivcf-stacey-woods`, a press release fetched this session) and to two
further secondary works (Norman E. Thomas, *Missions and Unity*, 2010; Michael Sills, *The
Missionary Call*, 2008) cited only for the 1946/1948 dating, not fetched this session. Neither
Thomas nor Sills is added as a full row since only a single date fact from each was needed and
that fact is already independently corroborated by `urbana-story` and `ivcf-ifes-history`.

## Institution-specific findings

Three sources fetched this session give three different founding dates/framings for IVCF's
U.S. beginning: IVCF's own site states "November 14, 1941" as the "official beginning"
(`ivcf-ifes-history`, `ivcf-growing-love-1941`); the Encyclopedia.com/Gale reference-work
entry states the American branch was established "in September 1941" (`encyclopedia-ivcf`);
and the 2007 IVP press release announcing MacLeod's Woods biography states Woods "was the
founder of the InterVarsity Christian Fellowship in the United States in 1939-1940"
(`ivcf-stacey-woods`). All three are recorded in `discrepancies.md` rather than resolved.
No pre-1900 nickname-origin step applies to this institution (that special step is unique to
`geneva-academy`).
