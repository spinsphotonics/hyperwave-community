STATUS: DONE
TICKET: S-wycliffe-sil
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (wycliffe-sil row); book/plan/05_repositories_and_search_strings.md (wycliffe-sil section and general repositories)

# Scout notes — wycliffe-sil

**Note on process:** This session combined Scout, Fetcher, and Extractor work rather than
running them as strictly separate tickets (the roster/protocol files were read and search
strings run before any content was read for extraction, but the resulting search log lives in
`sources.csv` directly rather than a separate scouting pass). Targets below are the roster's
own "Key primary documents" cell.

## Target documents (from roster row)

1. SIL/WBT doctrinal statement and statement of purpose (1934–42). NOT located as a digitized
   period document; see `sources.csv` row `wbt-sil-doctrinal-statement-1934-42` (NONE). Current
   substitute found and used in `charter_abridged.md`: `wycliffe-our-beliefs`.
2. Camp Wycliffe first-session announcement (1934) [VERIFY]. NOT located; see `sources.csv`
   row `camp-wycliffe-announcement-1934` (NONE). Holdings per roster: Wycliffe USA archives,
   Orlando; SIL archives, Dallas — not confirmed digitized.
3. Townsend's early letters. NOT located directly; a 1933 prayer-meeting quotation attributed
   to Townsend's circle survives only at second hand via Wikipedia (citing Hartch 2006); see
   quote_id wycliffe-sil-q031.

## Special step (per 03_research_protocol.md Procedure S step 5)

"`wheaton-college`, `cmml-brethren`, `prairie-bible-institute`, `wycliffe-sil`: record the
archive finding aid for the Elliot papers and the institutional catalogs for 1947–1952."

Both Wheaton College Archives finding aids (Jim Elliot Collection, resource 466; Elisabeth
Elliot Papers, resource 484) were attempted and BLOCKED by a Cloudflare Turnstile human-check
gate — see `text/wheaton-jimelliot-archives.REQUEST.md`. No 1947–1952 institutional catalog
(SIL/Camp Wycliffe or Wheaton) was located as a digitized text this session.

## Secondary sources (from roster "Secondary" cell)

- Hefley and Hefley, *Uncle Cam* (1974) — located on Internet Archive (item
  `unclecamstoryofw0000hefl`) but BLOCKED (controlled digital lending, HTTP 401). See
  `sources.csv` row `unclecam-hefley` and `text/unclecam-hefley.REQUEST.md`.
- Svelmoe, *A New Vision for Missions* (2008) — located via Gale Academic OneFile (paywalled
  database); not fetched this session. See `sources.csv` row `svelmoe-new-vision`.

## Bibliography mining

The Encyclopedia of Arkansas entry on Townsend (`eoa-townsend`) names, in its own
bibliography, two further Hugh Steven-authored memoir volumes not on the roster's original
list: *Doorway to the World: The Mexico Years* (1999) and *Wycliffe in the Making: The Memoirs
of W. Cameron Townsend, 1920–1933* (1995), both published by Harold Shaw, Wheaton, IL. Neither
was fetched this session; recorded here as `NEW: Doorway to the World` and `NEW: Wycliffe in
the Making` for a follow-up Scout/Fetcher ticket, per Procedure S step 4.

## Institution-specific findings

Two independently fetched sources disagree on whether SIL's own **incorporation** (as
distinct from the 1934 Camp Wycliffe founding) took place in 1934 or 1942 — see
`discrepancies.md` #1. The Jim Elliot/Camp Wycliffe/SIL connection at "Norman, Oklahoma" named
in the roster is corroborated by two fetched sources, but the year is contested (1950 vs.
1948) and could not be resolved this session — see `discrepancies.md` #2. No pre-1900 or
other special nickname-origin step applies to this institution (that special step is unique to
`geneva-academy`).
