STATUS: BLOCKED: print-only equivalent (controlled digital lending; not accessible to this session)
TICKET: F-wycliffe-sil
ROLE: Fetcher
INPUTS READ: sources.csv row `unclecam-hefley`

# Fetch request — Hefley and Hefley, *Uncle Cam* (1974)

**Archive:** Internet Archive, item `unclecamstoryofw0000hefl`.
**URL:** https://archive.org/details/unclecamstoryofw0000hefl
**Shelfmark/identifier:** unclecamstoryofw0000hefl (272 pages; also listed under identifier `unclecam0000jame`, per Wikipedia's citation, possibly a duplicate scan — not confirmed this session).

**What was tried:** `curl -sS -L --max-time 60 "https://archive.org/download/unclecamstoryofw0000hefl/unclecamstoryofw0000hefl_djvu.txt"` returned `HTTP 401 Authorization Required`. `archive.org/metadata/unclecamstoryofw0000hefl` confirms `"access-restricted-item": true` — this is a controlled-digital-lending title; the OCR text file exists (`unclecamstoryofw0000hefl_djvu.txt`) but is served only to a logged-in patron with an active one-hour loan.

**Request procedure:** A human (or an authenticated session) must (1) create/sign in to an archive.org account, (2) go to the item page above and click "Borrow," (3) use the in-browser BookReader to read or the loan session to access OCR text, respecting the one-book-at-a-time lending limit. Because this is a 1974 in-copyright biography, only short quotations should be copied out even once borrowed, per the rights rules in `01_book_design.md`/`06_rights_and_permissions.md`.

**Specific pages/sections needed** (per roster "Key primary documents" and the assignment's named priority source): the 1934 Camp Wycliffe founding chapter (early chapters, exact page numbers unknown — likely within the first 60–80 pages given a 1917–1934 chronology) and any chapter describing the SIL/WBT 1942 doctrinal statement or incorporation.

**Alternate copy found but not pursued:** Google Books listing (id `VAi54ZsE8eQC`) — snippet/preview view only for in-copyright books; not fetched via curl this session (see `sources.csv` row `hefley-uncle-cam-print`).
