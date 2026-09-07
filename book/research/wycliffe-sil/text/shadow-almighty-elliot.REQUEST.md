STATUS: BLOCKED: print-only equivalent (controlled digital lending; not accessible to this session)
TICKET: F-wycliffe-sil
ROLE: Fetcher
INPUTS READ: sources.csv row `shadow-almighty-elliot`

# Fetch request — Elisabeth Elliot, *Shadow of the Almighty* (1958; 1989 HarperCollins ed.)

**Archive:** Internet Archive, item `shadowofalmighty00elli`.
**URL:** https://archive.org/details/shadowofalmighty00elli

**What was tried:** `curl -sS -L --max-time 60 "https://archive.org/download/shadowofalmighty00elli/shadowofalmighty00elli_djvu.txt"` returned `HTTP 401 Authorization Required` (controlled digital lending; same pattern as `unclecam-hefley`).

**Why this document matters:** Wikipedia's Jim Elliot article cites this book, pp. 128–32, as the source for "In the summer of 1950, while at Camp Wycliffe (Cameron Townsend's linguistics training camp in Oklahoma), Elliot practiced the skills necessary for writing down a language..." — i.e., this is very likely the primary published source (Elisabeth Elliot's own biography of her husband) that would let a future Fetcher confirm the exact 1950/Norman/Oklahoma detail against Jim Elliot's own record, and resolve the discrepancy against the "graduation from Wheaton in 1948" wording found in `eef-journals-1948.txt` (see `discrepancies.md`).

**Request procedure:** same as `unclecam-hefley.REQUEST.md` — requires an authenticated archive.org borrow. This is a strong priority follow-up: it is the single source most likely to resolve the roster's `[VERIFY]` flag on the Jim Elliot/SIL/1950/Norman Oklahoma claim.

**Rights note:** *Shadow of the Almighty* is in copyright (1958 orig., successive editions); once fetched, only short quotations should be extracted, flagged `[RIGHTS: copyrighted, short quotation only]` and logged to `rights/permissions_log.csv`, per the pattern already used for *The Journals of Jim Elliot* in `wheaton-college` and in this dossier.
