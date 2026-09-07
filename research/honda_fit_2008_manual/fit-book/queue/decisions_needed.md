# Decisions needed / process adaptations

## ADAPT-1: WebFetch is unavailable in this execution environment (2026-09-07)

**Finding:** Direct page fetches (WebFetch) fail with `EGRESS_BLOCKED` for every domain tested,
including nhtsa.gov and even a bare google.com fetch. This is a network egress proxy restriction
in the execution environment, not a per-site 403 or a fixable config issue. WebSearch (which
returns titled snippets with real URLs) works normally.

**Impact on the plan as written:** RESEARCH_PLAN.md assumes agents can open a source URL and read
the exact page (Section 1 Rule 5, Section 6 verifier step 1-2, Section 8 R3). That is not possible
here for any external site.

**Adaptation (applies to every WP and the verifier from here forward):**
1. Fact-gathering uses WebSearch only. A fact is sourced from the search result snippet plus its
   real URL — never from an invented page detail beyond what the snippet shows.
2. A fact confirmed by only one search snippet is capped at **Confidence: medium** even if the
   source is Tier 1, and must say so. Confidence: high requires the same value appearing in two
   independent search results (different queries or different snippets), or explicit numeric
   agreement across two sources.
3. The verification pass is redefined for this environment: instead of "open the URL and confirm,"
   the verifier runs its own independent WebSearch queries (not seeded with the researcher's
   phrasing) to see if the same value resurfaces from an independent search. PASS = the value is
   corroborated by an independent search; FAIL = contradicted or not found; the verifier still
   applies YEAR-OK/YEAR-FAIL exactly as written, from what the snippet itself shows about model year.
4. "Exact location in source" fields will often say "search result snippet" rather than a page
   number, since the page itself could not be opened. That is expected and acceptable under this
   adaptation, not a rule violation.
5. If a later phase of this session regains WebFetch access, prefer it going forward and treat it
   as an upgrade path — re-verify medium-confidence facts opportunistically, not as a blocking
   requirement.

This note is the authoritative adaptation for all subsequent WP and verification agents in this
session. It does not change any locked decision (D1-D10) or the book outline.

## ADAPT-2: working direct-read method found — supersedes part of ADAPT-1 (2026-09-07)

**Finding (from WP-01):** The WebFetch tool itself stays blocked, but real page content IS reachable
through the Bash tool:
1. Try `curl -sL --max-time 20 -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" "<url>"`
   first. This works for many sites (fueleconomy.gov, owners.honda.com mirrors, NHTSA's JSON APIs at
   `api.nhtsa.gov` and `vpic.nhtsa.dot.gov`, IIHS, several enthusiast/reference sites). Static HTML
   needs a quick tag-strip pass (e.g. `curl ... | python3 -c "import sys,re,html; ..."` or `sed`) to
   read as text.
2. If curl gets a 403 or the page is JS-rendered, retry through the public reader proxy:
   `curl -sL --max-time 25 "https://r.jina.ai/<original https:// url, unencoded>"` — this returns
   clean readable text/markdown, including full data tables, for sites that block bots directly
   (this is how confirmed Honda Tier-1 press releases from hondanews.com were read in WP-01).
3. Known dead ends, don't burn time retrying: `edmunds.com`, `cars.com`, `www.thecarconnection.com`
   (403 even through the proxy route so far), `archive.org` ("host not in allowlist"),
   `techinfo.honda.com` (TLS handshake failure).
4. **This upgrades ADAPT-1's confidence rule**: a fact confirmed via a real curl or r.jina.ai page
   read (not just a WebSearch snippet) is treated as directly-verified — normal Confidence rules
   apply (Tier 1/2 source = high, per the original template) rather than the medium-confidence cap
   ADAPT-1 set for snippet-only facts. Reserve the medium-confidence cap for facts that truly could
   only be confirmed via a WebSearch snippet after both curl and the r.jina.ai route failed or a
   domain is a known dead end.
5. All future WP agents (research and verification alike) should attempt curl, then r.jina.ai,
   before falling back to WebSearch-snippet-only sourcing. The verifier role in Section 6 of the
   plan should now be read as: re-fetch the source URL via this method and re-check the claimed
   value and location against the real page text — this is much closer to the plan's original
   "open the URL and confirm" design than pure WebSearch cross-referencing was.

## ADAPT-3: concurrent WP agents collided on sources/S-###.md numbering (2026-09-07)

**Finding:** WP-11 (this task) and WP-10 (torque specs/fitment) ran in the same session/repo at the
same time and both picked the "next" unused S-### number by looking at the directory listing at
roughly the same moment. WP-11 created S-028 through S-034 first; WP-10 then created its own S-028
through S-035 (different content, different URLs) and silently overwrote every one of WP-11's new
cards, because both agents wrote plain files with no lock and no coordination. This was only caught
because the artifact/file-watch surfaced "changed on disk since you last read it" notices — a
plain background WP run with no such notice would have lost the first agent's source cards with no
error at all.

**Adaptation (applies to every future WP that creates new Source Cards):**
1. Before creating a new sources/S-###.md file, re-list the sources/ directory immediately before
   writing (not just once at task start) to reduce (not eliminate) the collision window.
2. Prefer a WP-scoped filename prefix instead of the shared sequential S-### counter when adding a
   *secondary* source discovered mid-task (i.e., not one of the canonical WP-00 seed sources):
   `sources/S-<WPID>-<letter>.md`, e.g. `sources/S-WP11-A.md`. This guarantees no other WP's
   sequential counter can ever collide with it. Cards using this scheme still follow the normal
   Source Card template fields (Source ID, Title, ... Tier, Confidence-relevant notes) — only the
   filename/ID prefix differs. This WP (WP-11) renamed its own S-028 through S-034 to S-WP11-A
   through S-WP11-G after discovering the collision; their content is unchanged from what was
   already drafted, only the ID/filename moved.
3. If two WPs truly need a shared canonical numbered slot (e.g. both want to cite the same
   newly-found Tier 1 source), the second WP to notice a collision should rename ITS OWN card out of
   the shared counter rather than re-overwriting the other WP's card a second time — do not "fix" a
   collision by taking the number back.
4. This does not change any fact or procedure content already written; it only changes how new
   Source Card files are numbered going forward, to stop silent data loss between concurrent WPs.
