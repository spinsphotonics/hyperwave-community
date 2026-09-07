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
