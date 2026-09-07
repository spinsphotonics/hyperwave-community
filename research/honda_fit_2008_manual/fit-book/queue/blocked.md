# Blocked items

## WP-00 Build the source library (2026-09-07)

### S-002 Helm Inc. factory service manual (2007-2008 Fit) — paywalled
Helm Inc. sells the official Honda factory service manual for the 2007-2008 Fit (KA) as a paid,
phone-ordered print product (~5 week delivery); no free online access was found.
Per RESEARCH_PLAN.md Section 3, using Haynes manual 42030 (S-003) as the designated fallback for
repair-procedure content, with the explicit caution (per D8) that the Haynes manual spans both the
2007-2008 GD3 and 2009-2013 GE8 generations and every page must be checked for which car it describes.

### S-003 Haynes manual 42030 — paywalled
Aftermarket repair manual; must be purchased (print or retailer digital edition) to read. No free
full-text equivalent found. Bibliographic facts (title, ISBN-13 9781620921425, coverage years) were
confirmed via WebSearch/retailer listings only.

### Tool limitation: WebFetch blocked for all domains this session
Every WebFetch call this session returned EGRESS_BLOCKED (or "unable to fetch"), including for
owners.honda.com, helminc.com, nhtsa.gov, fueleconomy.gov, iihs.org, fitfreak.net, hondapartsnow.com,
en.wikipedia.org, reddit.com, and even google.com. This is not a per-domain organization policy 403 —
even google.com failed — so it looks like a session-wide WebFetch restriction rather than a fixable
TLS/config issue (checked `curl $HTTPS_PROXY/__agentproxy/status`, which shows the Bash-level egress
proxy is healthy and unrelated; WebFetch is a separate server-side tool). All 20 Source Cards in
sources/ were therefore verified via live WebSearch queries and their returned result snippets
(real URLs, not invented) rather than by directly loading the page. Future WPs that need exact
quoted text, page numbers, or table values from these sources should retry WebFetch first in case
the restriction is lifted; if still blocked, flag any fact that cannot be confirmed beyond a search
snippet as UNVERIFIED per RESEARCH_PLAN.md Rule 2.

## WP-02 Recalls, TSBs, and known defects (2026-09-07)

### Tool limitation: WebFetch still blocked for all domains this session
Confirmed the same session-wide WebFetch restriction persists (tested nhtsa.gov, static.nhtsa.gov,
carcomplaints.com, fitfreak.net, reddit.com, en.wikipedia.org — all EGRESS_BLOCKED or "unable to
fetch"). All 26 fact cards in facts/WP-02.md were built from WebSearch result snippets and AI-search
summaries only, with real source URLs attached. Before facts/WP-02.md is marked `verified` in the
ledger, the verifier task (RESEARCH_PLAN.md Section 6) MUST attempt a direct read of every Source URL
in that file — several facts (recall exact dates, TSB model-year coverage) had conflicting snippet
summaries that only a direct primary-source read can resolve. See queue/conflicts.md for the specific
conflicts already identified during WP-02.

### Facts logged as leads only, not verified (WP-02)
F-WP02-010 lists four TSB numbers (A20-015, A18-053, A18-050, A02-053) surfaced by search snippets on
2008-Fit TSB aggregator pages, but with no independently confirmed detail. Do not use these in draft
chapters until a future pass with working WebFetch confirms each bulletin's content and model-year
applicability.
