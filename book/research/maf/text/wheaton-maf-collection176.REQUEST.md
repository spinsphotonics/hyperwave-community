STATUS: BLOCKED: print-only/access-gated, request filed
TICKET: F-maf
ROLE: Fetcher
INPUTS READ: sources.csv row wheaton-maf-collection176

# Fetch request — Billy Graham Center Archives, Collection 176 (MAF Records)

**Archive:** Archives of Wheaton College / Billy Graham Center, Wheaton, Illinois.
**Finding aid URL:** https://archives.wheaton.edu/repositories/4/resources/176
**What happened:** An unauthenticated `curl` fetch of this URL returned a Cloudflare
Turnstile "Human Check" interstitial page instead of the finding aid content, matching the
block already recorded against the same archive by the `wycliffe-sil`, `wheaton-college`, and
`inter-varsity-urbana` chapters' research sessions.
**Pages/sections needed:** The 1945 CAMF incorporation papers or founding statement of
purpose, and any founding-era doctrinal statement, if held in Collection 136/176 (the From
the Vault blog post fetched this session cites "Collection 136: Records of Mission Aviation
Fellowship," folders 1-9, 1-92, 47-55, and 148-9, among others — the full box list was not
visible behind the Cloudflare gate).
**Request procedure:** Archives of Wheaton College reading-room/reproduction request via
their contact page.
**Follow-up:** A human researcher with authenticated/in-person access should retry this
finding aid and, if it names a digitized 1945 founding document, file a new Fetcher ticket.
