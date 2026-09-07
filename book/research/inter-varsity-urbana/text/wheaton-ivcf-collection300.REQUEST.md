STATUS: BLOCKED: print-only/access-gated, request filed
TICKET: F-inter-varsity-urbana
ROLE: Fetcher
INPUTS READ: sources.csv row wheaton-ivcf-collection300

# Fetch request — Billy Graham Center Archives, Collection 300 (IVCF Records)

**Archive:** Archives of Wheaton College / Billy Graham Center, Wheaton, Illinois.
**Finding aid URL:** https://archives.wheaton.edu/repositories/4/resources/1225
**What happened:** An unauthenticated `curl` fetch of this URL returned a Cloudflare
Turnstile "Human Check" interstitial page (title "Human Check", form posting to
`/_turnstile/verify`) instead of the finding aid content. This matches the block already
recorded against the same archive by the `wycliffe-sil` and `wheaton-college` chapters'
research sessions.
**Pages/sections needed:** Per `03_research_protocol.md` Procedure S step 5 (special step
for institutions connected to the Elliot papers): the IVCF records finding aid description
and box list, specifically any correspondence or reports from 1941-1950 concerning IVCF's
founding, the 1941 or later doctrinal basis text, and the 1946 Toronto / 1948 Urbana
convention programs.
**Request procedure:** Archives of Wheaton College reading-room/reproduction request via
their contact page (linked from https://www.wheaton.edu/about-wheaton/museums-library-and-collections/wheaton-archives-and-special-collections/about/evangelism--missions-archives/).
**Follow-up:** A human researcher with authenticated/in-person access should retry this
finding aid and, if it names a digitized 1941 doctrinal basis or Urbana 1946/1948 program,
file a new Fetcher ticket.
