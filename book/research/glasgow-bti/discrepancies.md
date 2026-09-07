STATUS: DONE
TICKET: E-glasgow-bti
ROLE: Extractor

# Discrepancies — glasgow-bti

1. **Founder identity: the roster names "John Anderson" as founder; the one background source
   located (Wikipedia's "International Christian College" article) lists John Anderson only as
   Principal from 1898 to 1913 — six years after the Institute's 1892 opening — with no earlier
   principal or founder named for 1892-1898.** No primary source was located this session to
   resolve which is correct, or whether Anderson was involved in the 1892 founding in some other
   capacity before formally becoming Principal in 1898. Both are recorded; neither is asserted as
   settled. This is flagged `[SECONDARY, UNRESOLVED]` rather than silently following the roster.

2. **No primary source of any kind — prospectus, rule-book, annual report, founder's address, or
   institutional periodical — was located and fetched for this institution this session**, despite
   an extensive search recorded in `sources.csv` (Internet Archive full-text search under multiple
   phrasings, HathiTrust, Google Books, and searches for named principals' own writings). This is
   a materially thinner result than every other institution researched this session. Two structural
   obstacles are distinguished from a genuine absence of digitized material: HathiTrust's catalog
   and reader both returned HTTP 403 to this session's tooling (also encountered for the biola
   chapter; likely a network-level block rather than evidence the material doesn't exist there),
   and the Google Books API was rate-limited. By contrast, Internet Archive's full-text index
   itself returned a definitive near-zero result (4 hits for the exact phrase "Bible Training
   Institute" worldwide, none a BTI document), which is not attributable to a tooling block.
