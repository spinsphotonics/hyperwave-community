STATUS: DONE
TICKET: E-biola
ROLE: Extractor

# Discrepancies — biola

1. **First dean named at 1908 founding.** `biola-hope-street-1985` states that at the Institute's
   February 1908 founding, "Dr. W.E. Blackstone, a former Methodist pastor and author of the
   popular *Jesus Is Coming*, served as dean" (quote_id biola-q003 context). The same secondary
   source's own later narrative and timeline state that R. A. Torrey did not become dean until
   1911-1912 ("In the summer of 1911, Torrey was invited to be the new dean"; timeline: "1912. R.A.
   Torrey becomes dean of the Bible Institute of Los Angeles"). Both are from the same secondary
   source and are not in tension as dates (Blackstone first, Torrey following in 1912), but the
   roster's own "Why it is in the book" cell names only Torrey's institute type; no primary source
   read this session independently confirms Blackstone's 1908 deanship, so it is recorded here as
   `[SECONDARY: biola-hope-street-1985]` only, not stated as `[HIGH]`.

2. **Access blocked to the most likely primary founding document.** The Biola History Wiki
   (confluence.biola.edu) advertises a page titled "Articles of Incorporation" that (per its own
   search-result summary) purports to reproduce "the full text of the document including details
   about the founding directors and the institution's charter" and "a Statement of Belief." This
   session's fetch tooling could not retrieve it: the server's TLS certificate has expired
   (`curl: (60) SSL certificate problem: certificate has expired`), and this session is instructed
   never to disable TLS verification to work around such errors. This is recorded as `BLOCKED`, not
   silently skipped; see sources.csv row NONE-1.

3. **HathiTrust's 1920 BIOLA catalog, reported by web search as "full view," could not be fetched.**
   Every attempt (direct URL, with and without a browser User-Agent string) returned HTTP 403 from
   `babel.hathitrust.org`. This may be an anti-scraping block on this session's network egress
   rather than a genuine access restriction on the item, but the effect for this session is the
   same: the document could not be read. See sources.csv row NONE-2.

4. **The King's Business and the Academic Catalogs series are digitized (per Biola's own library
   page) but sit behind a JSTOR "community site" that renders via JavaScript**, which this
   session's `curl`-based fetch tooling cannot execute. The app shell was retrieved (HTTP 200) but
   carried no article or catalog text. See sources.csv rows NONE-3 and NONE-4.
