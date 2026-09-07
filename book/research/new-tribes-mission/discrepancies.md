STATUS: DONE
TICKET: E-new-tribes-mission
ROLE: Extractor
INPUTS READ: research/new-tribes-mission/quotes.jsonl; research/new-tribes-mission/text/*.txt

# Discrepancies — new-tribes-mission

| # | Field | Source A (id, page) | Value A | Source B (id, page) | Value B |
|---|---|---|---|---|---|
| 1 | S3 (current worldwide missionary count) | ethnosca-heritage, p. 1 (context near quote_id ntm-q014) | "Today more than 2,000 missionaries serve throughout the world" | ntmuk-heritage, p. 1 (context) and ethnos360-about, p. 1 (quote_id ntm-q048) | "Today more than 3,000 missionaries serve throughout the world" / "more than 3,000 missionaries." Both figures are presented as the present-day ("today") count rather than a founding-era number; the difference most likely reflects the pages having been published or last updated in different years, not a factual conflict about 1942. Not resolved; both readings recorded rather than averaged or silently preferring one. |
| 2 | F4 (site of the founding-era "boot camp" training facility) | wiki-ntm, p. 1 (quote_id ntm-q023, context) | "Shortly thereafter [after the 1944/45 move to Chico, California] it established a 'boot camp' (missionary training facility) at Fouts Springs, California" | roster row (`02_institution_roster.md`) | The roster names "Fredonia, Wis." as the boot-camp location. No source fetched this session corroborates a Fredonia, Wisconsin boot camp; the only located and dated founding-era boot camp is at Fouts Springs, California, and the only located Wisconsin NTM-family training site (Waukesha, home of Ethnos360 Bible Institute) is dated to 1955, a different town and a later, differently described facility. This session does not resolve whether the roster's "Fredonia, Wis." is a separate, undocumented site, an error, or a later relocation not captured by the sources fetched; recorded as NOT FOUND rather than silently omitted or silently affirmed. |
| 3 | F2/F3 (exact founding date and founder list) | e360-founding, p. 1 (quote_id ntm-q007) and e360-founded-1942, p. 1 (quote_id ntm-q010, dropped as a duplicate of ntm-q007's wording — see note below) | Both name only Paul Fleming and Cecil A. Dye as the two men who "felt led to form a committee" in 1942, with "God provid[ing] men to join their team" | ethnos360-namechange, p. 1 (quote_id ntm-q049) | "New Tribes Mission was founded in 1942 by American missionaries Paul Fleming, Cecil A. Dye, Lance B. Latham and M. Robert Williams" — names two additional founders (Latham, Williams) not mentioned by the Ethnos360 Bible Institute blog posts. The `housley-founding-fathers` source independently corroborates Williams as a signatory of the 1942 Pledge and Covenant ("Cecil Dye, Paul Fleming and Bob Williams"), but does not mention Latham in that context, though `ethnosca-heritage` separately names "Lance Latham" as a participant in the 1947 Toronto conference. This session does not resolve whether the founding "committee" numbered two, three, or four; all readings are recorded. |

**Note:** quote_id `ntm-q010` was drafted from `e360-founded-1942` but its wording proved
identical (within normalization) to `ntm-q007` from `e360-founding`; the qmake extraction
script could not locate that exact wording in `e360-founded-1942`'s own text, so the record
was dropped rather than forced. `ntm-q007` alone is cited for that fact.

No other discrepancies were found among the sources fetched this session.
