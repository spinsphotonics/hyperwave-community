# 07 — Risk Register, Known Confusions, and QA Gates

## A. Hallucination traps (the Verifier checks each of these explicitly)

| Trap | What a cheap model tends to do | Control |
|---|---|---|
| Invented quotations | Produces a plausible sentence "from" Calvin, Spurgeon, Taylor, Lloyd-Jones | Rule 2; `check_quotes.py`; V1 and V2 quote-by-quote check |
| Invented document titles | Cites a "Constitution of 1856" that never existed | Scout must log a repository hit or NONE for every title; Extractor may cite only `document_id`s in `sources.csv` |
| Date drift | Rounds or shifts founding years; conflates institution founding with building opening | Known-facts sheet; Discrepancies file; two sources for every date |
| Merging institutions | Treats CIM and OMF, Pastors' College and Spurgeon's College, Andover and Andover Newton, Nyack and C&MA as unrelated or identical without saying so | Roster fixes the relationships; Editor checks |
| Attribution of nickname | Says "Calvin called it the school of death" | Nickname-origin search; do not attribute without primary citation |
| Modern paraphrase in old quotation | "Modernizes" a 1559 or 1642 text inside quotation marks | Abridgement rule 7; Verifier compares to `text/` |
| Secondary-as-primary | Quotes Murray's biography as if it were Lloyd-Jones's own words | `is_primary` column; PRIMARY-ONLY fields |
| Numbers without year | "The College sent 900 men" with no date or source | Every number needs a quote record with year |
| Wikipedia citations | Cites Wikipedia as a source | Wikipedia may be used only to find primary sources; it is never cited |
| Translation from translation | Translates a French summary of a Latin statute | Abridgement rule 6 |
| Confident filling of blanks | Writes a curriculum "typical of the period" when the source is silent | NOT FOUND markers are required; Editor inserts "the sources are silent" |

## B. Known confusions (names and institutions)

| Term in brief or sources | Correct referent | Not to be confused with |
|---|---|---|
| "Calvin School of Death" | Geneva Academy, 1559 | Any school named "Calvin" (Calvin University, Grand Rapids; Calvin Theological Seminary) |
| "Spurgeon School" | Pastors' College, London, 1856 (Spurgeon's College from 1923) | Spurgeon's orphanage (Stockwell Orphanage) |
| "Martin Lloyd Jones' school" | London Theological Seminary, 1977; informal training via Westminster Chapel/Fellowship | London Bible College (1943; now London School of Theology), which Lloyd-Jones declined to join |
| "Adam and I from Judson" | Adoniram Judson (1788–1850) | Judson College (Alabama, 1838); Edward Judson (son) |
| "Elizabeth Elliott" | Elisabeth Elliot, née Howard (1926–2015) | John Eliot (1604–1690), missionary to the Massachusett; Elizabeth Elliott (other persons) |
| "Hudson Taylor" | James Hudson Taylor (1832–1905) | J. Hudson Taylor II/III (descendants, OMF) |
| Princeton | College of New Jersey (1746) vs Princeton Theological Seminary (1812) | Two separate institutions; two roster rows |
| Andover | Andover Theological Seminary (1808) | Andover Newton (merger 1965; now at Yale) |
| Wycliffe | Wycliffe Bible Translators (1942) / SIL (1934) | Wycliffe Hall, Oxford (1877); John Wycliffe (d. 1384); Wycliffe College, Toronto |
| Westminster | Westminster Theological Seminary (Philadelphia, 1929) | Westminster Chapel (London); Westminster College (Cambridge); Westminster Assembly (1643) |
| Moody | Moody Bible Institute (1886/89) | Moody Church; Northfield and Mount Hermon schools (also Moody's, and relevant to SVM) |
| CIM / OMF | Same organization, renamed 1964/65 | — |
| Nyack / C&MA | Missionary Training Institute (1882) is the school; the Christian and Missionary Alliance (1887) is the denomination/mission | Nyack College (later name) |
| Prophezei | Zurich daily exegetical exercise, 1525 | English Puritan "prophesyings" (1570s), a descendant but different |
| Log College | William Tennent's academy at Neshaminy, c.1726 | Princeton, which claims it as antecedent but is a separate foundation |
| Bristol | Bristol Baptist Academy/College (Baptist) | Bristol Tabernacle; Bristol Dissenting academies of other denominations |
| Basel Mission | Evangelical Missionary Society in Basel (1815) | Basel Christian Church of Malaysia (a fruit); Basel University |
| Herrnhut | Moravian (Unitas Fratrum) settlement, 1722 | Herrnhaag; Bethlehem, Pa. (daughter settlement) |
| "Brethren" | Plymouth/Christian Brethren (1820s–) for the Elliots; also "Society of the Brethren" (Andover/Williams student society, 1808) | The two are unrelated; use full names |

## C. QA gates

| Gate | When | Who | Pass criterion |
|---|---|---|---|
| G1 Scout complete | after S | Coordinator | Every target has a row; no blanks |
| G2 Transcription quality | after F | Coordinator | OCR error rate ≤5/100 words on sample |
| G3 Dossier verified | after V1 | Verifier | Zero FAILs |
| G4 Charter fidelity | after B | Verifier (as part of V2) | Every kept passage verbatim; cuts marked; word count in range |
| G5 Chapter verified | after V2 | Verifier | Every sentence supported; every footnote resolves |
| G6 Style | after ED | Editor | Zero forbidden words; names and dates match roster; sections in order |
| G7 Rights | before F-001 | Human | Every reprinted document has PD confirmation or permission on file |
| G8 Synthesis consistency | after X-001 | Editor | Every chapter figure matches the tables |
| G9 Random audit | F-003 | Verifier | 5% sample, zero FAILs (any FAIL reopens chapter) |
| G10 Final acceptance | F-004 | Human | Checklist below |

## D. Final acceptance checklist (F-004)

- [ ] Every Tier A institution has an ACCEPTED chapter.
- [ ] Every chapter has the ten sections in order.
- [ ] Every chapter's Charter section is verbatim from a `text/` file, with cuts marked, and its rights status is logged.
- [ ] Every quotation in the manuscript exists in `quotes.jsonl` and passes `check_quotes.py`.
- [ ] Every date in the manuscript matches the known-facts sheet or has a resolved discrepancy entry.
- [ ] The Geneva "school of death" nickname is presented with its earliest found source or explicitly as of unknown origin.
- [ ] No sentence attributes a quotation to Calvin, Spurgeon, Taylor, Judson, Lloyd-Jones, or the Elliots without a primary citation.
- [ ] All Lloyd-Jones and Elliot quotations are within the fair-use limit or covered by permission.
- [ ] Comparative tables, lineage chart, timeline, glossary present and cited.
- [ ] Forbidden-word count is zero.
- [ ] Bibliography lists every `sources.csv` primary row used.

## E. Escalation rules

1. Any `BLOCKED` ticket older than 7 days goes to the Human with the scout notes attached.
2. Any discrepancy that affects a chapter's Founding section is resolved by the Human before D starts.
3. Any document whose only copy is in a physical archive is requested by the Human; the pipeline for that institution continues with the other documents and the chapter is marked "pending archive copy."
4. Any translation marked `[New translation, draft]` is reviewed by a reader of the language before V2.
5. Any proposed addition to the roster is logged, not acted on, until the Coordinator approves.
