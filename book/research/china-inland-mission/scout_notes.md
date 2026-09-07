STATUS: DONE
TICKET: S-china-inland-mission
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (row: china-inland-mission); book/plan/05_repositories_and_search_strings.md (section: china-inland-mission); book/plan/04_task_tickets.md (E-china-inland-mission extra-step row)

# Scout notes — China Inland Mission (`china-inland-mission`)

## Target documents (from roster "Key primary documents" cell)
1. Taylor, *China's Spiritual Need and Claims* (1865)
2. CIM *Principles and Practice* (earliest printed edition; date [VERIFY], revised editions 1885-1900s)
3. *Arrangements of the China Inland Mission* (1886) [VERIFY]
4. *China's Millions* (1875-)
5. Taylor's letters

## Searches run and results

1. `"China's Spiritual Need and Claims" Hudson Taylor 1865 archive.org full text` (WebSearch): found multiple
   digitized editions on archive.org under different titles/dates. Fetched two:
   - `b29353518` — *China: Its Spiritual Need and Claims; With Brief Notices of Missionary Effort, Past and
     Present*, Third Edition (London: James Nisbet & Co., 1868). Its own "Preface to the Third Edition" states
     the pamphlet was originally prepared to represent matters as of March 1865.
   - `cu31924023067972` — *China's Spiritual Need and Claims*, Seventh Edition (London: Morgan & Scott, 1887).
     Its "Prefatory Note to the Seventh Edition" (by Taylor, March 1887) gives the full edition history: written
     1865, 2nd ed. 1866, 3rd ed. 1868, 4th ed. 1872, 5th/6th ed. June 1884 (5000 copies each), 7th ed. (10,000
     copies) March 1887.
   - A third identifier, `pts_chinasspiritualn_3720-1090` (microform, catalogued 1884), was located and its
     metadata fetched but its djvu text was NOT fetched this session (time-boxed; the 1868 and 1887 editions
     already fetched bracket it and were judged sufficient). Logged as a NONE-equivalent row (full text not
     retrieved) in sources.csv for the record.
   - The original 1865 first edition itself was not located as a separately digitized full text this session
     (only later editions, 1868 onward, were found and fetched). This is noted in discrepancies.md: quotations
     below are from the 1868 (3rd) and 1887 (7th) editions, not the 1865 first edition, though the 1868 preface
     and 1887 prefatory note both independently corroborate an 1865 origin.

2. `"Principles and Practice of the China Inland Mission" archive.org` (WebSearch): found `principlespracti00chin_0`
   (Columbia University Libraries copy, catalogued by Internet Archive/Columbia as "1900" — this date is a
   library catalog assignment; no date is printed on the fetched pages themselves, which are an undated pamphlet,
   cover title only). Fetched and read in full (18 pages, 15 numbered articles plus a blank candidate-signature
   form). This is the document used as the chapter's charter (F5).

3. `China Inland Mission "Principles and Practice" edition date first published` (WebSearch), per the
   ticket's extra-step instruction to find at least two dated editions and record title differences:
   - HKBU Special Collections & Archives (Maurice Hutton Family Collection, catalogue entry, fetched as
     `hkbu_pp1905_catalog.html`) lists a physical item titled "**Principles and Practice of The China Inland
     Mission**," dated **1905** (capitalization differs slightly from the archive.org copy's title; catalog
     record only, not a digitized full text — SECONDARY/catalog metadata, not usable for quotation).
   - Wycliffe College Archives (Toronto, fetched as `wycliffe_pp1904_catalog.html`) lists a physical item
     "Principles and Practice of the China Inland Mission item. – **1904**. – 4 pages." (catalog record only,
     not a digitized full text; also notes the document as only 4 pages, versus the archive.org copy's 18
     pages/15 articles — suggesting the 1904 item may be an abbreviated leaflet rather than the same full
     text, but this was NOT verified by reading it — see discrepancies.md).
   - So: three distinct catalogued dates for editions of "Principles and Practice" were found this session —
     archive.org/Columbia's "1900" (full text fetched and used), Wycliffe's "1904" (catalog only, 4pp.,
     not fetched), and HKBU's "1905" (catalog only, not fetched). None of these three dates is a date printed
     on the document itself in any copy actually read this session; all three are library-assigned catalog
     dates. This satisfies the ticket's "find at least two editions with dates" instruction, but full text
     was obtained for only one edition. Flagged as follow-up: request the Wycliffe or HKBU items, or search
     SOAS's CIM archive (Adam Matthew digital guide) for a dated original.

4. `"Arrangements of the China Inland Mission" 1886 archive.org` (WebSearch): NOT found digitized. A search
   result (Adam Matthew Digital's guide to "Minutes and Papers of the China Inland Mission", a PDF finding aid)
   places "The Arrangements of the China Inland Mission" (Shanghai, 1886) on "REEL 60" of the microfilmed CIM
   archive held at SOAS, London. Not accessible as full text this session. Logged as print/microfilm-only in
   sources.csv with the SOAS/Adam Matthew reference as the shelfmark note. Ticket marked BLOCKED for this one
   target document only (see sources.csv row `cim-arrangements-1886`); the rest of the pipeline proceeds using
   *Principles and Practice* as the charter document, per the task instructions' fallback provision.

5. `Lammermuir party 1866 China Inland Mission number of missionaries` (WebSearch, for the Extractor's
   required Lammermuir-numbers step): WebSearch's own summary asserted "18 missionaries plus 4 children" citing
   Wikipedia — NOT used as a source (Wikipedia is never citable per Rule 3). Instead this number was checked
   against two primary/primary-adjacent full texts actually fetched and read:
   - `MN41583ucmf_9` (Taylor's own *Three Decades of the China Inland Mission*, [1895]) states the Lammermuir
     party as "seventeen adults and four children."
   - `jubileechinamis00broouoft` (Broomhall, *The Jubilee Story of the China Inland Mission*, 1915) states the
     "missionary party numbering 22 in all" and gives a full name list (which independently sums to 22: Mr. and
     Mrs. Taylor and their 4 children = 6; Mr. and Mrs. Nicol = 2; five named men; nine named women).
   These two figures (17 adults + 4 children = 21 total per Taylor's own 1895 account, vs. 22 in all per
   Broomhall's 1915 named list) do not agree. Recorded, not resolved, in discrepancies.md. Neither the archive.org
   full-text search nor WebSearch located Taylor's contemporary 1866 shipping list or the original CIM Occasional
   Paper announcement, which would be the strongest primary source for the exact number; flagged as follow-up.

6. `China Inland Mission archive.org Broomhall Hudson Taylor and China's Open Century` — the roster's secondary
   reference. NOT fetched this session (a 7-volume, 20th-century, still-copyrighted work per the roster; not
   expected to be freely full-text digitized, and not needed given the primary/primary-adjacent material already
   in hand). Logged as a NONE row.

7. Additional primary/primary-adjacent documents found and fetched beyond the roster's explicit list, via
   general archive.org search on "China Inland Mission" (both because they were needed to answer the ticket's
   Lammermuir-numbers and mortality-figures instruction, and because they surfaced repeatedly in the results
   above):
   - `martyredmission00broogoog` — Broomhall, comp., *Martyred Missionaries of the China Inland Mission* (London:
     Morgan & Scott, 1901). Contains a signed preface by J. Hudson Taylor himself (Davos, December 1900) and an
     "Editor's Preface" giving the Mission's own count of Boxer-crisis (1900) dead. Used for S5 (casualties).

## Secondary sources
- `jubileechinamis00broouoft` (Broomhall 1915) is formally SECONDARY (a Mission-commissioned history) but, like
  Borgeaud in the geneva-academy dossier, quotes Taylor's own letters and diary at length with quotation marks
  and dates; such quoted passages are treated PRIMARY per the same convention used in that chapter. Broomhall's
  own narrative sentences (e.g., the Lammermuir passenger list, the Boxer casualty analysis of the 1887 cohort)
  are cited [SECONDARY: jubileechinamis00broouoft, page].
- `martyredmission00broogoog`'s Editor's Preface (Broomhall, unsigned but by the compiler) is treated the same
  way: the Mission's own contemporary casualty tally, SECONDARY but close to the event and published by the
  Mission itself.

## Bibliography mining
Not performed this session (time-boxed); no secondary source's own bibliography/notes were separately read for
further primary citations beyond what general archive.org/WebSearch turned up directly. Flagged as follow-up.

## Done-when checklist
- [x] Every target document has at least one sources.csv row (hit, catalog-only hit, or NONE/BLOCKED).
- [x] Every row has all columns filled.
- [x] scout_notes.md lists targets, new targets found (Three Decades, Jubilee Story, Martyred Missionaries), and
      institution-specific findings (Lammermuir numbers, Principles and Practice edition dates).
- [x] No content from any document has been summarized here beyond locating it and the specific institution-
      specific findings the ticket required (edition dates, Lammermuir numbers) — full content extraction is in
      dossier.md.
