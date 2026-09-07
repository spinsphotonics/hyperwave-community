# The Reformed School — working plan

**Working title:** *The Reformed School: Founding Texts of Reformed Education, from Luther to Machen*

**Reader:** a longtime member of a Reformed church beginning to homeschool. Not a scholar. The
apparatus is a technical introduction to the subject: enough context to read each text with
understanding, and pointers to what in it still bears on teaching children today.

**Format:** 6 × 9 in., 100–200 pages (target ~55,000–65,000 words all in), LaTeX (memoir class,
LuaLaTeX, EB Garamond), a book worthy of a nice printing.

**Editorial rules**
- English only. Direct sources. Existing translations only, and only public-domain ones
  (published before 1931 or otherwise clearly out of copyright). No new translations.
- Print short documents whole. Abridge long ones with every omission marked by a centred
  ellipsis on its own line. Never reorder.
- Spelling: as printed for texts after 1700. For Knox (1560) and *New England's First Fruits*
  (1643): modernise spelling and u/v, i/j, long-s forms only; never change a word. State this in
  the headnote.
- Every text is proofread against the page images of its source edition, not merely the OCR.
  Uncertain readings are listed in `notes/<slug>-proof.md`.
- General introduction (~4,000 words) and a headnote before each text (300–600 words): who,
  when, what the document is, which edition we print, what to notice, and a few glossed words.
- Afterword on Martyn Lloyd-Jones and the London Theological Seminary (1977): discussed, briefly
  quoted, not reprinted (in copyright).

**The texts (order of the book)**

| # | Text | Edition printed | Target words |
|---|------|-----------------|--------------|
| 1 | Luther, *Letter to the Mayors and Aldermen of All the Cities of Germany in Behalf of Christian Schools* (1524) | Painter, *Luther on Education* (1889) | whole, ~8,500 |
| 2 | Luther, *Sermon on the Duty of Sending Children to School* (1530), selection | Painter (1889) | ~2,500 |
| 3 | Calvin, *Ecclesiastical Ordinances* (1541): the order of doctors and the school | PD English translation (to be located) | whole section, ~1,500 |
| 4 | Beza, inaugural address at the Geneva Academy (5 June 1559) | PD English (to be located) | as available |
| 5 | *First Book of Discipline* (1560): "For the Schools" and "The Erection of the Universities" | Laing, *Works of John Knox* II (1848), spelling modernised | ~5,000 |
| 6 | *New England's First Fruits* (1643): the college, the Rules and Precepts, the course of study | 1643 text as reprinted in Miller & Johnson / Quincy | whole, ~1,500 |
| 7 | Doddridge, letter describing the Northampton academy (1732) | Boyd, *Memoir of Doddridge* (1860) | whole, ~1,100 |
| 8 | The Log College: Whitefield's journal (Nov. 1739) and Alexander's account (1845) | Whitefield, *Fifth Journal*; Alexander, *Log College* | ~1,000 |
| 9 | Charter of the College of New Jersey (1748) | Maclean, *History of the College of New Jersey* (1877) | abridged, ~2,500 |
| 10 | Carey, *An Enquiry into the Obligations of Christians to Use Means* (1792), Section V | 1792 text | ~4,500 |
| 11 | The Serampore *Form of Agreement* (1805) | *Periodical Accounts* / Marshman | whole, ~2,300 |
| 12 | Andover, *Constitution and Associate Statutes* with the *Associate Creed* (1808) | Woods, *History of Andover* (1885) | whole, ~2,500 |
| 13 | *Plan of the Theological Seminary of the Presbyterian Church* (1811) | 1811 printing | abridged, ~6,000 |
| 14 | Boyce, *Three Changes in Theological Institutions* (1856); *Abstract of Principles* (1858) | 1856 printing / SBTS | ~4,000 + 1,200 |
| 15 | Spurgeon, on the founding of the Pastors' College | *Autobiography* II (1897–1900) | whole, ~1,600 |
| 16 | Machen, *Westminster Theological Seminary: Its Purpose and Plan* (1929) | *The Presbyterian*, 10 Oct. 1929 | whole, ~4,000 |
| — | Afterword: Lloyd-Jones and the London Theological Seminary (1977) | discussed only | ~2,000 |

**Directory**
- `sources/<slug>/` — fetched editions, page images, OCR, README with provenance
- `texts/<nn>-<slug>.md` — proofread reading texts, with cuts marked
- `notes/<slug>-proof.md` — uncertain readings and editorial decisions
- `tex/` — LaTeX sources; `tex/build/` output

**Stages**
1. Acquire the missing English texts (Calvin, Beza, Machen). — in progress
2. Proofread every held text against page images; produce reading texts.
3. Write headnotes, introduction, afterword.
4. Typeset in LaTeX; proof the PDF page by page; fix widows, orphans, breaks.
5. Deliver PDF.
