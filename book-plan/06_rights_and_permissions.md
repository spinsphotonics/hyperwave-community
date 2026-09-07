# 06 — Rights and Permissions

## Rules of thumb (United States publication)

- Works published before 1 January 1930 are public domain in the US (as of 2026, the line moves forward one year each 1 January; the Coordinator re-checks the year at ticket P-000).
- Works published 1930–1963 are public domain only if copyright was not renewed. Check the Stanford Copyright Renewal Database and the Catalog of Copyright Entries (HathiTrust). Record the search in `rights/permissions_log.csv`.
- Works published 1964 and later: assume copyrighted. Quote only within fair use (short passages, for commentary) or obtain permission.
- Unpublished archival documents (minute books, letters): copyright can persist; the archive usually sets reproduction terms. Record the archive's terms.
- New translations made for this book are the book's own copyright; translations by others follow the translator's date.
- UK and EU: life plus 70 years. For authors who died after 1955 (Lloyd-Jones d. 1981, Elisabeth Elliot d. 2015), text is in copyright in the UK/EU regardless of US status.

## Rights matrix (initial; the P-tickets confirm each row)

| Document | Date | Expected status | Action |
|---|---|---|---|
| Leges Academiae Genevensis | 1559 | Public domain (original). Modern translations copyrighted. | Use original; commission or verify PD translation. Flag any modern translation for permission. |
| Luther, *To the Councilmen* (1524); Melanchthon addresses | 1518–1524 | PD original; check translation date | Use PD translation (e.g., Philadelphia edition, 1915–1932, PD) |
| Sturm, *De literarum ludis* | 1538 | PD; translation likely needed | New translation, draft, human review |
| Knox, *First Book of Discipline* | 1560 | PD | Use PD edition |
| Emmanuel College statutes | 1585 | PD original; translation check | Check translation date |
| Harvard *Rules and Precepts*; *New England's First Fruits* | 1642–1646 | PD | Reprint whole |
| Halle, Francke, *Pietas Hallensis* | 1705 (Eng. trans.) | PD | Reprint |
| Herrnhut Brotherly Agreement / Statutes | 1727 | PD original; translation check | Check translation date |
| Bristol Academy (Terrill bequest, 1679; rules) | 1679–1770 | PD | Reprint |
| Log College sources (Alexander 1845; Whitefield journals) | 1740–1845 | PD | Reprint |
| Carey, *Enquiry* | 1792 | PD | Reprint |
| LMS *Fundamental Principle* | 1796 | PD | Reprint |
| Serampore *Form of Agreement* | 1805 | PD | Reprint whole (short) |
| Andover Constitution and Associate Statutes | 1808 | PD | Reprint |
| ABCFM Constitution; *Instructions* to Judson's party | 1810–1812 | PD | Reprint |
| Princeton Seminary *Plan* | 1811 | PD | Reprint |
| Basel Mission rules | 1815–1840 | PD original (German); translation check | Translate |
| Serampore College charter and prospectus | 1818–1827 | PD | Reprint |
| Spurgeon, *Lectures to My Students*; *Sword and Trowel* college reports; *Autobiography* | 1875–1900 | PD | Reprint |
| SBTS Abstract of Principles; Boyce, *Three Changes* | 1856–1859 | PD | Reprint |
| Taylor, *China's Spiritual Need and Claims*; CIM *Principles and Practice* | 1865–1900 | PD (editions before 1930) | Use a pre-1930 edition; record edition |
| Guinness, East London Institute reports | 1873–1900 | PD | Reprint |
| Nyack Missionary Training Institute prospectus; Simpson writings | 1882–1919 | PD | Reprint |
| Moody Bible Institute prospectus / charter | 1886–1900 | PD | Reprint |
| SVM declaration and reports | 1886–1920 | PD | Reprint |
| Machen, *Westminster Theological Seminary: Its Purpose and Plan* | 1929 | PD in US (pre-1930) — verify edition date | Verify; reprint if 1929 |
| Prairie Bible Institute catalog / Maxwell | 1922–1960 | 1922–1929 PD; later check renewal | Check renewal |
| Wycliffe / SIL doctrinal statement and purpose | 1934–1942 | Check renewal; likely permission needed | Request permission from Wycliffe USA |
| Wheaton College catalog, statement of faith | 1860s PD; 1940s check | Check renewal for 1940s catalogs | Request from Wheaton archives |
| Christian Missions in Many Lands / Brethren principles | 1900s–1950s | Check | Request |
| Jim Elliot journals; Elisabeth Elliot writings | 1949–2015 | Copyrighted (Elisabeth Elliot Foundation; publishers) | Short fair-use quotation only; request permission for anything over ~200 words |
| Lloyd-Jones, LTS opening address; *Preaching and Preachers* | 1969–1977 | Copyrighted (MLJ Trust; publishers) | Short fair-use quotation only; request permission |
| London Theological Seminary aims/prospectus | 1977 | Copyrighted | Request from LTS |
| OMF *Principles and Practice* (post-1964 revisions) | 1964+ | Copyrighted | Use pre-1930 CIM edition for the charter; quote later revisions briefly |

## Permission request template (ticket P-nnn)

Subject: Permission to reprint [document title] in a forthcoming book

Body: We are preparing a book, working title *The School of Death*, collecting abridged founding documents of Protestant seminaries and missionary societies. We request non-exclusive world rights, all languages, print and electronic, to reprint approximately [N] words from [document title, date], as an abridgement with omissions marked. Attribution will read: [attribution line]. Please advise of any fee or conditions.

Log every request, reply, fee, and condition in `rights/permissions_log.csv` with columns: `slug, document, rights_holder, contact, date_requested, date_replied, status, fee, conditions, file`.
