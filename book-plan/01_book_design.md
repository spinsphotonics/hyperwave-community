# 01 — Book Design

## 1. Thesis (one paragraph, to be printed in the Introduction)

From 1559 onward, Protestants built schools whose stated purpose was not to produce scholars but to produce men and women who would be sent, often to die. Calvin's Geneva Academy was called a "school of death" because its graduates were dispatched into France as pastors and many were executed. The same intent reappears, in the founding documents, at Emmanuel College Cambridge, at Harvard, at the Log College, at Andover (which sent Adoniram Judson), at Princeton, at Spurgeon's Pastors' College, in the "Principles and Practice" of Hudson Taylor's China Inland Mission, in the Bible institutes of Guinness, Simpson and Moody, at Prairie and Wheaton (which trained Jim and Elisabeth Elliot), and in the seminary Martyn Lloyd-Jones opened in 1977. This book puts the charters side by side, abridged and annotated, and asks of each: whom did they admit, what did they teach, how did teachers and students live together, what did the school emphasize, and where did it send its people.

## 2. Audience

Pastors, seminary faculty and students, mission agency staff, and serious lay readers. Assume the reader knows who Calvin, Spurgeon and Hudson Taylor are, but not what the Leges Academiae Genevensis say. Do not assume Latin, French or German.

## 3. Length and shape

| Item | Target |
|---|---|
| Total length | 130,000–160,000 words |
| Parts | 6 (one per era) + Introduction + Appendices |
| Chapters | 28–34 institution chapters (Tier A = required, Tier B = if budget allows) |
| Institution chapter | 3,500–5,000 words including the abridged charter |
| Abridged charter within a chapter | 1,200–2,500 words |
| Part introduction | 1,500–2,500 words |
| Book introduction | 5,000–7,000 words |
| Appendices | Full text of up to 8 short public-domain charters; lineage chart; timeline; glossary; comparative tables |

## 4. Parts and chapter order

Chapter order inside a part is by founding date. Tier is from `02_institution_roster.md`.

**Introduction — "The School of Death."** What a charter is; why founding documents reveal intent better than later histories; the Geneva nickname and its source; how to read the book; the five questions asked of every school (admission, curriculum, common life and faculty interaction, emphasis, sending).

**Part I — Reformation Academies (1518–1620).** Wittenberg; Zurich Prophezei; Strasbourg; Geneva Academy; Scotland's First Book of Discipline; Leiden; Emmanuel College Cambridge; Saumur/Sedan (Tier B).

**Part II — Puritans, Pietists and Dissenters (1636–1790).** Harvard; Yale (Tier B); Halle and the Danish-Halle Mission; Herrnhut and the Moravian sending; Bristol Baptist Academy; Doddridge's Northampton Academy (Tier B); the Log College; the College of New Jersey (Princeton); Trevecca (Tier B).

**Part III — The Missionary Awakening and the Seminary (1792–1860).** Baptist Missionary Society and the Serampore Form of Agreement; London Missionary Society; Church Missionary Society and Islington College; Andover Theological Seminary; the American Board of Commissioners for Foreign Missions and Judson; Baptist Triennial Convention and Judson (combined chapter with ABCFM or separate, decided by Coordinator at ticket D-stage); Princeton Theological Seminary; Basel Mission Seminary; Serampore College; New College Edinburgh (Tier B); Spurgeon's Pastors' College; Southern Baptist Theological Seminary.

**Part IV — Faith Missions and the Bible Institute (1865–1930).** China Inland Mission; East London Institute for Home and Foreign Missions (Harley College); Missionary Training Institute, Nyack; Moody Bible Institute; Student Volunteer Movement; Africa Inland Mission and Sudan Interior Mission (combined, Tier B); Bible Institute of Los Angeles (Tier B); Glasgow Bible Training Institute (Tier B); Prairie Bible Institute; Columbia Bible College (Tier B); Westminster Theological Seminary.

**Part V — Translators, Tribes and the Auca Five (1930–1960).** Wycliffe Bible Translators / Summer Institute of Linguistics; Wheaton College; Christian Missions in Many Lands (Brethren) and the Elliots; Inter-Varsity and Urbana (Tier B); New Tribes Mission (Tier B); Missionary Aviation Fellowship (Tier B).

**Part VI — Lloyd-Jones and the Reformed Recovery (1938–1980).** Westminster Chapel and the Westminster Fellowship; Evangelical Library, IFES and Banner of Truth (combined, Tier B); London Theological Seminary; Overseas Missionary Fellowship (CIM renamed; combined with CIM chapter or short coda); Reformed Theological Seminary / Trinity / Gordon-Conwell (Tier C, appendix note only).

**Appendices.** A: full-text charters (public domain only, see `06_rights_and_permissions.md`). B: lineage chart. C: timeline. D: comparative tables. E: glossary. F: sources and archives.

## 5. Chapter template (mandatory; file `templates/chapter.md`)

Every institution chapter has exactly these sections, in this order, with these headings. Word budgets are targets, not limits, but the Editor flags any section more than 30% over.

1. **Epigraph** (≤60 words). A verbatim quotation from the charter or founder, with citation.
2. **Founding** (400–600 words). Who, when, where, why. What crisis or need produced the school. Cite dossier fields F1–F8.
3. **The Charter** (1,200–2,500 words). The abridged founding document, set as a block, with a one-sentence headnote stating the document's title, date, language, and edition used. Editorial omissions are marked `[...]`; editorial insertions are in square brackets.
4. **Admission** (250–400 words). Who could enter, at what age, with what prior learning, what examination, what testimony of character or conversion, whether women were admitted and in what capacity.
5. **Curriculum** (500–800 words). Subjects, languages, set texts, sequence, length of course, examinations, degrees or certificates. Include a short table if the dossier has a weekly or yearly schedule.
6. **Common Life and the Faculty** (500–800 words). Daily timetable, worship, lodging, meals, discipline, the ratio of teachers to students, the forms of contact between teacher and student (lecture, disputation, tutorial, Friday afternoon lecture, Saturday sermon class, household table, letters), and named examples.
7. **Emphasis** (400–600 words). What the founders said mattered most, in their words. The school's distinctive stress (doctrine, piety, preaching, languages, evangelism, self-support, faith principle, etc.).
8. **Sending** (300–500 words). How students were placed or sent; the sending body; where they went; casualty figures if primary sources give them.
9. **Fruit** (200–400 words). Three to six named alumni or missionaries with one sentence each and a citation.
10. **Sources** (uncounted). Primary sources used, then secondary, in Chicago notes-bibliography form.

## 6. Abridgement rules for charters (Abridger must apply exactly)

1. Work only from the Fetcher's plain-text transcription in `research/<slug>/text/`. Never from memory, never from a secondary quotation.
2. Keep, in order of priority, and in this order if the document has them:
   a. Title, date, issuing authority, preamble or statement of purpose.
   b. Doctrinal basis or confession required.
   c. Admission requirements.
   d. Course of study, subjects, languages, set books, duration.
   e. Rules of daily life, worship, discipline, dress, residence.
   f. Duties of teachers and their relation to students.
   g. Examinations, degrees, certificates, or the licensing/sending clause.
   h. Financial principle only if it is doctrinal (for example CIM's no-solicitation rule, Spurgeon's free tuition).
3. Cut: administrative detail (trustee elections, quorum rules, property clauses, salaries, seals), repeated formulae, and lists of names, unless the Coordinator has marked them as needed.
4. Mark every cut with `[...]` on its own line if a whole article is cut, or inline if part of a sentence is cut.
5. Never reorder. If the document's own order is confusing, add a bracketed editorial heading such as `[On the course of study]` but keep the original sequence.
6. Translations: use the public-domain or licensed translation named in the source log. If none exists, produce a literal translation, mark it `[New translation, draft]`, and flag for human review by a competent reader of the language. Never translate from a translation.
7. Modernize spelling only for English before 1700 and only by changing letters (u/v, i/j, long s), never words. Keep original punctuation.
8. Target length 1,200–2,500 words. If the document is under 1,200 words, print it whole. If it is over 15,000 words, produce the abridgement in two passes: first an outline of all articles with word counts, then the selection, and record the outline in `charter_abridged.md` under "Structure of the full document."
9. Record, at the top of `charter_abridged.md`, the original's total word count, the abridgement's word count, and the list of articles/sections retained and cut.

## 7. Style guide

- American spelling in narrative; original spelling in quotations.
- Past tense for narrative; present tense when describing what a document says ("The Plan requires...").
- Dates as "5 June 1559." Centuries spelled out. Use "c." for approximate years and cite.
- Names: use the form most common in English scholarship, fixed in the roster (Adoniram Judson, Hudson Taylor, Elisabeth Elliot, Martyn Lloyd-Jones, Theodore Beza). First mention gives full name and dates in parentheses.
- Scripture references in the form "2 Timothy 2:2."
- Footnotes, Chicago notes-bibliography. Every quotation and every specific fact (date, number, name) is footnoted.
- No adjectives of praise ("great," "godly," "remarkable") in the narrative voice. Let the documents speak.
- Do not use the words "unique," "iconic," "vibrant," "tapestry," "testament to."
- Sentences under 30 words on average. One idea per sentence.
- The narrative voice never speculates. If a fact is `UNVERIFIED` in the dossier, the chapter does not state it.

## 8. The five questions (printed in the Introduction, applied in every chapter)

1. Whom did they admit, and on what testimony?
2. What did they teach, in what order, from what books, in what languages?
3. How did teachers and students live and speak together?
4. What did the founders say mattered most?
5. Where did they send their people, and what did it cost?
