STATUS: DONE (descriptions in lieu of a charter — see note below)
TICKET: B-zurich-prophezei
ROLE: Abridger
INPUTS READ: dossier.md field F5; text/christoffel1858.txt; text/jackson1901.txt; text/simpson1902.txt; text/grob1883.txt; text/uzh-news-prophezey.txt; text/adfontes-prophezei.txt; book/plan/01_book_design.md section 6

NOTE ON `check_quotes.py --charter`: that script's automated charter check assumes a
single `document_id` in backticks and one `text/` file (it was written for a chapter
built from one abridged primary document). This file instead assembles passages from
six different fetched sources (see the table below), because no single founding
document was located (F5 = NOT FOUND). The script therefore cannot run in `--charter`
mode against this file. In its place, every passage below was verified the same way
every `quotes.jsonl` record was verified: it is one of the 21 quote records that already
passed `python3 check_quotes.py research/zurich-prophezei` (21/21, 0 failures — see
`verification.md`), and every passage's quote_id is named in the table below so the
correspondence can be checked line by line. No passage below was typed from memory.

# Abridged charter — The Prophezei, Zurich (`zurich-prophezei`)

**No founding document was located and read this session.** Dossier field F5 records
`NOT FOUND` for a primary charter, ordinance, or council-decree text: no digitized text
of Zwingli's own founding statement, the Zurich Council's authorizing decree, or a Zurich
school order was located and read this session (see `sources.csv` and `discrepancies.md`
for what was searched and what remains outstanding — principally Zwingli's *Sämtliche
Werke*/Corpus Reformatorum edition, held at e-rara.ch and the Zentralbibliothek Zürich
per the roster, and Bullinger's *Reformationsgeschichte*, whose one fetched scan was too
badly OCR'd to use).

Per the precedent set for `log-college` in `04_task_tickets.md` ("Since no charter
exists... mark the chapter's Charter section as 'Descriptions in lieu of a charter'"),
this file assembles, instead of an abridged primary text, the descriptions of the
Prophezei's founding order and daily practice that this session's fetched sources
provide — several of which reproduce short phrases attributed to Zwingli or his
contemporaries, always at second hand through a later narrator, never from an edition of
the document itself. Every passage below is verbatim against its named `text/` file and
passes `check_quotes.py`.

## What is assembled here, and from what

| # | Content | Source (quote_id) | Nature |
|---|---|---|---|
| 1 | Naming and founding of "the prophesying" (19th July 1525 per this 1858 translation; see discrepancies.md item 1 on the date) | christoffel1858 (zurich-prophezei-q001) | 1858 English translation of a German biography; narrator's own sentence |
| 2 | Daily assembly at eight o'clock; who was required to attend | christoffel1858 (zurich-prophezei-q002) | Same |
| 3 | Method of exposition: Vulgate read, Hebrew and Greek compared, explained in Latin | christoffel1858 (zurich-prophezei-q003) | Same |
| 4 | Hebrew masters: Ceporin, then Pellican; exercise lasted about an hour | christoffel1858 (zurich-prophezei-q004) | Same |
| 5 | The German sermon and Zwingli's own prayer, given in English translation | christoffel1858 (zurich-prophezei-q005) | Same (prayer attributed directly to Zwingli, but only as translated and transmitted by Christoffel) |
| 6 | Myconius's assessment of Zwingli's plan for the institution | christoffel1858 (zurich-prophezei-q006) | Myconius quoted at second hand within the 1858 narrative |
| 7 | The name "prophesying" and its scriptural warrant, 1 Corinthians 14 | christoffel1858 (zurich-prophezei-q007) | Narrator's footnote |
| 8 | Funding: benefices redirected to pay the new institution's teachers | christoffel1858 (zurich-prophezei-q008) | Narrator's own sentence |
| 9 | Zwingli chosen rector of the Carolinum, 14 April 1525 | jackson1901 (zurich-prophezei-q009) | 1901 scholarly biography; narrator's own sentence |
| 10 | Zwingli "took part himself in the biblical instruction, which he had made part of the curriculum" | jackson1901 (zurich-prophezei-q010) | Same |
| 11 | Zwingli's 30 June 1525 tract on the preaching office, against the Baptists' claim that all believers are prophets | jackson1901 (zurich-prophezei-q011) | Same |
| 12 | 1523 Council decree: daily exposition in Hebrew, Greek, Latin; candidates for the ministry "thoroughly trained" | simpson1902 (zurich-prophezei-q012) | 1902 biography's narrative paraphrase of a Council decree, not the decree's own text |
| 13 | Daily biblical discourse except Friday, in the chapter-house | grob1883 (zurich-prophezei-q013) | 1883 biography; narrator's summary |
| 14 | "A prosperous theological seminary," with named teachers | grob1883 (zurich-prophezei-q014) | Same |
| 15 | 19 June 1525, 8:00 am, Grossmünster; daily except Friday (market day) and Sunday | uzh-news-prophezey (zurich-prophezei-q015) | 2025 University of Zurich Faculty of Theology feature |
| 16 | Council's approval for Zwingli's classes; mission to teach Hebrew, Greek, Latin "necessary to the proper understanding of the Holy Scriptures" | uzh-news-prophezey (zurich-prophezei-q016) | Same, partly quoting an unspecified "remit" |
| 17 | Judith Engeler (UZH): Zwingli's aim was to "re-educate" the orthodox priests | uzh-news-prophezey (zurich-prophezei-q017) | Same, a present-day scholar's summary |
| 18 | Claimed time of the lay service (9:00 pm — see discrepancies.md item 2) | uzh-news-prophezey (zurich-prophezei-q018) | Same |
| 19 | "The sessions began on 19 June 1525... a clear order that they were to undertake a combination of study and worship" | adfontes-prophezei, quoting Bruce Gordon 2021 p.142 (zurich-prophezei-q019) | Blog quotation of a named 2021 academic monograph; second-hand |
| 20 | The first morning's work: Genesis in Hebrew, Greek, Latin, German; daily except Fridays and Sundays; required attendance of canons, city clergy, Latin School students | adfontes-prophezei, quoting Gordon p.142 (zurich-prophezei-q020) | Same |
| 21 | Zwingli drew together the theological fruits; Leo Jud prepared the German sermon | adfontes-prophezei, quoting Gordon p.143 (zurich-prophezei-q021) | Same |

## Headnote

No single founding document is reproduced here. This session's fetched sources — four
public-domain English biographies of Zwingli (1858, 1883, 1901, 1902) and two
contemporary web accounts, one from the University of Zurich's own Faculty of Theology
(2025) and one quoting Bruce Gordon's 2021 academic biography *Zwingli: God's Armed
Prophet* by page number — describe, but do not themselves reprint, any Zurich Council
decree, Zwingli ordinance, or Bullinger passage establishing the Prophezei. What follows
is a description in lieu of a charter, in the sense set for `log-college` in this
project's ticket list.

## Text

[On the founding and its name.] In place of the choir-service in the morning, "heedlessly
mumbled over by canons and chaplains," a new service came into existence — dated 19th
July 1525 by this 1858 translation, but 19 June 1525 by two independently dated sources
read this session (see discrepancies.md item 1) — called "the prophesying," or exposition
of Scripture. Zwingli himself called the exercise "prophesying," in reference to the
proceedings alluded to in 1 Corinthians 14.

[...]

[On the daily order.] At eight o'clock, all the town-parsons, predicants, canons, and
chaplains, and the more advanced scholars, assembled in the choir of the Minster church.
Zwingli having delivered, in the Latin language, the opening prayer, the exposition was
begun with the first chapter of the First Book of Moses: a scholar read a section of the
Latin translation of the Bible (the Vulgate), and the teacher commented upon it; the same
section was then read in the Hebrew text, then in the Greek translation (the Septuagint),
and critically, as well as doctrinally and practically, explained in Latin. At first
Zwingli himself expounded out of the original text and the Greek translation, but
afterward a special master was appointed for the Hebrew — first Ceporin, and after his
early death, Pellican. This exercise lasted about an hour.

[...]

In the meantime, the congregation had assembled to hear the sermon. An ecclesiastic
mounted the pulpit and delivered a prayer composed by Zwingli:

> "O merciful God, heavenly Father! since Thy Word is a light to our feet and a lamp to
> our path, we pray Thee that Thou wouldest, through Christ, who is the true light of the
> whole world, open and illuminate our minds, clearly and purely to understand Thy truth,
> that so we may in no respect offend Thy High Majesty, through our Lord and Saviour,
> Jesus Christ. Amen."

The section of Scripture already treated learnedly was then expounded in a manner level
to the capacities of the congregation, and the whole proceeding closed with prayer.

[...]

[On what mattered most.] Of Zwingli's idea for the institution, Myconius said: "Zwingli
formed the plan of founding an institution specially intended for the study of profane
learning and scientific theology, and I doubt not that, had he survived the full
execution of his plan, it would not have found its equal anywhere."

[...]

[On funding.] The amount of the benefices set free by the reduction in the number of
canons was applied to the better payment of the teachers of the foundation school, so
that better qualified men might be obtained; it was by this same reallocation that
Zwingli "called into life an altogether new institution of a higher order, specially
adapted for advancing the work of education among the candidates for the priesthood, for
affording intellectual and spiritual exercise to the canons, and promoting edification
also among the people."

[...]

[On the 1523 Council decree that prepared the way.] It was enacted that all the clergy
of the Minster should preach the word of God; that the Bible should be read and
explained daily in three languages — Hebrew, Greek, and Latin; that greater attention
should be paid to education, and that candidates for the ministry should be thoroughly
trained.

[...]

[On Zwingli's own office.] On 14 April 1525, Zwingli was chosen rector of the Carolinum,
the Great Minster school. He used his new position to improve the schools and took part
himself in the biblical instruction, which he had made part of the curriculum.

[...]

[On a contemporary summary of the exercise's shape, quoting Bruce Gordon's 2021
biography at second hand.] "The sessions began on 19 June 1525 in the Grossmünster, with
a clear order that they were to undertake a combination of study and worship... The
Prophezei began work at eight o'clock that first morning, turning to the opening of
Genesis in Hebrew, Greek, Latin, and German. The sessions then took place every day of
the week except Fridays and Sundays, with the canons, city clergy, and students of the
Latin School required to attend... The plan was simple: to work through all the books of
the Old Testament — and when finished, to start again." Once the Latin, Hebrew, and Greek
texts had been examined and Zwingli had drawn together the theological fruits, "it fell
to Leo Jud... to prepare a sermon based on Zwingli's words, to be delivered in German for
the faithful who had come to the Grossmünster for the final part of the session."
