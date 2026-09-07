STATUS: DONE (partial — see note below)
TICKET: B-geneva-academy
ROLE: Abridger
INPUTS READ: text/borgeaud1900-v1.txt; dossier.md field F5; book/plan/01_book_design.md section 6

# Abridged charter — Geneva Academy (`geneva-academy`)

**Source document:** *Leges Academiae Genevensis* (1559, Latin), promulgated and read in French
at the inauguration of 5 June 1559; reprinted in the appendix of Charles Borgeaud, *Histoire de
l'Universite de Geneve: L'Academie de Calvin, 1559-1798* (Geneva, 1900), document_id
`borgeaud1900-v1`.

**Translation:** none published and located this session. The 1562 Robert Fills English
translation (*The Lawes and Statutes of Geneva*) was identified by title in an earlier search
but not fetched or read this session (see book/research/geneva-academy/sources.csv — flagged
as a follow-up Fetcher ticket).

**IMPORTANT — scope of this abridgement.** The OCR text of the Latin *Leges* appendix (lines
approx. 29817-30355 of the fetched file) is badly corrupted by ligature and long-s
misrecognition to a degree well past the 5-errors-per-100-words threshold in
`03_research_protocol.md` Procedure F step 3. Per that rule, transcription of the full Latin
statutes is marked **NEEDS_RETRANSCRIPTION** and is **not** reproduced here — reproducing that
OCR output as though it were a reliable primary text would misrepresent the document, even
though it would technically pass a verbatim string-match against the (corrupted) fetched file.
This is logged as a BLOCKED sub-ticket: a cleaner scan (e.g. from e-rara.ch's own image set,
re-OCR'd, or the Fills 1562 English translation) is needed before the Academy's own statute text
can be abridged to the book's normal standard.

What follows instead is a **partial charter-in-substance**: the passages of Beza's inaugural
oration and of the statute-reading ceremony that Borgeaud quotes cleanly (in French, with
quotation marks, from legible OCR), plus his close paraphrase of the Latin curriculum and
governance articles. This is analogous to the `log-college` chapter's handling (descriptions in
lieu of a charter), except that here genuine short quotations from the founding act survive
alongside description. Every quoted sentence below is verbatim against `text/borgeaud1900-v1.txt`
and passes `check_quotes.py`.

## Structure of this partial abridgement

| # | Content | Source (quote_id) | KEEP/CUT |
|---|---|---|---|
| 1 | Ceremony of promulgation, 5 June 1559: officials present, Calvin's announcement | geneva-academy-q003 | KEEP (a) |
| 2 | Statutes and student confession of faith read aloud; oath required of rector and all teachers | geneva-academy-q004 | KEEP (b, f) |
| 3 | Beza elected rector; his Latin inaugural oration (French quotation of the close) | geneva-academy-q001 | KEEP (a) |
| 4 | Beza's description of the students as a "militia" | geneva-academy-q002 | KEEP (a) |
| 5 | Seven-class curriculum of the schola privata | geneva-academy-q005, geneva-academy-q006 | KEEP (d) |
| 6 | Governance: regents under a principal under the rector | geneva-academy-q007 | KEEP (f) |
| 7 | Schola publica: no classes, higher instruction, sole obligation a signed profession of faith | geneva-academy-q008 | KEEP (b, c) |
| 8 | Annual ceremony of 1 May: Leges read aloud, prizes given | geneva-academy-q009 | KEEP (e) |
| — | The Latin statute text itself (full articles) | borgeaud1900-v1 appendix | CUT — NEEDS_RETRANSCRIPTION, see note above |

## Headnote

*Leges Academiae Genevensis*, promulgated in Latin with a French reading, Geneva, 5 June 1559;
quoted here at second hand from Charles Borgeaud's 1900 edition, which reprints the statutes and
the inaugural proceedings from the Genevan council registers and Beza's own oration.

## Translation note

All passages below are given in English. No published translation of Borgeaud was located this session, so this is a new, literal working translation of his French (and, where he quotes it directly, of the Latin), produced by the Abridger. Per abridgement rule 6 this is marked **[New translation, draft]** and should be checked by a competent reader of French/Latin before publication. The French/Latin original for each passage is preserved verbatim in quotes.jsonl under the matching quote_id, so the translation can be checked sentence by sentence.

## Text

[Ceremony of promulgation, 5 June 1559.] On 5 June, in the presence of the four lord syndics,
Henri Aubert, Jehan Porral, Jehan-François Bernard and Barthélemy Lect, of several councillors,
of the ministers, professors and regents, and of a large assembly of men of letters and
schoolboys, Calvin mounted the pulpit and, announcing the institution of the Academy, invited
the assembly to join their prayers to his. Then, at the syndics' order, the secretary of the
Council, Michel Roset, read aloud, in French, the laws and statutes of the college, together
with the confession of faith required of the students and the oath that the rector and all
those who taught in either section of the school were required to swear. He then proclaimed the
elevation to the rectorate of Theodore de Beza, elected by the ministers and confirmed by the
Seigneurie.

[...]

Beza then delivered a written inaugural oration in Latin, on the origin, usefulness and dignity
of learning, closing with an appeal to the students:

> "You have not come to this place — said the first rector of the Academy of Geneva, ending his
> oration, to those who were about to become its first students — as most of the Greeks once
> did, who went off to the spectacles of their gymnasia to watch fleeting games."

[...]

Elsewhere, recalling this same 5 June 1559 address at St Pierre, it is recorded that the first
rector reminded the young men listening to him that they were a militia [*milice*].

[...]

[On the course of study.] The Schola privata comprises seven classes, with strictly fixed
programs suited to the pupils' degree of preparation. From the seventh, where the child learns
to read, in French and in Latin, and to write, he is led by successive stages to the fourth,
where Greek is begun; to the third, where Cicero, Virgil, and Caesar are studied; to the second,
where Homer, Xenophon, and Polybius are read; and to the first, where the pupil perfects himself
in dialectic and rhetoric, studying the orations of Cicero and Demosthenes. Pupils are grouped
within each class into "decuries," or groups of ten, without regard to age or household, but
solely according to each one's progress. The foremost of the group sits at the head and serves
as monitor.

[...]

The regents of the college are subject to a principal, the *ludimagister*, who is himself
subordinate to the rector, supreme head of the whole school, elected from within the Company of
ministers and professors.

[...]

[On the Schola publica.] The Schola publica is marked by the absence of classes, by the higher
instruction given there, by the standing of the holders of its chairs, who sit in the cloister
beside the city's pastors, and by the fact that its students are no longer bound by the
discipline of the gymnasium and are held to no other outward obligation than to give their name
to the rector and to sign a profession of faith.

[...]

[On the annual assembly.] On the first of May, the whole school must gather in the temple of St
Pierre — *tota schola in S. Petri templo convenito* — and, once the rector, surrounded by the
company of ministers and professors, the principal, and the regents, has had the Leges Academiae
read aloud and has briefly urged their observance, the two most deserving pupils of each class
are presented to the lord syndic or councillor present [and receive a small prize].
