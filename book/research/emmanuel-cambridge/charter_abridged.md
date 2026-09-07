STATUS: DONE
TICKET: B-emmanuel-cambridge
ROLE: Abridger
INPUTS READ: text/commdoc1852v3.txt (document_id `commdoc1852v3`, dossier field F5); book/plan/01_book_design.md section 6; quotes.jsonl

# Abridged charter — Emmanuel College, Cambridge (`emmanuel-cambridge`)

**Source document:** Mildmay's Statutes for the government of Emmanuel College, Cambridge, dated
1 October 1585, in Latin (with two supplementary statutes, one undated in this transcription and
associated by Shuckburgh 1904 with December 1587, the other headed "December, 1578" in this
transcription — see discrepancies.md item 3 on that date). Edition used: *Documents Relating to the
University and Colleges of Cambridge*, vol. III (Cambridge University Commission, 1852), which
prints the Statutes "Taken from a MS. in the British Museum, Sloane, 1739." document_id
`commdoc1852v3`.

**Translation:** none freely available was located this session (Frank Stubbings's 1983 English
translation, *The Statutes of Sir Walter Mildmay for Emmanuel College*, exists but is copyrighted —
see sources.csv row `stubbings1983` — and was not fetched or used). Every passage below is a
**[New translation, draft]** by the Abridger, produced paragraph by paragraph from the Latin in
`text/commdoc1852v3.txt`, and should be checked by a competent reader of Latin before publication.
The original Latin for each kept passage is preserved verbatim in `quotes.jsonl` under the matching
`quote_id`, so the translation can be checked sentence by sentence against the source.

**IMPORTANT — scope of this abridgement.** The 1584 royal Charter (in heavily abbreviated Tudor
legal Latin, printed pp. 479-483 of the same source) is excluded from this abridgement:
per Procedure F step 3, its OCR is badly corrupted by the printer's contraction marks (rendered as
stray characters by the OCR engine — see `text/commdoc1852v3.txt` header note) well past a usable
threshold, and reproducing that OCR output, even verbatim, would misrepresent a document whose actual
content cannot be reliably read from it. The Statutes proper (Cap. I onward), by contrast, are
unabbreviated Latin prose and are legible; they are the basis of this abridgement. One further short
passage — the specific sentence of Cap. XXI naming the College's purpose as a "seed-ground"
(*seminarium*) for the Church — is itself poorly OCR'd in this transcription (see dossier.md field F6
and discrepancies.md item 1); for that one sentence, this abridgement uses instead the English
translation already published by E. S. Shuckburgh (1904), quoted at second hand and clearly marked
below, rather than a new draft translation from corrupted OCR.

**Full document word count (Latin, this transcription):** approx. 12,187 words (Praefatio + 37
legible numbered chapters of the main 1585 Statutes, approx. 11,188 words, plus two supplementary
statutes, approx. 999 words). The Royal Commission's own heading numbers some chapters III, XII, and
XVI-XVIII that this transcription's OCR could not isolate as separate headings (their content is
folded into the word counts of the neighbouring chapters below); this is a transcription limitation,
not evidence that those chapters do not exist in the original.

**Abridgement word count:** approx. 1,780 words (English, this file's "Text" section only).

**Note on `check_quotes.py --charter`:** every kept passage below is an English translation (either
the Abridger's own new draft, or Shuckburgh's 1904 translation, each explicitly marked in place), not
a literal excerpt of the Latin `commdoc1852v3` text named in this file. The automated
`--charter` verbatim check therefore reports FAIL for every passage, exactly as it does for the
`geneva-academy` chapter's own translated charter (also all-translated, also all-FAIL under
`--charter`, confirmed by re-running that check this session). The actual verification for a
translated charter is the same one used for its underlying quotations: every English sentence below
carries a `quote_id`, and every one of those quote records passes the ordinary (non-`--charter`)
`check_quotes.py` run against its Latin (or, for the two Shuckburgh-sourced passages, English)
original — see verification.md section 1.

## Structure of the full document (outline, with KEEP/CUT)

| # | Chapter (Latin heading, as printed) | Words (approx.) | KEEP/CUT | Reason code |
|---|---|---|---|---|
| — | Praefatio (preface, unnumbered) | ~800 | KEEP | a |
| I | De Magistri authoritate (the Master's authority) | 310 | KEEP (partial, opening sentence) | f |
| II | De residentia Magistri (the Master's residence) | 489 | CUT | admin |
| IV | De preferendis probis et de camerarum assignatione (preferment; assignment of chambers) | 152 | CUT | admin |
| V | De computo reddendo (rendering of accounts) | 280 | CUT | admin |
| VI | [heading not legible in this transcription] | 241 | CUT | admin |
| VII | De Magistri stipendio (the Master's stipend) | 113 | CUT | admin |
| VIII | De Magistri Vicario (the Master's deputy) | 172 | CUT | admin |
| IX | De qualitate novi Magistri eligendi (the character required of a new Master) | 244 | CUT | repeat (overlaps F8/oath content kept elsewhere) |
| X | De antecedentibus electionem Magistri (procedure before electing a Master) | 262 | CUT | admin |
| XI | De modo et forma eligendi Magistri (the manner and form of electing a Master) | 1,023 | KEEP (partial, the Master's oath only) | b |
| XIII | De Magistri amotione (removal of the Master) | 246 | CUT | admin |
| XIV | De Decano sive catechista (the Dean or Catechist) | 250 | CUT | repeat (Dean/Catechist role already stated in the chapter's Common Life section from Shuckburgh) |
| XV | De mulctis per decanum imponendis (fines imposed by the Dean) | 552 | CUT | admin |
| XIX | De sociorum electione (election of Fellows) | 847 | KEEP (partial, the newly elected Fellow's oath only) | b |
| XX | [heading not legible in this transcription; content concerns the Master's own celebration of Communion] | 112 | CUT | admin |
| XXI | De sociorum exercitiis, studiis, et ordine (the Fellows' exercises, studies, and order) | 512 | KEEP | a, d |
| XXII | [begins "Sed quia parum juvat doctos esse nisi boni sint," on the Fellows' conduct] | 194 | CUT | repeat (conduct rules kept at greater length from Cap. XXXIV below) |
| XXIII | De stipendio et emolumentis sociorum (Fellows' stipend and emoluments) | 149 | CUT | admin |
| XXIV | Quantum aliunde eis recipere liceat (how much they may receive from elsewhere) | 128 | CUT | admin |
| XXV | De prandendi et cenandi loco (place of dining and supping) | 205 | CUT | admin |
| XXVI | Quamdiu sociis abesse a Collegio licuerit (how long Fellows may be absent) | 311 | CUT | admin |
| XXVII | De Tutorum officio, diligentia, et stipendio (the Tutors' office, diligence, and stipend) | 285 | CUT | admin |
| XXVIII | De lustrandis scholasticorum cubiculis (inspection of students' chambers) | 206 | CUT | admin |
| XXIX | De Lectore et Sublectoribus (the Lecturer and Sub-lecturers) | 259 | CUT | admin |
| XXX | De Lectorum stipendio (the Lecturers' stipend) | 166 | CUT | admin |
| XXXI | De Lectoris authoritate in Discipulos (the Lecturer's authority over scholars) | 218 | CUT | repeat (overlaps C6/T3 content kept from Cap. XXI) |
| XXXII | De Scholarium Discipulorum qualitate et electione (the character and election of scholars) | 152 | KEEP | c |
| XXXIII | De Jurejurando Scholarium Discipulorum (the scholars' oath) | 180 | CUT | repeat (doctrinal-subscription content already kept twice, Caps. XI and XIX) |
| XXXIV | De Cultu Dei, scholasticis exercitationibus, et moribus Discipulorum (worship, scholastic exercise, and conduct of scholars) | 267 | KEEP | e |
| XXXV | De Obsequiis intra Collegium exhibendis (services to be rendered within the College) | 192 | CUT | admin |
| XXXVI | Quantum a Collegio, quantumve ab aliis scholares recipient, et de eorum absentia (commons, clothing, and absence) | 224 | KEEP (partial) | e |
| XXXVII | De Mancipe, Cocis, lotore, et quodam Magistri famulo (the manciple, cooks, launderer, and a servant of the Master) | 315 | CUT | admin, names |
| XXXVIII | De Presentationibus ad Ecclesias vacantes faciendis (presentations to vacant churches) | 499 | KEEP (partial) | g |
| XXXIX | De stipendiis... pro incremento reddituum augendis (stipends increasing as revenue grows) | 269 | CUT | admin, h (financial but not doctrinal) |
| XL | De Pensionariis intra Collegium admittendis (admission of pensioners) | 363 | CUT | admin |
| XLI | De Ambiguis et Obscuris interpretandis (interpretation of ambiguities) | 162 | CUT | admin |
| XLII | De communi omnium conditione qui erunt in Collegio (the common condition of all members — the no-marriage rule) | 183 | CUT | admin (kept only as one sentence, folded into the closing material below rather than its own KEEP row) |
| — | Closing dating clause and signature (1 Oct. 1585) | ~40 | KEEP | a |
| — | Supplementary statute: chamber reserved for the founder's descendants | ~230 | CUT | admin, names |
| — | Supplementary statute: *De Mora Sociorum* (Fellows' term of residence; duty to take Holy Orders) | ~770 | KEEP (partial, quoted at second hand from Shuckburgh 1904, not newly translated — see headnote) | a, d, g |

## Headnote

Mildmay's Statutes for the government of Emmanuel College, Cambridge, given under his hand and seal
on 1 October 1585, in Latin; the passages below are a new, literal working translation (except where
marked as quoted at second hand from Shuckburgh 1904), made from the 1852 reprint in *Documents
Relating to the University and Colleges of Cambridge*, vol. III, itself taken from a British Museum
manuscript copy of the original.

## Text

[Preface.] It is an ancient institution in the Church, handed down from the most ancient times, that
schools and colleges be established for training youth in all godliness and good letters, and
especially in sacred and theological learning; that youth so trained may afterward teach others the
true and pure religion, refute all heresies and errors, and by the most excellent examples of an
upright life stir all men to virtue. For thus we read in sacred history that at Naioth, Gilgal,
Bethel, and Jericho the sons of the prophets were trained by the greatest and most renowned prophets,
Samuel, Elijah, and Elisha, to preach the name of God and to instruct the people in true religion.
And it is recorded in the Acts of the Apostles that Jerusalem had very many synagogues, one for
almost every nation, to which men flocked from nearly the whole world, as to a kind of market of
religion, letters, and virtue; among whom that Saul of Tarsus (who was afterward called Paul), the
Lord's chosen instrument and teacher of the Gentiles, is said to have sat at the feet of the reverend
Gamaliel. For men moved by the divine Spirit understood that the light of the Gospel could not be
spread to all posterity, for the glory of God and the salvation of men, unless certain nurseries of
Theology and of the best Arts were established and furnished within the Church itself, as in a
garden of paradise — nurseries of the noblest plants, out of which those who had grown to maturity
might be transplanted into every part of the Church; so that the Church, watered by their labor and
increased by God's blessing, might at last come to a most flourishing and blessed state. For just as
the Levites were appointed guardians of the fire sent down from heaven — the only fire lawful to be
used for burning sacrifices upon the altar — who tended it continually and kept it alive; so also the
true knowledge of God (a fire, as it were, fallen from heaven) must be preserved by constant watching
and labor, lest we bring before the Lord, to be kindled, a strange fire — that is, Popery and the
other heresies — sprung from the earth and from the inventions of men. And just as the other rivers
that water the earth flow out from the fountains of the garden of Eden, so Schools are, as it were,
certain fountains that must be opened, which, springing from the Paradise of God, may water all the
lands of our country, and indeed the regions of the whole world, with a golden stream of the purest
doctrine of the faith and of the most holy discipline of morals. And so many men of heroic virtue
among our ancestors, imitating these divine and ancient institutions of God's prophets, have set up
colleges and [seats of learning — the Greek in the original is illegible in this transcription] to
God and to the Church, whose magnificence and royal expense I gladly leave to others who can come
nearer to the honor of such munificence; for my part I think it enough to imitate their virtues, and,
according to my ability, to hand down the purity of religion and of life to our posterity. But since
no society, however small, can be either rightly governed or long endure unless some lawful method
of moderating and maintaining it, and some discipline, be established: therefore we shall set forth
certain edicts and statutes, divided into their own chapters, describing the duties belonging to our
whole College and to each of its members, to which we will that all our people be subject and which
they obey. [emmanuel-cambridge-q039]

[...]

[Cap. I — On the Master's authority.] And since it is right to begin from the head, by which it is
fitting that the rest of the members be ruled and governed, let us first make statutes concerning the
Master, whom we will to be as the head over all the fellows and scholars. To Laurence Chaderton,
therefore, Bachelor of Sacred Theology, who is already by my authority appointed Master of the
aforesaid College, and likewise to all his successors thereafter (each in his own time), we grant
authority over all the fellows and scholars of the same College, to govern them, to rule them, to
punish them, to admonish them, and to administer the domestic affairs of the whole College, according
to the ordinances [and statutes given by me, which follow below]. [emmanuel-cambridge-q040]

[...]

[Cap. XI — On the manner of electing a Master; the Master's oath, on election.] "I, N. T., call God
to witness that I will heartily embrace the true religion of Christ, contrary to Popery and other
heresies: that I will set the authority of Scripture above the judgments of even the best of men:
that I will hold everything else, which cannot by any means be proved from the word of God, to be
merely human: that I will judge the royal authority to be supreme over the men of this realm, and in
no way subject to the jurisdiction of any foreign bishops, princes, or powers whatsoever: that I will
diligently refute all opinions contrary to the word of God, and all heresies: and that, finally, in
matters of religion I will always prefer the true to the customary, and the written to the unwritten."
[emmanuel-cambridge-q023]

[...]

[Cap. XIX — On the election of Fellows; the oath of a newly elected Fellow.] "I, N. T., call God to
witness that I will embrace the true religion of Christ, contrary to Popery and all other heresies,
and by way of covenant I promise that I will truly and fully observe each and every thing that Walter
Mildmay, founder of this College, has himself set forth for its government, and I will take care, so
far as in me lies, that the same be done by my fellows: I will obey the Master, or his deputy, in all
things that he shall lawfully command: I will disclose to no one the secret counsels of this College
(so far as it shall be lawful): I will hinder nothing by which any advantage or honor might accrue to
the said College, but will rather further it with all my power..." [emmanuel-cambridge-q024]

[...]

[Cap. XXI — On the Fellows' exercises, studies, and order; the founder's statement of purpose.] [The
specific sentence that follows is quoted at second hand from E. S. Shuckburgh's 1904 translation
(document_id `shuckburgh1904`), not newly translated here, because this transcription's own OCR of
this one sentence in `commdoc1852v3` is too corrupted to translate reliably — see the headnote above
and discrepancies.md item 1. Shuckburgh calls this "the nineteenth statute"; the original numbers it
XXI.] "I wish all... to understand, whether Fellows, scholars, or even pensioners, who are to be
admitted into the College, that the one object which I set before me in erecting this College was to
render as many as possible fit for the administration of the Divine Word and Sacraments; and that
from this seed-ground the English Church might have those that she can summon to instruct the people
and undertake the office of pastors, which is a thing necessary above all others. Therefore let
Fellows and scholars who obtrude into the College with any other design than to devote themselves to
sacred theology, and eventually to labour in preaching the Word, know that they are frustrating my
hope and occupying the place of Fellow or scholar contrary to my ordinance." [emmanuel-cambridge-q006]

[Continuing in the Abridger's own new translation, from the same chapter of `commdoc1852v3`:] ...we
ordain nevertheless that the Fellows of the said College hold one disputation in Theology every week,
in which each in his turn shall be the respondent, and there shall be two opponents; and each of the
Fellows shall likewise take that place in his turn, according to the custom observed in the other
Colleges. [emmanuel-cambridge-q025]

[...]

Out of the whole number of Fellows of the said College we will that at least the four most senior be
ministers of the Word and Sacraments, and be admitted to that order within one year from the day of
the publication of our statutes in the said College: and that as each of those ministers departs from
the College, the next Fellow in order of seniority be promoted, within six months of that minister's
departure, to the order of minister of the Word and Sacraments: so that there be never fewer than
four ministers in the said College. And whoever fails to become a minister of the Word and Sacraments
as here prescribed shall forfeit his right in the fellowship for ever. [emmanuel-cambridge-q026]

[...]

[Cap. XXXII — On the character and election of scholars.] That the nursery, then, may not be unfit
and unmanageable, nor such as can be handled only softly and indulgently, we will and ordain that the
choice of scholars be made from among those young men who are poorer, more upright, more able, and of
more outstanding character; who are of proven honesty, disposition, and good hope; who are neither
yet Bachelors of Arts nor admitted to the sacred ministry; who have set before themselves sacred
Theology and the holy ministry; and who are (at least moderately) instructed and skilled in Greek,
Rhetoric, and Logic: preferring, however, above all, the needy, provided they are equal in the other
conditions. For which reason we will that those especially who are natives of the counties of Essex
and Northampton be preferred; of whom we will that there always be two scholars in the College
itself; yet so that not more than three from any one of the aforesaid, or from any other county of
the whole of England, be held as scholars at one time. We will that the same method be observed in
choosing and examining the scholars as we have described in the Statutes concerning the election of
Fellows and those preceding it. [emmanuel-cambridge-q027]

[...]

[Cap. XXXIV — On the worship of God, scholastic exercises, and the conduct of scholars.] We will and
ordain that every scholar be occupied always, so far as it can be done, either in the worship of God,
or in the study of the liberal arts, or in the forming of outstanding character: on all Sundays and
on the other days of public prayer we will that each one, without any exception, attend the divine
offices in full in the chapel of the said College. [emmanuel-cambridge-q028]

[...]

[Cap. XXXVI — On what the scholars shall receive from the College and from others, and on their
absence.] We will and ordain that twelve pence a week be paid to the Steward for the commons of each
of them... and fourteen shillings and eightpence a year for clothing; the color of which clothing we
will to be fixed by the Master. We will also that they all sit together at dinner and supper in the
hall of the said College, at the tables assigned to them; that four, and no more, sleep in each
chamber... [emmanuel-cambridge-q029]

[...]

[Cap. XXXVIII — On making presentations to vacant churches.] [...] But we absolutely forbid that
those whom they shall present as pastors or rectors of churches be taken from anywhere other than the
College itself, since we do not doubt that it will suffice to supply men fit for this office.
[emmanuel-cambridge-q030]

[...]

[Closing dating clause.] Given the first day of October, in the year of our Lord one thousand five
hundred and eighty-five, and in the twenty-seventh year of the most illustrious Lady [Elizabeth,
Queen of England, France, and Ireland]. [emmanuel-cambridge-q031]

[...]

[Supplementary statute, *De Mora Sociorum* — on the Fellows' term of residence and on taking the
degree of Doctor of Sacred Theology. Quoted at second hand from E. S. Shuckburgh's 1904 translation
(document_id `shuckburgh1904`), for the same reason given above at Cap. XXI: this transcription's own
OCR of the supplementary statutes was not independently re-translated this session for the passage
below, though the underlying Latin was located and read (see dossier.md field F6 and discrepancies.md
item 3 on the date of the companion supplementary statute).] "We have founded the College with a
design that it should be seed-plot of learned men for the supply of the Church, and for the sending
forth of as large a number as possible of those who shall instruct the people in the Christian faith.
We would not have any Fellow suppose that we have given him in this College a perpetual abode — a
warning which we deem the more necessary in that we have ofttimes been present when many experienced
and wise men have taken occasion to lament, and have supported their complaints by past and present
instances, that in other Colleges a too protracted stay of Fellows has been no slight bane to the
common weal and to the interests of the Church." [emmanuel-cambridge-q020]
