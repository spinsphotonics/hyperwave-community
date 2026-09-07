STATUS: DONE
TICKET: E-emmanuel-cambridge
ROLE: Extractor
INPUTS READ: text/commdoc1852v3.txt (Documents Relating to the University and Colleges of Cambridge,
vol. III, 1852 — the 1584 Charter and 1585 Statutes of Emmanuel College); text/shuckburgh1904.txt
(E. S. Shuckburgh, Emmanuel College, 1904); text/fuller1655.txt (Thomas Fuller, The History of the
University of Cambridge, 1655/1840); text/vch1959.txt (Victoria County History, Cambridgeshire vol.
3, 1959, "The colleges and halls: Emmanuel"); sources.csv

# Dossier — Emmanuel College, Cambridge (`emmanuel-cambridge`)

Note on sourcing: this session located and read the actual 1585 Latin Statutes of Emmanuel College
(and the 1584 Latin royal Charter) in a 19th-century scholarly reprint (commdoc1852v3), which
identifies its source as a manuscript copy in the British Museum (Sloane 1739). Passages copied
directly from that reprint are cited PO (primary) below. The Charter itself is in heavily abbreviated
Tudor legal Latin and its OCR is badly corrupted; it is NOT quoted directly anywhere in this dossier
(see charter_abridged.md). The body of the Statutes (Cap. I onward) is legible prose Latin and is
quoted directly in several fields, each with a new literal English translation marked
`[New translation, draft]` in quotes.jsonl (`translation_source: NEW-DRAFT`), since no freely
available published translation was located this session (Frank Stubbings's 1983 English translation
exists but is copyrighted and was not fetched — see sources.csv row `stubbings1983`). Where
Shuckburgh (1904) or the Victoria County History (1959) quote the founder's own statutes in English
translation, those quotations are also cited PO (quoted-at-second-hand), following the same
convention used in `book/research/geneva-academy/dossier.md` for Borgeaud's quotations of primary
Genevan records. Shuckburgh's, Fuller's, and the VCH's own narrative sentences are cited PS
(secondary).

## F — Founding
- **F1 Name(s) of the institution, with dates of each name** (PS): The institution was founded and
  named "Emmanuel College" from the outset in 1584; no evidence of an earlier or later name was
  found. [emmanuel-cambridge-q002]
- **F2 Founding date(s): decision, opening, charter/incorporation** (PO/PS): The royal charter
  empowering Sir Walter Mildmay to found a college is dated 11 January 1583/84; his own deed of
  foundation is dated 25 May 1584; the first admissions are recorded from 1 November 1584; and
  Mildmay's own Statutes for governing the College are dated, and were solemnly given under his hand
  and seal, 1 October 1585 ("the twenty-seventh year" of Elizabeth I). [emmanuel-cambridge-q004,
  emmanuel-cambridge-q005, emmanuel-cambridge-q031]. See discrepancies.md item 5 for the several
  dates that can be conflated under "founding."
- **F3 Founders and first officers (names, roles)** (PO/PS): Sir Walter Mildmay (Chancellor of the
  Exchequer to Elizabeth I) was the sole founder. The College's first Master was Laurence Chaderton,
  B.D., late Fellow of Christ's College; the three original Fellows were Charles Chadwick (of
  Christ's), William Jones and Laurence Pickering (both of Clare Hall); the four original scholars
  were John Duke, Richard Rolffe, John Starkye, and Robert Houghton; four further Fellows were
  admitted within the first year. [emmanuel-cambridge-q003, emmanuel-cambridge-q002]. Chaderton was
  "well versed in Greek, Latin, and Hebrew," a Calvinist and preacher whom Mildmay judged "exactly
  the man" to build the new College and give it "the right direction in theology."
  [emmanuel-cambridge-q021]
- **F4 Place(s)** (PS): Cambridge, on the site of a former Dominican ("Black Friars"/"Preaching
  Friars") convent. [emmanuel-cambridge-q002]
- **F5 Principal founding document: title, date, language, document_id** (PO): Mildmay's Statutes for
  Emmanuel College, dated 1 October 1585, in Latin. document_id: `commdoc1852v3` (a 19th-century
  scholarly reprint of a British Museum manuscript copy; the original manuscript itself was not seen
  this session). [emmanuel-cambridge-q031, emmanuel-cambridge-q004, emmanuel-cambridge-q034]. See
  discrepancies.md item 1 on the differing statute-numbering given by Shuckburgh (1904) versus the
  Victoria County History (1959) and this session's own reading of the original.
- **F6 Stated purpose, in the founders' words** (PO): In the statute numbered XXI in the original text
  (Shuckburgh calls it the "nineteenth statute" — see discrepancies.md item 1), Mildmay wrote that
  "the one object which I set before me in erecting this College was to render as many as possible
  fit for the administration of the Divine Word and Sacraments; and that from this seed-ground the
  English Church might have those that she can summon to instruct the people and undertake the office
  of pastors, which is a thing necessary above all others," warning that Fellows or scholars who came
  "with any other design than to devote themselves to sacred theology, and eventually to labour in
  preaching the Word" would be "frustrating my hope." [emmanuel-cambridge-q006, corroborated in the
  Victoria County History's own translation of the same statute at emmanuel-cambridge-q033]. In a
  later supplementary statute (*De Mora Sociorum*, 1587 — see discrepancies.md item 3 on its date),
  Mildmay restated the same aim: "We have founded the College with a design that it should be
  seed-plot of learned men for the supply of the Church, and for the sending forth of as large a
  number as possible of those who shall instruct the people in the Christian faith."
  [emmanuel-cambridge-q020] The Statutes' own Preface (Praefatio), preceding the numbered chapters,
  frames the same purpose theologically: schools and colleges are "an ancient institution in the
  Church," on the model of the "sons of the prophets" trained at Naioth, Gilgal, Bethel, and Jericho
  "by the greatest and most renowned prophets, Samuel, Elijah, and Elisha," and of the Jerusalem
  synagogues at which "Saul of Tarsus... is said to have sat at the feet of... Gamaliel"; such
  schools are "nurseries of Theology," likened to the Levites' guardianship of the altar fire, "lest
  we bring before the Lord, to be kindled, a strange fire — that is, Popery and the other heresies."
  [emmanuel-cambridge-q039, translation new-draft]
- **F7 Need or crisis the founders said they were answering** (PS): The Victoria County History
  states that Mildmay's "chief aim in establishing the College was to provide for a perpetual supply
  of educated clergy for the reformed church in a University where the study of theology had much
  decayed." [emmanuel-cambridge-q032]
- **F8 Doctrinal basis or confession, and who must subscribe** (PO): A newly elected Master was
  required to swear an oath that he would "embrace" "the true religion of Christ" as "contrary to
  Popery and other heresies," to set "the authority of Scripture above the judgments of even the best
  of men," and to hold the royal authority supreme and "in no way subject to the jurisdiction of any
  foreign bishops, princes, or powers." [emmanuel-cambridge-q023, translation new-draft]. Every newly
  elected Fellow swore a similarly worded oath against "Popery and all other heresies" and to observe
  the founder's statutes. [emmanuel-cambridge-q024, translation new-draft]. The Victoria County
  History corroborates: "The Master took an oath against 'Popery and other heresies'... At least four
  of the fellows were to be 'Ministers of the Word and Sacraments'... the College was not to have two
  fellows from the same county at the same time." [emmanuel-cambridge-q035]
- **F9 Antecedents the founders cite or imitate (earlier schools, documents, persons)** (PO/PS): No
  passage read this session has Mildmay naming an earlier *English college or its statutes* as his
  model, though the Victoria County History states, as an editorial note rather than a quotation from
  Mildmay: "The original statutes, dated 1 Oct. 1585... closely follow those of Christ's, the
  founder's own college." [emmanuel-cambridge-q034] Mildmay had himself been a benefactor of Christ's
  College, Cambridge, before founding Emmanuel. [background from F3/shuckburgh1904, not separately
  quoted] Mildmay's own Statutes, however, do open by naming explicit biblical antecedents for the
  idea of a school of this kind: the Preface invokes "the sons of the prophets" trained at Naioth,
  Gilgal, Bethel, and Jericho "by the greatest and most renowned prophets, Samuel, Elijah, and
  Elisha," the synagogues of Jerusalem recorded in Acts, and Saul of Tarsus at "the feet of...
  Gamaliel," as the pattern Emmanuel imitates: "so many men of heroic virtue among our ancestors,
  imitating these divine and ancient institutions of God's prophets, have set up colleges." Mildmay
  names no particular later college among those "men of heroic virtue" he says he imitates; only the
  biblical pattern is named explicitly. [emmanuel-cambridge-q039, translation new-draft]

## A — Admission
- **A1 Minimum and maximum age** (PO): NOT FOUND (searched: commdoc1852v3, shuckburgh1904, fuller1655,
  vch1959) for the founding era. (Shuckburgh 1904, p. 144-145, separately records that in the 1730s-
  1760s — nearly a century and a half after the founding — "the usual age for admission was seventeen
  or eighteen"; this is NOT founding-era evidence and is not used to answer A1. Not separately quoted
  in quotes.jsonl since it falls outside the founding period this dossier otherwise covers.)
- **A2 Prior education required** (PO): Candidates for scholarships were required to be "(at least
  moderately) instructed and skilled in Greek, Rhetoric, and Logic." [emmanuel-cambridge-q027,
  translation new-draft]
- **A3 Testimony of conversion, character, or church membership required** (PO): No personal
  conversion-narrative requirement was found. The statutes required scholars to be chosen from young
  men "who are poorer, more upright, more able, and of more outstanding character; who are of proven
  honesty, disposition, and good hope." [emmanuel-cambridge-q027, translation new-draft] Doctrinal
  subscription (the oath against "Popery and other heresies") was required of Fellows and the Master
  on admission to those offices. [emmanuel-cambridge-q023, emmanuel-cambridge-q024, F8]
- **A4 Entrance examination (subjects)** (PO): The statutes prescribe that scholars be chosen and
  examined "in the same method... as we have described in the Statutes concerning the election of
  Fellows" — i.e. an oral examination before the Master and Fellows, followed by a secret ballot.
  [emmanuel-cambridge-q027, translation new-draft]
- **A5 Women: admitted? in what capacity?** (PO): NOT FOUND (searched: commdoc1852v3, shuckburgh1904,
  fuller1655, vch1959).
- **A6 Prior experience required (e.g., preaching, service)** (PO): Candidates for scholarships were
  required to be "neither yet Bachelors of Arts nor admitted to the sacred ministry" — i.e. not yet
  advanced in either the University's or the Church's own ranks. [emmanuel-cambridge-q027, translation
  new-draft] The founder's purpose-statute likewise addressed those "who are to be admitted into the
  College" as not yet holding any prior standing beyond their intention to enter "sacred theology."
  [emmanuel-cambridge-q006]

## C — Curriculum
- **C1 Length of course** (PO): NOT FOUND as an explicit stated number of years for scholars or
  Fellows (searched: commdoc1852v3, shuckburgh1904). The statutes tie a scholar's tenure to University
  progress rather than a fixed term: "no scholar was to be kept longer... after he was once, in the
  course of his years, entitled to take the degree of Master of Arts" (Shuckburgh's paraphrase of the
  relevant statute, not separately quoted here as a verbatim passage). Fellows' tenure was separately
  capped by the *De Mora Sociorum* statute at effectively about seven years from the M.A. (through the
  requirement to vacate on taking the D.D. degree, or within a year of eligibility to do so).
  [emmanuel-cambridge-q020]
- **C2 Subjects, in the order taught** (PO): A later College order (1598-1600, still within living
  memory of the founding) required "every Scholler under the degree of a Bachelour of Arte" to attend
  the Greek Lecturer, alongside the existing Head Lecturer's "morning lectures." Undergraduates nearing
  their degree ("Questionists") were examined on set books of logic and philosophy (see C5).
  [emmanuel-cambridge-q007]
- **C3 Structure (classes, years, terms)** (PS): Teaching was organized under a Head Lecturer, a Greek
  Lecturer, and a Lecturer in Logic, each examining his own class and empowered to punish defaulters;
  religious instruction was given by a Catechist, who both lectured in chapel and publicly questioned
  the students; discipline was in the hands of the Master and an annually appointed Dean; each
  undergraduate had a Tutor, chosen from among the Fellows (the single-College-Tutor system had "not
  then begun"). [emmanuel-cambridge-q008 (partially); background summary from shuckburgh1904 p.30,
  not separately quoted beyond q008's context]
- **C4 Languages taught, and level required** (PO): Greek was required of scholarship candidates,
  alongside Rhetoric and Logic. [emmanuel-cambridge-q027, translation new-draft] The first Master,
  Chaderton, was himself "well versed in Greek, Latin, and Hebrew." [emmanuel-cambridge-q021]
- **C5 Set texts and authors** (PO): The books "necessary" for undergraduates approaching examination
  ("Questionists," to "sit to be opposed") were "Ramus' Logick, Aristotle's Organo[n], Ethics,
  Politiques and Physiques, and if they can or will they may read Phrigius his Natural Philosophia."
  [emmanuel-cambridge-q007]
- **C6 Preaching or practical instruction (form, frequency)** (PO): The statutes required that "the
  Fellows of the said College hold one disputation in Theology every week, in which each in his turn
  shall be the respondent, and there shall be two opponents," each Fellow filling that role in turn
  "according to the custom observed in the other Colleges." [emmanuel-cambridge-q025, translation
  new-draft] Separately, the statutes required that at least the four most senior Fellows be admitted,
  within a set time, to the order of "minister of the Word and Sacraments," with the College bound to
  keep never fewer than four Fellows in that order at any time — those who failed to be so admitted
  forfeiting their Fellowship. [emmanuel-cambridge-q026, translation new-draft] The Victoria County
  History corroborates: "At least four of the fellows were to be 'Ministers of the Word and
  Sacraments.'" [emmanuel-cambridge-q035]
- **C7 Examinations and their form** (PO): Election to a Fellowship or scholarship was by the same
  method: an oral examination and a secret written ballot among the Master and Fellows present, each
  writing "I, N., elect N. T. as a fellow of this College" (Shuckburgh/commdoc content, per
  emmanuel-cambridge-q027's cross-reference to "the Statutes concerning the election of Fellows";
  the ballot mechanics themselves are described in the same chapter of the statutes as the oaths at
  emmanuel-cambridge-q024, part of the same statute).
- **C8 Degree, certificate, license, or ordination at completion** (PO): No University degree was
  itself conferred by the College; scholars remained only until eligible, "in the course of their
  years," for the M.A., and Fellows were required to proceed to the D.D. within a set time or forfeit
  their place. [emmanuel-cambridge-q020, emmanuel-cambridge-q026] What the statutes explicitly provide
  for on completion is not a certificate but the presentation of the College's own men, by the College
  itself, to livings in its patronage (see S1).
- **C9 Weekly or daily timetable (table if given)** (PO): No table survives in the material read this
  session, but individual timed elements are recorded in the College order book (quoted by
  Shuckburgh): dinner was at 11 a.m., supper at 6 p.m.; the buttery was open 8-8.30 a.m., briefly
  before 3 p.m., and 7-8 p.m.; the College gates were shut at ten, after which being out was an
  offence; and every undergraduate and Bachelor was required to attend his Tutor's prayers "everie
  night at eight of the clock." [emmanuel-cambridge-q009, emmanuel-cambridge-q008]

## L — Common life
- **L1 Daily order: rising, worship, meals, study, bed** (PO): See C9 for the timed elements
  (dinner 11 a.m., supper 6 p.m., gates shut at 10 p.m., nightly Tutor's prayers at 8 p.m.).
  [emmanuel-cambridge-q009, emmanuel-cambridge-q008] Undergraduates were required to be "at work 'in
  the calling' at all times in the day except certain hours of recreation" (one hour after dinner and
  after supper) "unless some publique exercise of learning or religion doe require their presence"
  (Shuckburgh's quotation of the order book, embedded in the same passage as q008/q009's surrounding
  context; not separately broken into its own quote record).
- **L2 Residence: where students lived; with whom** (PO): The statutes required that scholars "sleep
  four, and no more, in each chamber." [emmanuel-cambridge-q029, translation new-draft] Shuckburgh
  states, as narrative, that "all students had to live within the College walls."
  (context of emmanuel-cambridge-q014, not separately quoted as its own record)
- **L3 Worship: form and frequency** (PO): The statutes required every scholar to "attend the divine
  offices in full in the chapel of the said College" "on all Sundays and on the other days of public
  prayer... without any exception." [emmanuel-cambridge-q028, translation new-draft] In practice, from
  the first, Emmanuel's own worship diverged markedly from other Cambridge colleges: "All other
  Colledges in Cambridge do strictly observe... the form of public prayer prescribed in the Communion
  Booke. In Emm. Colledge they do follow a private course of public prayer, after their own fashion,
  both Sondaies, Holy daies, and workie daies," a divergence confirmed at the 1604 Hampton Court
  Conference, where "Mr. Chaderton was told of sitting Communions in Emmanuel College; which hee said
  was so by reason of the seats so placed as they be; yet that they had some kneeling also."
  [emmanuel-cambridge-q012, emmanuel-cambridge-q013] The Victoria County History corroborates that
  "attendance at daily prayers and the eucharist was compulsory." [emmanuel-cambridge-q035]
- **L4 Discipline: rules and penalties** (PO): Graver offences were punished by the Master "admonishing"
  the offender in the Dean's presence; forty-four such admonitions are recorded across Chaderton's
  38-year mastership (1584-1622), "an average of little more than one a year," ranging from a 1586
  case of "disobedience to the Master and unreverent abusing of the Fellows" and "bending a stone bowe
  and charging it with a pellet," through later cases of "drunkenness and spending 18d in ale,"
  "riding the horses in Mr. Wolfe's close," and "being taken by the Proctor in a scandalous house."
  [emmanuel-cambridge-q010, emmanuel-cambridge-q011]
- **L5 Dress, conduct, recreation rules** (PO): The statutes required scholars to "prefer modesty of
  countenance, bodily bearing, and kind and form of dress" (from the same chapter as
  emmanuel-cambridge-q028's context; the specific clause on dress and bearing was read but not
  separately extracted as its own quote record — see charter_abridged.md, Cap. XXXIV, for the fuller
  Latin text). A fixed annual clothing allowance of "fourteen shillings and eightpence," in a colour
  fixed by the Master, was provided to scholars. [emmanuel-cambridge-q029, translation new-draft]
- **L6 Fees, board, and how students were supported** (PO): Scholars received twelve pence a week
  toward their commons, paid to the Steward, plus the annual clothing allowance above.
  [emmanuel-cambridge-q029, translation new-draft] The first recorded stipend list (1585) shows the
  Master receiving £15 for half a year, six named Fellows each receiving about £4 (plus an additional
  16s 8d each), and "Eighteen scholars, each" receiving £1 13s 4d. [emmanuel-cambridge-q015]
- **L7 Number of students (with years)** (PS): In the first year and a half there were 76 admissions;
  20 in 1586; then "from twenty-seven to forty-two" annually until 1611, when admissions rose to 58,
  climbing to a peak of 79 in 1622 (Chaderton's last year as Master). [emmanuel-cambridge-q014] The
  Victoria County History gives annual admissions in the years immediately after the founding as
  averaging "between 30 and 40," and for 1586-1611 gives the range as "27 and 44" (not 42 — see
  discrepancies.md item 2), rising to a peak of "79 in 1622-3," with 1624-5's 74 matriculations "the
  highest of any college in the University" that year. [emmanuel-cambridge-q036,
  emmanuel-cambridge-q037]
- **L8 Health, deaths, or hardship recorded among students** (PS): NOT FOUND for the founding-era
  College itself as a matter of collective hardship (searched: commdoc1852v3, shuckburgh1904,
  vch1959). The one individual death recorded in the material read is John Harvard's, in New England
  after emigration, of consumption — see R1.

## T — Teachers
- **T1 Number and titles of teachers (with years)** (PS): At the founding (1585 stipend list): a
  Master, and Fellows serving also as "Lecturer" and "Catechist" (named: Chadwick, Fellow and
  Lecturer; Jones, Fellow and Catechist; also Pickering, Cock, Richardson, Gilbie, Branthwaite — seven
  Fellows named on the list in total). [emmanuel-cambridge-q015] The statutes required at least four
  of the Fellows to hold the further title/order of "minister of the Word and Sacraments."
  [emmanuel-cambridge-q026, translation new-draft] The Victoria County History corroborates: "The
  first list of stipends belongs to 1585 and records payments to the Master and seven fellows."
  [emmanuel-cambridge-q036] Chaderton himself was chosen as "exactly the man" — "a Calvinist, a
  preacher, accomplished in the three languages Latin, Greek, and Hebrew." [emmanuel-cambridge-q021]
- **T2 Teachers' duties as set out in the charter** (PO): The Statutes' first chapter gives the
  Master "authority over all the fellows and scholars of the same College, to govern them, to rule
  them, to punish them, to admonish them, and to administer the domestic affairs of the whole
  College." [emmanuel-cambridge-q040, translation new-draft] Lecturers "examined" their own classes
  and "could punish the youths when they offend"; the Catechist "not only lectured in the chapel, but
  questioned the students publicly." (context surrounding emmanuel-cambridge-q007/q008, drawn from the
  same order-book passage; not separately extracted as its own quote record beyond what q007/q008
  already capture)
- **T3 Forms of teacher-student contact (lecture, disputation, tutorial, table, correspondence,
  preaching class)** (PO): Weekly theological disputation among the Fellows themselves (respondent and
  two opponents, each Fellow rotating through the role). [emmanuel-cambridge-q025, translation
  new-draft] Nightly Tutor's prayers, at which "everie Tutor may see him [his pupil] at that time and
  be answerable for their good behaviour." [emmanuel-cambridge-q008] Lectures given by the Head
  Lecturer, Greek Lecturer, and Logic Lecturer, each with authority to examine and punish his own
  pupils. (context of emmanuel-cambridge-q007/q008)
- **T4 Named examples of teacher-student interaction with source** (PS): On King James I's visit to
  Cambridge in 1615, when someone pointed out to the King that Emmanuel's chapel was "far out of the
  eastward position," Chaderton "remarked to the King that he had been told that the same was true of
  the royal chapel at Whitehall," to which the King answered, "God will not turn away his face from the
  prayers of any holy and pious man, to whatever region of heaven he directs his eyes. So, doctor, I
  beg you to pray for me." [emmanuel-cambridge-q022]
- **T5 Teacher:student ratio if computable (show the two numbers and years)** (PS): For 1585 (the
  first recorded stipend list): 7 named Fellows to 18 named scholars, i.e. roughly 1:2.6 — not
  counting Fellow-Commoners or pensioners, whose numbers are not given on this particular list.
  [emmanuel-cambridge-q015]

## M — Emphasis
- **M1 What the founders said mattered most (their words)** (PO): "The one object which I set before
  me in erecting this College was to render as many as possible fit for the administration of the
  Divine Word and Sacraments." [emmanuel-cambridge-q006] Restated in the *De Mora Sociorum* statute:
  "We have founded the College with a design that it should be seed-plot of learned men for the supply
  of the Church, and for the sending forth of as large a number as possible of those who shall
  instruct the people in the Christian faith." [emmanuel-cambridge-q020] According to Thomas Fuller
  (1655), when Elizabeth I told Mildmay she had heard "you have erected a puritan foundation," he
  replied: "far be it from me to countenance any thing contrary to your established laws; but I have
  set an acorn, which, when it becomes an oak, God alone knows what will be the fruit thereof."
  [emmanuel-cambridge-q001 — SECONDARY, earliest source located is Fuller, 66 years after the events;
  see discrepancies.md item 4]
- **M2 Distinctive stress (doctrine, piety, preaching, languages, evangelism, faith principle,
  self-support, other)** (PO): Doctrinal subscription against "Popery and other heresies"
  [emmanuel-cambridge-q023, emmanuel-cambridge-q024, emmanuel-cambridge-q035] combined with a
  structural requirement that a fixed minimum of Fellows actually be admitted to the preaching
  ministry [emmanuel-cambridge-q026] and a weekly practice of theological disputation
  [emmanuel-cambridge-q025]. Emmanuel's own worship "after their own fashion" departed from the
  Prayer-Book form used elsewhere in Cambridge. [emmanuel-cambridge-q012]
- **M3 What the founders explicitly rejected or warned against** (PO): Per Fuller's account, Mildmay
  denied "countenanc[ing] any thing contrary to your established laws" even while defending the
  Puritan tendency of his foundation. [emmanuel-cambridge-q001 — SECONDARY, see M1] Per the founder's
  own statute, Fellows or scholars who came to the College "with any other design than to devote
  themselves to sacred theology, and eventually to labour in preaching the Word" were warned they
  would be "frustrating my hope." [emmanuel-cambridge-q006]
- **M4 Motto or watchword, with earliest source** (PS): NOT FOUND (searched: commdoc1852v3,
  shuckburgh1904, fuller1655, vch1959). (Shuckburgh 1904 records the door inscription "Sacrae
  Theologiae Studiosis posuit Gualterus Mildmaius, Ano Domini 1584" — "Walter Mildmay set this up for
  students of Sacred Theology, in the year of our Lord 1584" — over the original gate; this is a
  dedicatory inscription, not stated in the sources as a motto, and is not separately quoted as its
  own record.)
- **M5 Later leaders' restatements of the emphasis (with dates)** (PO): In 1627, when the Fellows
  petitioned the King to abolish the *De Mora Sociorum* statute limiting Fellows' tenure, Chaderton
  (by then retired as Master but still living in Cambridge) wrote in its defence: "I am fully
  persuaded, upon divers speeches which I have heard at severall tymes from him, that he would rather
  not have founded the College, than have omitted this statute. Other statutes were suggested to him
  by friends, but this proceeded soly from himself, and he took more time of deliberation to compose
  and sett downe this statute than any other." [emmanuel-cambridge-q019]

## S — Sending
- **S1 Sending mechanism (licensing body, mission board, church commendation, self-sending)** (PO):
  The statutes bound the College, whenever a church in its own patronage fell vacant, to choose and
  present a pastor to the bishop "from the College itself," and explicitly forbade drawing such
  presentees from anywhere else: "we absolutely forbid that those whom they shall present as pastors
  or rectors of churches be taken from anywhere other than the College itself, since we do not doubt
  that it will suffice to supply men fit for this office." [emmanuel-cambridge-q030, translation
  new-draft] The Victoria County History lists specific livings that came under the College's
  patronage in this way from the founding years: Stanground with Farcet (given 1588 by the founder),
  Little Melton (1584, Francis Chamberlain), Thurcaston (1585, Sir Francis Walsingham), and (1586)
  Loughborough, North Cadbury, Aller, and Puddletown (Henry, Earl of Huntingdon — though title to some
  of these was later lost in a legal dispute). [emmanuel-cambridge-q038]
- **S2 Destinations (regions, countries)** (PS): English parishes under the College's own patronage
  (see S1) and New England: Professor Franklyn B. Dexter (Yale), in an 1880 paper to the Massachusetts
  Historical Society on English university influence on New England, found that Emmanuel supplied "the
  largest contingent (21)" of any Oxford or Cambridge college among the Oxbridge-educated settlers he
  traced, including Thomas Hooker and John Cotton (both Emmanuel Fellows) and John Harvard.
  [emmanuel-cambridge-q016]
- **S3 Numbers sent, by period, with source** (PS): 21 identified New England settlers/ministers, per
  Dexter's 1880 paper as quoted by Shuckburgh (1904) — "the largest contingent" of any English
  university college. [emmanuel-cambridge-q016] No comparable count was found for ordinary English
  parish placements (searched: commdoc1852v3, shuckburgh1904, vch1959).
- **S4 Support on the field (salary, faith principle, self-support)** (PO): NOT FOUND (searched:
  commdoc1852v3, shuckburgh1904, vch1959). The statutes address the College's presentation of pastors
  to livings in its own gift (S1) but no clause specifying stipend or support for those it sent to
  other livings, or to New England, was located.
- **S5 Deaths, martyrdoms, or casualties recorded, with source** (PS): No martyrdom or persecution
  death connected with sending is recorded in the material read this session. The one relevant death
  recorded is John Harvard's: he emigrated to Massachusetts in 1637 and "died of consumption" on 14
  September 1638, aged about thirty-one; his brother Thomas had died of the same disease the year
  Harvard sailed. [emmanuel-cambridge-q017, emmanuel-cambridge-q018] This is emigration and natural
  death, not a casualty of a sending mechanism in the sense the roster's other institutions (e.g.
  Geneva's pastors sent into France) illustrate; it is recorded here for completeness, not as
  evidence of a "school of death" pattern at Emmanuel.
- **S6 Instructions given to those sent (title, date, document_id)** (PO): NOT FOUND (searched:
  commdoc1852v3, shuckburgh1904, vch1959).

## R — Fruit
- **R1 John Harvard (1607-1638)** (PS): Entered Emmanuel 17 April 1627; took his M.A. in 1635;
  emigrated to Massachusetts Bay, was admitted a townsman of Charlestown 6 August 1637, and there
  "ministered in the 'First Church,'" though "there is no record of his having been ordained." He died
  of consumption 14 September 1638, aged about thirty-one, "leaving one half of his estate (£779 17s.
  2d.), with a library of 320 volumes, for the College" then being founded at New Town (Cambridge),
  Massachusetts — the college that took his name. [emmanuel-cambridge-q017, emmanuel-cambridge-q018]
- **R2 Thomas Hooker (entered Emmanuel 1596, Fellow)** (PS): Listed by Dexter (1880) among the 21
  Emmanuel men who settled in New England. [emmanuel-cambridge-q016] (Hooker's own subsequent career
  founding Connecticut is standard historical knowledge but was not itself quoted from a source read
  this session; not stated as more than his Emmanuel entry and New England settlement.)
- **R3 John Cotton (entered Emmanuel 1617, B.D., Fellow)** (PS): Listed by Dexter (1880) among the 21
  Emmanuel men who settled in New England. [emmanuel-cambridge-q016]
- **R4 William Branthwaite and John Richardson (Fellows, early 1580s cohort)** (PS): Both remained in
  Cambridge; Branthwaite proceeded D.D. in 1598 and was "one of the Revisers of the Bible in
  1607-1611," later Master of Caius; Richardson was appointed Regius Professor of Divinity in 1607,
  Master of Peterhouse in 1609, and Master of Trinity in 1615. Chaderton himself, with these two and
  Samuel Ward, was "amongst those responsible for the Authorized Version of the Bible (1611)."
  (drawn from shuckburgh1904 and corroborated by vch1959's own list of the same four KJV
  translators — this specific composite fact was read across both sources but is not separately
  extracted as a single verbatim quote record; the individual facts of D.D. 1598, Regius Professor
  1607, and KJV translation are standard institutional-history statements in both texts rather than
  quoted founder's or eyewitness language, and are therefore given here as a PS summary rather than
  padded with a quote_id it does not need.)

## X — For the chapter
- **X1 Epigraph candidate**: emmanuel-cambridge-q006 (the founder's own purpose-statute, "the one
  object which I set before me... to render as many as possible fit for the administration of the
  Divine Word and Sacraments") is the strongest candidate for stating the founder's purpose in his own
  words. emmanuel-cambridge-q001 (the acorn exchange) is flagged `epigraph_candidate: true` in
  quotes.jsonl as a secondary, more memorable alternative, but per Rule 3 an epigraph drawn from a
  SECONDARY source (Fuller, not Mildmay's own document) is a weaker choice than F6's PRIMARY statute
  language; the Writer should prefer emmanuel-cambridge-q006.
- **X2 Three strongest quotations for the Emphasis section**: emmanuel-cambridge-q006,
  emmanuel-cambridge-q020, emmanuel-cambridge-q026.
- **X3 Sections of the founding document that should be reprinted whole (with word counts)**: The
  admission statute for scholars (Cap. XXXII, ~150 words in the original Latin) and the sending/
  presentation-to-livings statute (Cap. XXXVIII, the operative sentence quoted at
  emmanuel-cambridge-q030 is short, under 40 words) are both short, legible, and self-contained enough
  to reprint whole in charter_abridged.md; see that file for the full abridgement.

## Discrepancies
See `discrepancies.md`. Count: 5.
