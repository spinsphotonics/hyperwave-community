STATUS: TODO | IN_PROGRESS | DONE | BLOCKED: <reason>
TICKET: E-<slug>
ROLE: Extractor
INPUTS READ: <list every text/ file and sources.csv>

# Dossier — <Institution name> (`<slug>`)

Every field: one to three summary sentences (no quotation marks), then the supporting `quote_id`s in square brackets, then the confidence label. Empty = `NOT FOUND (searched: <document_ids>)` or `UNVERIFIED`. Field types: **PO** = PRIMARY-ONLY; **PS** = PRIMARY-OR-SECONDARY.

## F — Founding
- **F1 Name(s) of the institution, with dates of each name** (PS):
- **F2 Founding date(s): decision, opening, charter/incorporation** (PS):
- **F3 Founders and first officers (names, roles)** (PS):
- **F4 Place(s)** (PS):
- **F5 Principal founding document: title, date, language, document_id** (PO):
- **F6 Stated purpose, in the founders' words** (PO):
- **F7 Need or crisis the founders said they were answering** (PO):
- **F8 Doctrinal basis or confession, and who must subscribe** (PO):
- **F9 Antecedents the founders cite or imitate (earlier schools, documents, persons)** (PO):

## A — Admission
- **A1 Minimum and maximum age** (PO):
- **A2 Prior education required** (PO):
- **A3 Testimony of conversion, character, or church membership required** (PO):
- **A4 Entrance examination (subjects)** (PO):
- **A5 Women: admitted? in what capacity?** (PO):
- **A6 Prior experience required (e.g., preaching, service)** (PO):

## C — Curriculum
- **C1 Length of course** (PO):
- **C2 Subjects, in the order taught** (PO):
- **C3 Structure (classes, years, terms)** (PO):
- **C4 Languages taught, and level required** (PO):
- **C5 Set texts and authors** (PO):
- **C6 Preaching or practical instruction (form, frequency)** (PO):
- **C7 Examinations and their form** (PO):
- **C8 Degree, certificate, license, or ordination at completion** (PO):
- **C9 Weekly or daily timetable (table if given)** (PO):

## L — Common life
- **L1 Daily order: rising, worship, meals, study, bed** (PO):
- **L2 Residence: where students lived; with whom** (PO):
- **L3 Worship: form and frequency** (PO):
- **L4 Discipline: rules and penalties** (PO):
- **L5 Dress, conduct, recreation rules** (PO):
- **L6 Fees, board, and how students were supported** (PO):
- **L7 Number of students (with years)** (PS):
- **L8 Health, deaths, or hardship recorded among students** (PS):

## T — Teachers
- **T1 Number and titles of teachers (with years)** (PS):
- **T2 Teachers' duties as set out in the charter** (PO):
- **T3 Forms of teacher–student contact (lecture, disputation, tutorial, table, correspondence, preaching class)** (PO):
- **T4 Named examples of teacher–student interaction with source** (PS):
- **T5 Teacher:student ratio if computable (show the two numbers and years)** (PS):

## M — Emphasis
- **M1 What the founders said mattered most (their words)** (PO):
- **M2 Distinctive stress (doctrine, piety, preaching, languages, evangelism, faith principle, self-support, other)** (PO):
- **M3 What the founders explicitly rejected or warned against** (PO):
- **M4 Motto or watchword, with earliest source** (PS):
- **M5 Later leaders' restatements of the emphasis (with dates)** (PS):

## S — Sending
- **S1 Sending mechanism (licensing body, mission board, church commendation, self-sending)** (PO):
- **S2 Destinations (regions, countries)** (PS):
- **S3 Numbers sent, by period, with source** (PS):
- **S4 Support on the field (salary, faith principle, self-support)** (PO):
- **S5 Deaths, martyrdoms, or casualties recorded, with source** (PS):
- **S6 Instructions given to those sent (title, date, document_id)** (PO):

## R — Fruit
- **R1–R4 Up to six named alumni or missionaries: name, dates, field, one-sentence outcome, source** (PS):

## X — For the chapter
- **X1 Epigraph candidate (quote_id)**:
- **X2 Three strongest quotations for the Emphasis section (quote_ids)**:
- **X3 Any sections of the founding document that should be reprinted whole (with word counts)**:

## Discrepancies
See `discrepancies.md`. Count: <n>.
