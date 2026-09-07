# 04 — Task Tickets

Ticket ID format: `<series>-<slug>-<step>` for per-institution tickets; `<series>-<nnn>` for cross-cutting tickets.

Per-institution series, in strict order (each depends on the previous being DONE or, for V, VERIFIED):

| Step | Procedure | Role | Depends on | Output |
|---|---|---|---|---|
| S | Scout | Scout | C-001 | `research/<slug>/sources.csv`, `scout_notes.md` |
| F | Fetcher | Fetcher | S DONE | `research/<slug>/raw/`, `text/` |
| E | Extractor | Extractor | F DONE (or F BLOCKED with at least one text file) | `dossier.md`, `quotes.jsonl`, `discrepancies.md` |
| V1 | Verifier | Verifier | E DONE | `verification.md` (dossier) |
| B | Abridger | Abridger | V1 VERIFIED | `charter_abridged.md` |
| P | Rights | Coordinator/Human | V1 VERIFIED | `rights/permissions_log.csv` row(s) |
| D | Writer | Writer | B DONE and V1 VERIFIED | `chapters/<part>-<nn>-<slug>.md` |
| V2 | Verifier | Verifier | D DONE | `verification.md` (chapter section) |
| ED | Editor | Editor | V2 VERIFIED and X-001 DONE | edited chapter + editlog |
| H | Human sign-off | Human | ED DONE, P resolved | status ACCEPTED |

If V1 or V2 returns `RETURNED`, the preceding step (E or D) is reopened as `<step>-r2`, `<step>-r3`, and re-verified. Three failed rounds escalate to the Human.

## Setup tickets (C-series)

| ID | Role | Steps | Done when |
|---|---|---|---|
| C-001 | Coordinator | Create the directory layout from `00_README.md`; copy `book-plan/` into `book/plan/`; copy `templates/` files into place. | Directories exist; `ls book/` matches the layout. |
| C-002 | Coordinator | Generate `book/roster.csv` from the roster tables in `02_institution_roster.md` (columns: slug, name, founded, tier, part, chapter_order). Chapter order = founding date within part. | CSV has one row per slug; no duplicates; tiers filled. |
| C-003 | Coordinator | Generate `book/tickets.csv` by expanding every Tier A and Tier B slug through the step table above, plus the C, X, P, F series. Columns: ticket_id, slug, step, role, depends_on, status, assignee, output, notes. | Row count = (#TierA+#TierB) × 10 + cross-cutting count. All statuses TODO. |
| C-004 | Coordinator | Install the check script `templates/check_quotes.py` and confirm it runs on the sample in `templates/sample/`. | Script exits 0 on the sample; exits 1 when a quote is altered. |
| C-005 | Human | Confirm public-domain cutoff year for the publication year (see `06`). Record in `rights/permissions_log.csv` header. | Year recorded. |
| C-006 | Coordinator | Decide whether `abcfm` and `judson-triennial-convention` are one chapter or two (default: two, cross-referenced). Record in `roster.csv`. | Decision recorded. |

## Per-institution tickets

Tier A, in execution order (Scouts may run in parallel across institutions; the pipeline within one institution is serial):

Part I: `wittenberg`, `zurich-prophezei`, `strasbourg-academy`, `geneva-academy`, `scotland-first-book-of-discipline`, `emmanuel-cambridge`
Part II: `harvard-college`, `halle-francke`, `herrnhut-moravians`, `bristol-baptist-academy`, `log-college`, `college-of-new-jersey`
Part III: `baptist-missionary-society`, `london-missionary-society`, `church-missionary-society`, `andover-seminary`, `abcfm`, `judson-triennial-convention`, `princeton-seminary`, `basel-mission`, `serampore-college`, `pastors-college`, `southern-baptist-seminary`
Part IV: `china-inland-mission`, `east-london-institute`, `nyack-mti`, `moody-bible-institute`, `student-volunteer-movement`, `prairie-bible-institute`, `westminster-seminary`
Part V: `wycliffe-sil`, `wheaton-college`, `cmml-brethren`
Part VI: `westminster-chapel-fellowship`, `london-theological-seminary`, `omf`

Tier B (after all Tier A reach V1 VERIFIED): `leiden`, `saumur-sedan`, `yale-college`, `northampton-academy`, `trevecca`, `new-college-edinburgh`, `aim-sim`, `biola`, `glasgow-bti`, `columbia-bible-college`, `inter-varsity-urbana`, `new-tribes-mission`, `maf`, `evangelical-library-ifes-banner`

Tier C (appendix note only; one ticket, `S` and `E` only, reduced dossier): `modern-reformed-seminaries`

For each slug the ten tickets are, for example:

```
S-geneva-academy    Scout      depends C-001
F-geneva-academy    Fetcher    depends S-geneva-academy
E-geneva-academy    Extractor  depends F-geneva-academy
V1-geneva-academy   Verifier   depends E-geneva-academy
B-geneva-academy    Abridger   depends V1-geneva-academy=VERIFIED
P-geneva-academy    Rights     depends V1-geneva-academy=VERIFIED
D-geneva-academy    Writer     depends B-geneva-academy, V1
V2-geneva-academy   Verifier   depends D-geneva-academy
ED-geneva-academy   Editor     depends V2=VERIFIED, X-001
H-geneva-academy    Human      depends ED, P
```

### Institution-specific extra steps (added to the named ticket)

| Ticket | Extra step |
|---|---|
| S-geneva-academy | Nickname-origin search (Procedure S step 5). |
| E-geneva-academy | Extract from the *Registres de la Compagnie des Pasteurs* the list of pastors sent to France 1555–1562 with dates; record any noted deaths. Field S5. |
| E-zurich-prophezei | Record the daily order of the Prophezei (time, languages, sequence) as a table in field L1. |
| E-strasbourg-academy | Record Sturm's class structure (number of classes, years) in C3. |
| E-emmanuel-cambridge | Extract the statute that limits fellows' tenure to push them into parish ministry [VERIFY existence] into field S2. |
| E-harvard-college | Reprint *Rules and Precepts* in full as the charter (it is short); dossier field F5 = *New England's First Fruits*. |
| E-herrnhut-moravians | Extract the lot and the "Brotherly Agreement" articles on common life; record the 1732 sending in S1–S3. |
| E-log-college | Since no charter exists, F5 = Whitefield's 1739 journal description plus Alexander 1845 chapter 1; mark the chapter's Charter section as "Descriptions in lieu of a charter." |
| E-baptist-missionary-society | Charter = Serampore *Form of Agreement* (1805), whole. Also extract Carey's *Enquiry* section V (means) as a second quotation block. |
| E-andover-seminary | Extract the *Associate Creed* in full and the statutes on student piety; extract the Brethren society constitution if found. |
| E-abcfm | Charter = 1812 *Instructions* to Judson's party. Extract the ABCFM constitution's purpose clause separately. |
| E-judson-triennial-convention | Charter = 1814 Convention constitution. Extract Judson's letters to Baldwin/Bolles (1812) on his change of views. |
| E-princeton-seminary | Extract the *Plan*'s articles on "Design," "Admission," "Course of Study," "Devotion and Improvement in Practical Piety," and the professors' subscription formula. |
| E-pastors-college | Extract: admission rule (prior preaching, church testimony), free tuition, course length, George Rogers's role, Friday afternoon lecture, students' weekend preaching, numbers of men sent to the mission field and churches founded (from annual reports, with year). |
| E-southern-baptist-seminary | Extract the *Abstract of Principles* in full and Boyce's three changes as three quotations. |
| E-china-inland-mission | Extract each numbered principle of *Principles and Practice* as its own quote record; record the edition. Extract from *China's Spiritual Need and Claims* the appeal for twenty-four workers. Record Lammermuir party numbers and the mortality figures the sources give. |
| E-east-london-institute | Extract the Institute's stated aim, course, and the number trained and sent (with year and source). |
| E-nyack-mti | Extract Simpson's stated purpose for a "Missionary Training College" and the course outline. |
| E-moody-bible-institute | Extract Moody's "gap men" statement and the first prospectus's course and admission. |
| E-student-volunteer-movement | Extract the declaration card wording and the watchword with earliest source. |
| E-prairie-bible-institute | Extract the motto, the daily schedule, dating/social rules, and the "search question" method. Record Elisabeth Howard's attendance if the archive confirms it. |
| E-westminster-seminary | Extract Machen's opening address purpose paragraphs and the first catalogue's course of study. |
| E-wycliffe-sil | Extract the doctrinal statement and purpose; the Camp Wycliffe first-session description; the 1950 Norman summer course description. |
| E-wheaton-college | Extract the statement of faith, the 1947–49 catalog's requirements, Foreign Missions Fellowship aims; Jim Elliot's journal on his Wheaton training (short quotations only, rights). |
| E-cmml-brethren | Extract the Brethren principle of commendation and Groves's *Christian Devotedness* core passage; record the Elliot/Fleming commendation. |
| E-westminster-chapel-fellowship | Extract the Westminster Fellowship's aims and the Friday discussion format from Murray or primary minutes; record membership rule after 1966 if sources give it. |
| E-london-theological-seminary | Extract LTS aims (no degrees; two-year course; church testimony) from the 1977 prospectus and the opening address. Rights: MLJ Trust. |
| E-omf | Extract the post-1964 *Principles and Practice* clauses that changed from CIM's. |

## Cross-cutting tickets

| ID | Role | Depends on | Steps | Done when |
|---|---|---|---|---|
| X-001 | Extractor | All Tier A V1 VERIFIED | Procedure X1: comparative tables. | Six tables, every cell cited or marked NOT FOUND. |
| X-002 | Extractor | X-001 | Procedure X2: lineage chart. | Edge list and Mermaid diagram. |
| X-003 | Extractor | X-001 | Procedure X3: timeline. | Sorted, cited. |
| X-004 | Extractor | All Tier A D DONE | Procedure X4: glossary. | Every term cited. |
| X-005 | Extractor | X-001..X-003 | Procedure X5: introduction brief. | Ten items with row references. |
| X-006 | Human | X-005 | Write the Introduction. | 5,000–7,000 words. |
| X-007 | Writer | All chapters of a Part V2 VERIFIED | Draft the Part introduction from the chapters' Founding and Emphasis sections only. One ticket per Part (X-007-I … X-007-VI). | 1,500–2,500 words, footnoted. |
| P-000 | Human | C-005 | Confirm rights policy and who sends permission requests. | Recorded. |
| P-<slug> | Coordinator | V1 VERIFIED | Procedure P. | Row(s) in permissions_log. |
| F-001 | Coordinator | All ED DONE, all H ACCEPTED | Concatenate in order: front matter, Introduction, Parts (part intro + chapters), Appendices A–F. | `assembly/manuscript.md` exists; TOC matches. |
| F-002 | Editor | F-001 | Global checks: every footnote resolves; every slug appears in the lineage chart and timeline; forbidden words zero; names consistent. | Check report with zero errors. |
| F-003 | Verifier | F-002 | Random audit: pick 5% of all quotations across the manuscript (minimum 60) and re-verify against `text/` files. | Audit report; any FAIL reopens the chapter. |
| F-004 | Human | F-003 | Final acceptance against `07_risk_register_and_qa.md` final checklist. | Signed. |

## Effort estimate (for scheduling cheap models)

| Step | Typical model calls | Typical wall time |
|---|---|---|
| S | 10–40 searches | 1–3 hours |
| F | 1 per document | 1–4 hours (OCR bound) |
| E | 1 per 30 pages of text, plus assembly | 2–8 hours |
| V1 | 1 per 20 quotes | 1–3 hours |
| B | 2–3 | 1–2 hours |
| D | 3–5 | 1–2 hours |
| V2 | 2–4 | 1–2 hours |
| ED | 1–2 | under 1 hour |

Tier A: 36 institutions × 10 steps ≈ 360 tickets. Tier B: 14 × 10 = 140. Cross-cutting ≈ 25. Total ≈ 525 tickets.
