# Execution Plan: "The School of Death"

**Working title:** *The School of Death: Charters of Protestant Seminaries and Missionary Societies, from Calvin's Geneva to the Faith Missions*

**What this plan is.** A complete, step-by-step plan for producing a book that collects, abridges, and explains the founding documents ("charters") of the Reformation and Protestant schools and missionary societies that trained and sent men and women into the ministry and to foreign fields. The line runs from Calvin's Geneva Academy (nicknamed the "school of death" because so many of its graduates were sent into France and killed) through the Puritan colleges, Andover and Adoniram Judson, Spurgeon's Pastors' College, Hudson Taylor's China Inland Mission, the Bible institutes, Martyn Lloyd-Jones and the London Theological Seminary, and the mid-century missions that sent Jim and Elisabeth Elliot.

**Who this plan is for.** It is written so that inexpensive, low-reasoning AI models (or junior human assistants) can execute it. Every task is broken into small tickets with numbered mechanical steps, fixed input files, fixed output files, and a "done when" checklist. No ticket asks the executor to make a judgment call that is not spelled out. Where judgment is unavoidable, the ticket says "STOP and flag for human review."

## Files in this plan

| File | Purpose | Who reads it |
|---|---|---|
| `00_README.md` | This file. Rules, roles, directory layout, ticket lifecycle. | Everyone, before every ticket |
| `01_book_design.md` | Thesis, audience, part/chapter structure, chapter template, style guide, rules for abridging a charter. | Abridger, Writer, Editor |
| `02_institution_roster.md` | The exhaustive list of institutions, tiers, lineage map, key documents per institution, known-facts sheet. | Scout, Extractor, Writer |
| `03_research_protocol.md` | The step-by-step procedures for each role, with verbatim prompt templates. | Everyone executing tickets |
| `04_task_tickets.md` | The full ticket list with IDs, dependencies, and order of execution. | Coordinator, everyone |
| `05_repositories_and_search_strings.md` | Where the documents live (archives, digital libraries) and the exact search strings to use. | Scout, Fetcher |
| `06_rights_and_permissions.md` | Copyright status of each document; what may be reprinted, quoted, or must be paraphrased. | Abridger, Editor, Human |
| `07_risk_register_and_qa.md` | Known confusions, hallucination traps, QA gates, and the final acceptance checklist. | Verifier, Editor, Human |
| `templates/` | Fill-in-the-blank files: dossier, source log, quote record, abridged charter, chapter, verification report. | Everyone |

## The Ten Rules (read before every ticket)

1. **Never invent.** If you cannot find a fact, date, name, or quotation in a source you have actually opened, write `UNVERIFIED` in its place. An `UNVERIFIED` marker is a success. An invented fact is a failure that poisons the book.
2. **Every quotation must be copied, not remembered.** A quotation is allowed only if you have the source text open in front of you and copy it character for character. Record the source, page (or URL and paragraph), and the 20 words before and after the quote in the quote record.
3. **Distinguish primary from secondary.** A charter, constitution, statute, prospectus, minute book, letter, or founder's address is *primary*. A history book, encyclopedia, Wikipedia, or website "About" page is *secondary*. Secondary sources are used only to *find* primary sources and for background. Nothing goes into an abridged charter from a secondary source.
4. **One ticket at a time.** Open the ticket, read its inputs, do its numbered steps in order, write its output file, tick its "done when" list. Do not start the next ticket until the current one is marked `DONE` or `BLOCKED`.
5. **Output in the template, exactly.** Do not add sections, rename headings, or change field names. Empty fields are written as `UNVERIFIED` or `NOT FOUND (searched: <where>)`, never left blank and never deleted.
6. **When blocked, stop and flag.** Write `BLOCKED: <reason>` at the top of the output file and in the ticket log. Do not guess your way past a block.
7. **Paraphrase never goes inside quotation marks.** If you summarize, write "The statutes require that..." without quotation marks. If you quote, use quotation marks and a citation.
8. **Confidence labels are mandatory.** Every fact in a dossier carries one label: `[PRIMARY: source]`, `[SECONDARY: source]`, or `[UNVERIFIED]`.
9. **Do not correct the sources.** If two sources disagree (for example on a founding date), record both, with citations, in the `Discrepancies` field. A human resolves it.
10. **Do not skip the checklist.** A ticket is not done until every box in its "done when" list is ticked with evidence (a file path or a line number).

## Roles

Each ticket names one role. A single model can play all roles, but never two roles in the same ticket.

| Role | Does | Must not |
|---|---|---|
| **Scout** | Finds where the primary documents are (archive, digital library, printed edition). Fills the source log. | Extract content, quote, or summarize. |
| **Fetcher** | Downloads or transcribes the located documents into plain text with page markers. | Edit, correct, or "improve" the text. |
| **Extractor** | Reads the fetched text and fills the institution dossier, field by field, with citations. | Use memory or secondary sources for any dossier field marked PRIMARY-ONLY. |
| **Verifier** | Re-opens every cited source and checks every quotation and fact in a dossier or chapter. | Fix errors silently; the Verifier only reports. |
| **Abridger** | Produces the abbreviated charter text from the fetched primary text using the abridgement rules. | Add words that are not in the source, except bracketed editorial insertions. |
| **Writer** | Drafts the chapter from the dossier and the abridged charter using the chapter template. | Introduce facts not in the dossier. |
| **Editor** | Applies the style guide, checks the chapter against the template and word budgets, runs cross-chapter consistency checks. | Alter quotations or citations. |
| **Coordinator** | Assigns tickets, tracks status, escalates blocks. | Execute research tickets. |
| **Human reviewer** | Resolves `BLOCKED` tickets, discrepancies, rights questions, and signs off each Part. | — |

## Directory layout (create at project start, ticket C-001)

```
book/
  plan/                      <- copy of this plan
  roster.csv                 <- machine-readable roster (generated from 02_institution_roster.md)
  tickets.csv                <- ticket tracker (generated from 04_task_tickets.md)
  research/
    <slug>/                  <- one folder per institution, slug from roster
      sources.csv            <- source log (templates/source_log.csv)
      raw/                   <- downloaded files (PDF, images, HTML); never edited
      text/                  <- plain-text transcriptions with page markers; never edited after Fetcher
      dossier.md             <- filled templates/dossier.md
      quotes.jsonl           <- one JSON object per quotation (templates/quote_record.json)
      charter_abridged.md    <- filled templates/charter_abridged.md
      verification.md        <- filled templates/verification_report.md
      discrepancies.md       <- conflicts between sources, for human resolution
  chapters/
    <part>-<nn>-<slug>.md    <- filled templates/chapter.md
  synthesis/
    comparative_tables.md    <- cross-institution tables (curriculum, length, languages, funding, sending)
    lineage_chart.md         <- who trained whom, who founded what
    timeline.md
    glossary.md
  rights/
    permissions_log.csv
  assembly/
    manuscript.md            <- concatenated, ordered
```

## Ticket lifecycle

`TODO` → `IN_PROGRESS` → (`DONE` | `BLOCKED`) → (`VERIFIED` after the Verifier ticket passes) → `ACCEPTED` (human sign-off).

A ticket's output file must begin with a status line:

```
STATUS: DONE | BLOCKED: <reason>
TICKET: <id>
ROLE: <role>
INPUTS READ: <list of file paths>
```

## Slugs

Institution slugs are fixed in `02_institution_roster.md` (column `slug`). Never invent a slug. Examples: `geneva-academy`, `andover-seminary`, `pastors-college`, `china-inland-mission`, `london-theological-seminary`.

## Order of work (summary)

1. Setup tickets (C-series).
2. For each Tier A institution, in roster order: Scout → Fetcher → Extractor → Verifier → Abridger → Writer → Editor.
3. Tier B institutions, same pipeline, after all Tier A dossiers are `VERIFIED`.
4. Synthesis tickets (X-series) after all Tier A chapters are `VERIFIED`.
5. Rights tickets (P-series) run in parallel from the moment a dossier is `VERIFIED`.
6. Assembly tickets (F-series) last.

Full detail and dependencies are in `04_task_tickets.md`.
