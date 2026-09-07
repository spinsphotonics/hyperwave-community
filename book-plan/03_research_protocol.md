# 03 — Research Protocol

This file gives the exact steps for every role. Each ticket in `04_task_tickets.md` names one of the procedures below (S, F, E, V, B, D, ED, X, P) and one institution slug. Execute the numbered steps in order. Do not skip. Do not combine.

Before any procedure: read `00_README.md` (the Ten Rules), the roster row for the slug in `02_institution_roster.md`, and the template you will fill in `templates/`.

---

## Procedure S — Scout (find the primary documents)

**Input:** roster row for `<slug>`; `05_repositories_and_search_strings.md`.
**Output:** `research/<slug>/sources.csv` (template `templates/source_log.csv`) and `research/<slug>/scout_notes.md`.

Steps:

1. Copy the "Key primary documents" cell of the roster row into `scout_notes.md` as a numbered list. Each item is a **target document**.
2. For each target document, in order:
   1. Open `05_repositories_and_search_strings.md`, section for this institution. Run every listed search string in every listed repository. Record every hit in `sources.csv` (one row per hit) with: `document_id, target_document, title_as_catalogued, author, date, language, repository, url_or_shelfmark, format (pdf/images/html/print-only), full_text_available (yes/no), pages_or_extent, is_primary (yes/no), edition_notes, found_by_search_string`.
   2. If a repository returns nothing, record a row with `url_or_shelfmark = NONE` and `found_by_search_string = <the string>` so the search is documented.
   3. If a target document exists only in print or in a physical archive, record the shelfmark and the archive's reproduction request procedure (copy the URL of the archive's "request copies" page into `edition_notes`).
3. Secondary sources: for each item in the roster's "Secondary" cell, find a library record or online copy and add a row with `is_primary = no`.
4. Bibliography mining: open each secondary source found (if digitized) and read only its bibliography/notes for the chapter on this institution. Add any primary document it cites that is not already a target as a new row with `target_document = NEW: <title>`. Do not read the secondary source's narrative.
5. Special steps by institution (do all that apply):
   - `geneva-academy`: search for the earliest printed use of "school of death" / "école de la mort" / "schola mortis" in connection with the Academy. Search strings in `05`. Record every hit with date. Record the earliest in `scout_notes.md` under "Nickname origin." If none is found before 1900, write `UNVERIFIED: no pre-1900 use found`.
   - `china-inland-mission`: find at least two editions of *Principles and Practice* with dates, and the 1886 *Arrangements*. Record the differences in titles.
   - `pastors-college`: find the *Sword and the Trowel* issues that print the College's annual report (search 1865–1892 volumes, "Pastors' College" + "report").
   - `andover-seminary` and `abcfm`: find the 1812 *Instructions* to Judson's party (Salem, 1812) and the *Constitution and Associate Statutes* (1808).
   - `london-theological-seminary`: find the LTS 1977 prospectus/aims and the opening address; record the rights holder.
   - `wheaton-college`, `cmml-brethren`, `prairie-bible-institute`, `wycliffe-sil`: record the archive finding aid for the Elliot papers and the institutional catalogs for 1947–1952.
6. Fill the `STATUS` header. Done when:
   - [ ] Every target document has at least one row (hit or NONE).
   - [ ] Every row has all columns filled (no blanks).
   - [ ] `scout_notes.md` lists targets, new targets, and any institution-specific findings.
   - [ ] No content from any document has been summarized (Scout does not read for content).

---

## Procedure F — Fetcher (acquire and transcribe)

**Input:** `research/<slug>/sources.csv` rows with `is_primary = yes` and `full_text_available = yes`.
**Output:** files in `research/<slug>/raw/` and `research/<slug>/text/`; updated `sources.csv` column `fetched_file`.

Steps:

1. For each primary row, download the file to `raw/<document_id>.<ext>`. If it is an HTML page, save the page as HTML and also as PDF (print to PDF). If it is a set of page images, save all images in a folder `raw/<document_id>/`.
2. Produce a plain-text transcription `text/<document_id>.txt`:
   1. If the source provides text (OCR or born-digital), copy it.
   2. If only images: run OCR (tesseract or the repository's OCR). For pre-1800 print, set the OCR language to include Latin/French/German as appropriate and record the OCR engine and language setting at the top of the file.
   3. Insert a page marker on its own line before each page: `[[p. N]]` using the printed page number, or `[[img. N]]` if unpaginated.
   4. Do not correct spelling. Do not modernize. Do not remove line breaks inside paragraphs from OCR; just leave them.
   5. At the top of the file write a header:
      ```
      DOCUMENT_ID: ...
      TITLE (as printed): ...
      DATE: ...
      LANGUAGE: ...
      SOURCE URL/SHELFMARK: ...
      EDITION: ...
      TRANSCRIPTION METHOD: copied / OCR (engine, lang)
      TOTAL PAGES: ...
      WORD COUNT: ...
      ```
3. Quality sample: pick three pages (first, middle, last), compare the OCR text to the image, and count the errors per 100 words. Record the rate in `sources.csv` column `ocr_error_rate`. If above 5 errors per 100 words, mark the row `NEEDS_RETRANSCRIPTION` and STOP for that document; a human or better OCR will handle it.
4. For documents that are print-only, create `text/<document_id>.REQUEST.md` containing the archive, shelfmark, the request procedure, and the specific pages or sections needed (from the roster's key documents). Mark the ticket `BLOCKED: print-only, request filed`.
5. Done when:
   - [ ] Every primary row has `fetched_file` filled or a `.REQUEST.md`.
   - [ ] Every `text/*.txt` has the header and page markers.
   - [ ] `ocr_error_rate` filled for every transcription.

---

## Procedure E — Extractor (fill the dossier)

**Input:** `research/<slug>/text/*.txt`; `templates/dossier.md`; the roster row.
**Output:** `research/<slug>/dossier.md`; `research/<slug>/quotes.jsonl`; `research/<slug>/discrepancies.md`.

Rules specific to this procedure:
- Fields marked `PRIMARY-ONLY` in the template may be filled only from `text/` files. Fields marked `PRIMARY-OR-SECONDARY` may use secondary sources that are in `sources.csv`, cited as `[SECONDARY: document_id, page]`.
- A field is filled by (a) a one-to-three sentence summary in your own words, without quotation marks, followed by (b) one or more supporting quotations, each as a quote record.
- A quote record is one line of JSON in `quotes.jsonl` (schema in `templates/quote_record.json`) with the exact text, `document_id`, page marker, the 20 words before, the 20 words after, and the dossier field it supports. The dossier cites the quote by its `quote_id`.
- Copy quotations by selecting text from the `text/` file. Never type a quotation from memory.

Steps:

1. Open `templates/dossier.md`; save as `research/<slug>/dossier.md`; fill the header.
2. Read every `text/*.txt` file end to end once. While reading, whenever a passage answers one of the dossier fields, create a quote record immediately (do not wait until the end). Tag it with the field code (F1–F8, A1–A6, C1–C9, L1–L8, T1–T5, M1–M5, S1–S6, R1–R4, X1–X3).
3. After reading, go through the dossier fields in order. For each field:
   1. List the quote_ids tagged for that field.
   2. Write the summary sentence(s) from those quotes only.
   3. Paste the quote_ids.
   4. If no quote exists and the field is PRIMARY-ONLY: write `NOT FOUND (searched: <list of document_ids>)`.
   5. If no quote exists and the field is PRIMARY-OR-SECONDARY: search the secondary sources in `sources.csv`; if found, fill with `[SECONDARY: ...]`; else `UNVERIFIED`.
4. Field F9 "Antecedents": record only what the sources say about which earlier schools or documents the founders imitated or cited. Give quote_ids.
5. Discrepancies: whenever two sources give different dates, names, numbers, or wording, add a row to `discrepancies.md`: `field, source A (id, page), value A, source B (id, page), value B`. Do not choose.
6. Numbers: every number (students, missionaries sent, deaths, years of study, hours per day) needs a quote record. If the source gives a number for a specific year, record the year.
7. Done when:
   - [ ] Every field in the template is present and non-blank.
   - [ ] Every summary is followed by at least one quote_id or a NOT FOUND/UNVERIFIED marker.
   - [ ] `quotes.jsonl` parses as JSON, one object per line.
   - [ ] Every quote's text appears verbatim in the named `text/` file (run the check script in `templates/check_quotes.py`).
   - [ ] `discrepancies.md` exists (may say "none found").

---

## Procedure V — Verifier (check everything)

**Input:** the dossier, quotes, and text files for `<slug>` (or a chapter file at stage V2).
**Output:** `research/<slug>/verification.md` (template `templates/verification_report.md`).

Steps:

1. Run `templates/check_quotes.py research/<slug>` and paste the output into the report. Any quote not found verbatim is a FAIL.
2. For every quote record, open the `text/` file at the page marker and confirm: (a) the text matches, (b) the 20-words-before and after match, (c) the page marker is correct. Record PASS/FAIL per quote_id.
3. For every dossier field, confirm that the summary sentence does not claim more than the quotes support. Write one line per field: `F3: PASS` or `F3: FAIL — summary says X, quotes only support Y`.
4. For every `[SECONDARY: ...]` citation, open the secondary source and confirm the page says what is claimed.
5. Check every date and number against the known-facts sheet in `02_institution_roster.md`. Any disagreement goes to `discrepancies.md` and the report.
6. Check that no field marked PRIMARY-ONLY cites a secondary source.
7. Check that nothing in the dossier is stated without a label.
8. Verdict: `VERIFIED` only if zero FAILs. Otherwise `RETURNED` with the list of failing items. The Verifier never edits the dossier.
9. Stage V2 (chapter verification): repeat steps 1–7 against the chapter file, additionally checking that every footnote resolves to a `sources.csv` row and that no sentence in the chapter contains a fact absent from the dossier.

---

## Procedure B — Abridger (produce the abridged charter)

**Input:** `research/<slug>/text/<document_id>.txt` for the charter chosen in the dossier field F5 ("Principal founding document"); `01_book_design.md` section 6; `templates/charter_abridged.md`.
**Output:** `research/<slug>/charter_abridged.md`.

Steps:

1. Count the words of the full document. Write it in the header.
2. Produce the structure outline: list every article/section/heading in the document in order with its word count. (For undivided documents, split by paragraph and number the paragraphs.)
3. Against each outline item, mark KEEP or CUT using the priority list in `01_book_design.md` section 6.2–6.3. Record the reason code (a–h for keep; "admin", "names", "repeat" for cut).
4. Assemble the abridgement: copy KEEP passages verbatim, in original order, with `[...]` at each cut. Insert bracketed editorial headings only where 6.5 allows.
5. If a translation is needed: use the translation named in `sources.csv`. If none, produce a literal translation paragraph by paragraph, keep the original alongside in a two-column table in a separate file `charter_abridged_bilingual.md`, mark `[New translation, draft]`, and set the ticket status `BLOCKED: translation review required`.
6. Word-count the abridgement. If outside 1,200–2,500 words, adjust by cutting or restoring whole outline items (never by rewriting sentences), and record the change.
7. Write the headnote (one sentence: title, date, language, edition, translator).
8. Done when:
   - [ ] Header has both word counts and the KEEP/CUT outline.
   - [ ] Every kept passage matches the `text/` file verbatim (run `templates/check_quotes.py --charter`).
   - [ ] Every cut is marked.
   - [ ] Headnote present.

---

## Procedure D — Writer (draft the chapter)

**Input:** `research/<slug>/dossier.md` (status VERIFIED), `quotes.jsonl`, `charter_abridged.md`, `01_book_design.md` sections 5 and 7, `templates/chapter.md`.
**Output:** `chapters/<part>-<nn>-<slug>.md`.

Steps:

1. Copy `templates/chapter.md`. Fill the header (title, institution, dates, part, chapter number from `04_task_tickets.md`).
2. Epigraph: choose the quote record tagged `EPIGRAPH` in the dossier field X1. Paste verbatim with footnote.
3. For each of the sections Founding, Admission, Curriculum, Common Life and the Faculty, Emphasis, Sending, Fruit: open the dossier fields listed for that section in the chapter template; write the section using only those fields; footnote every fact with the quote_id's source and page; paste up to three short quotations per section from the quote records.
4. The Charter section: paste `charter_abridged.md` body under the headnote.
5. Sources: generate from `sources.csv` (primary first, then secondary), Chicago style. Use the citation formatter in `templates/cite.py` if available; otherwise follow the examples in the template.
6. Word count each section; write the counts in the header table.
7. Self-check against the style guide (`01_book_design.md` section 7): search the draft for the forbidden words and delete them; check average sentence length.
8. Done when:
   - [ ] All ten sections present, in order, with the exact headings.
   - [ ] Every footnote points to a quote_id or a `sources.csv` document_id with page.
   - [ ] No sentence states a fact that is `UNVERIFIED` or `NOT FOUND` in the dossier.
   - [ ] Word counts within budget or flagged.

---

## Procedure ED — Editor

**Input:** a chapter with status VERIFIED (after V2).
**Output:** the same chapter, edited, plus `chapters/<file>.editlog.md`.

Steps:

1. Apply the style guide mechanically: spelling, dates, names as fixed in the roster, forbidden words, sentence length.
2. Check every name and date against the roster and known-facts sheet.
3. Check headings and section order against the template.
4. Check that the chapter answers all five questions (`01_book_design.md` section 8); if one is unanswered because the dossier has NOT FOUND, insert one sentence saying the sources are silent on it.
5. Cross-chapter consistency: open `synthesis/comparative_tables.md` and confirm the chapter's figures (course length, languages, fees, numbers sent) match the table; if not, log in the editlog and notify the Coordinator.
6. Do not touch quotations or footnotes. If one looks wrong, log it for the Verifier.
7. Write the editlog: every change as `line N: before -> after`.

---

## Procedure X — Synthesis (cross-institution)

Runs after all Tier A dossiers are VERIFIED.

X1 Comparative tables (`synthesis/comparative_tables.md`): for every institution, one row per table, values copied from dossier fields with the quote_id in a citation column:
- Table 1 Admission: minimum age; prior education; conversion/testimony required; examination; women admitted (yes/no/role).
- Table 2 Curriculum: length of course; languages taught (Hebrew/Greek/Latin/vernacular/field language); set theological texts; preaching instruction (yes/no; form).
- Table 3 Common life: residential (yes/no); daily worship (times); teacher:student ratio; forms of teacher–student contact.
- Table 4 Funding: fees charged; source of support; salary or "faith" principle.
- Table 5 Sending: sending body; typical destinations; number sent in the first 25 years; recorded deaths in service.
- Table 6 Doctrinal basis: name of confession or creed subscribed; who subscribed (students, faculty, both).

X2 Lineage chart (`synthesis/lineage_chart.md`): for every institution, the dossier's F9 antecedents and the successors named in other dossiers; produce a list of edges `A -> B (evidence quote_id)`; then a Mermaid diagram.

X3 Timeline (`synthesis/timeline.md`): every dated event in the dossiers' F-fields and S-fields, sorted, with citation.

X4 Glossary (`synthesis/glossary.md`): every technical term appearing in three or more chapters (disputation, prophesying, faith mission, commendation, Bible institute, licensure, congrégation, deeper life, and so on), with a one-sentence definition drawn from one of the dossiers and cited.

X5 Introduction brief (`synthesis/introduction_brief.md`): a bullet list of the ten most striking contrasts and continuities visible in the tables, each with the table row references. A human writes the Introduction from this.

---

## Procedure P — Rights

For each document used in a charter or quoted at length: fill one row in `rights/permissions_log.csv` per `06_rights_and_permissions.md`. If permission is needed, draft the request from the template and STOP for human sending.

---

## Prompt templates (copy verbatim into the model; replace angle-bracket fields)

### Scout prompt
```
You are the Scout for institution <slug>. Read book-plan/00_README.md rules 1–10 and book-plan/03_research_protocol.md Procedure S. Your only job is to find where the primary documents listed in the roster row below are held or digitized and to log every search in sources.csv. Do NOT read the documents for content. Do NOT summarize them. If a search returns nothing, log the search string with NONE.
Roster row: <paste row>
Search strings: <paste section from 05_repositories_and_search_strings.md>
Output: the completed sources.csv and scout_notes.md, with the STATUS header.
```

### Fetcher prompt
```
You are the Fetcher for institution <slug>. Read Procedure F. For each primary row in sources.csv with full_text_available = yes, download the file to raw/ and produce text/<document_id>.txt with the required header and [[p. N]] page markers. Do not correct or modernize any text. Report ocr_error_rate from a three-page sample. If a document is print-only, write the .REQUEST.md file and mark BLOCKED.
```

### Extractor prompt
```
You are the Extractor for institution <slug>. Read Procedure E and templates/dossier.md. Fill every field of the dossier from the text/ files only (for PRIMARY-ONLY fields) or from sources listed in sources.csv (for PRIMARY-OR-SECONDARY fields). For every fact create a quote record in quotes.jsonl by copying the exact text from the text/ file with its page marker and 20 words of context on each side. If you cannot find something, write NOT FOUND (searched: ...) or UNVERIFIED. Never write a fact from memory. Never put paraphrase inside quotation marks. Record every disagreement between sources in discrepancies.md without resolving it.
```

### Verifier prompt
```
You are the Verifier for institution <slug>. Read Procedure V. Run the quote check script. Open every cited page and confirm each quotation and each summary. Report PASS or FAIL per item in verification.md. Do not edit the dossier or chapter. Verdict is VERIFIED only with zero FAILs.
```

### Abridger prompt
```
You are the Abridger for institution <slug>. Read Procedure B and 01_book_design.md section 6. Work only from text/<document_id>.txt named in dossier field F5. Produce the outline with KEEP/CUT marks, then the abridgement with every cut marked [...]. Copy kept passages verbatim. Do not rewrite sentences. Target 1,200–2,500 words.
```

### Writer prompt
```
You are the Writer for institution <slug>. Read Procedure D, 01_book_design.md sections 5–8, and templates/chapter.md. Write the chapter using only facts in dossier.md (status VERIFIED) and quotations from quotes.jsonl. Footnote every fact. Do not state anything the dossier marks UNVERIFIED or NOT FOUND. Follow the style guide: no praise adjectives, no forbidden words, sentences under 30 words on average.
```

### Verifier (chapter) prompt
```
You are the Verifier for chapter <file>. Read Procedure V step 9. For every sentence, find the dossier field or quote_id that supports it. Any sentence without support is a FAIL. Check every footnote resolves. Report in verification.md.
```
