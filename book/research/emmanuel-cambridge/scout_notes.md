STATUS: DONE
TICKET: S-emmanuel-cambridge
ROLE: Scout
INPUTS READ: book/plan/02_institution_roster.md (row `emmanuel-cambridge`); book/plan/05_repositories_and_search_strings.md (section `emmanuel-cambridge`)

# Scout notes — Emmanuel College, Cambridge (`emmanuel-cambridge`)

## Target documents (from the roster's "Key primary documents" cell)

1. Mildmay's Statutes of Emmanuel College (1585, Latin).
2. Mildmay's letters to Elizabeth (the "I have set an acorn..." anecdote) [roster flags: VERIFY wording and source].

## Search strings run (from `05_repositories_and_search_strings.md`)

- `"Statutes of Sir Walter Mildmay" Emmanuel` — found the Stubbings 1983 Cambridge University Press
  translation (copyrighted; abstract page only; see sources.csv row `stubbings1983`). Did not find a
  freely downloadable full text of Stubbings's translation.
- `Emmanuel College statutes 1585 Latin` — led (via a follow-up search on the VCH footnote citation
  "Comm. Doc. iii. 483-526") to the 1852 Royal Commission compilation *Documents Relating to the
  University and Colleges of Cambridge*, vol. III, which reprints the actual 1585 Latin statutes
  (from a British Museum Sloane MS.) and the 1584 Latin charter. This is the best primary text located
  this session. FOUND on archive.org: `documentsrelati03commgoog`. See sources.csv row `commdoc1852v3`.
- `Mildmay "acorn" Elizabeth Emmanuel` — WebSearch snippets attributed the anecdote to Thomas Fuller.
  Traced to Fuller's *History of the University of Cambridge* (1655; 1840 Prickett & Wright reprint),
  archive.org identifier `historyuniversi01nichgoog`. FOUND and fetched; see sources.csv row
  `fuller1655`. This is the earliest source for the anecdote located this session — see
  discrepancies.md for the "no earlier source found" negative-search result.
- `"Emmanuel College" Cambridge Puritan seminary founding` — general orientation only; led to
  Wikipedia (not fetched, never citable per Rule 3) and to the Emmanuel College Cambridge official
  site's "Our History" page (not fetched this session; would need curl if pursued further).

## Additional finds (bibliography-mining / follow-on search, not in the original search-string list)

- E. S. Shuckburgh, *Emmanuel College* (1904), in the "University of Cambridge College Histories"
  series — found via a direct search for "History of Emmanuel College" Cambridge archive.org
  (identifier `emmanuelcollege00shucrich`). This is a full, freely downloadable scholarly history by
  a former Fellow and Librarian of the College, and quotes extensively (with citation) from the
  founder's statutes, the College order book, and the Bursar's accounts. FETCHED — see
  sources.csv row `shuckburgh1904`. This became the single richest source of the session.
- Victoria County History, *A History of the County of Cambridge and the Isle of Ely*, vol. 3 (1959),
  "The colleges and halls: Emmanuel" — found via British History Online. FETCHED for corroboration;
  its footnote apparatus supplied the exact citation (`Comm. Doc. iii. 483-526`) that led to the 1852
  Commission volume above. See sources.csv row `vch1959`.
- A. Sarah Bendall, Christopher Brooke, and Patrick Collinson, *A History of Emmanuel College
  Cambridge* (1999) — the roster's own named secondary holding. Located on archive.org
  (`historyofemmanue0000bend`) but listed as a controlled-digital-lending (borrow-only) item; full
  text NOT accessible without a login/borrow step, which was not performed this session. NOT used.
  See sources.csv.
- Sargent Bush, *The Library of Emmanuel College, Cambridge, 1584-1637* — same access situation as
  above (`libraryofemmanue0000bush`); NOT used.

## Nickname / anecdote origin (institution-specific note, analogous to the geneva-academy "school of
death" instruction in Procedure S step 5, applied here to the acorn anecdote per the ticket's own
special instruction)

The earliest printed source located for the "acorn" exchange between Mildmay and Elizabeth I is
Thomas Fuller, *The History of the University of Cambridge* (1655), Book VI, section 17-19
("Emmanuel College founded by Sir Walter Mildmay, who causelessly fell into the Queen's Displeasure.
His Answer to Queen Elizabeth"). No earlier (16th-century or early-17th-century) printed source was
located or searched for beyond this session's scope. E. S. Shuckburgh (1904) reprints the same
anecdote and calls it "the often-repeated story," citing no source earlier than Fuller. STATUS:
earliest source found = Fuller 1655 (66 years after Mildmay's death, not a contemporary eyewitness
record). Treated as SECONDARY throughout the dossier, not as Mildmay's or Elizabeth's own words
verified from a contemporary document. See discrepancies.md.

## Done-when checklist

- [x] Every target document has at least one row (hit) in sources.csv.
- [x] Every sources.csv row has all columns filled (no blanks; NOT FETCHED / NONE used explicitly
      where applicable).
- [x] This file lists targets, new targets found, and the institution-specific nickname/anecdote
      finding.
- [x] No content was summarized in this file beyond identifying what each document is and where it
      was found (per Procedure S: "Scout does not read for content" — note that in this single
      session Scout, Fetcher, and Extractor roles were executed in sequence by the same operator, as
      the ticket instructs; this file records only the finding, not the extraction).
