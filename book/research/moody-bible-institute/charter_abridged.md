STATUS: BLOCKED (partial substitute provided — see note below)
TICKET: B-moody-bible-institute
ROLE: Abridger
INPUTS READ: text/lifedwight00mood.txt; text/moodybibleinstit00camp.txt; dossier.md field F5; book/plan/01_book_design.md section 6

# Abridged charter — Moody Bible Institute (`moody-bible-institute`)

**IMPORTANT — no founding document was located.** The roster's key primary documents for this
institution are "Chicago Evangelization Society charter (1887) [VERIFY]," the first *Prospectus*
/ catalog (1889-90), and Moody's "gap men" address (1886). None of these was located as digitized
full text this session (searched: Internet Archive full-text and advancedsearch, HathiTrust
catalog records, Google Books, general web search — see `sources.csv` rows
`ces-charter-1887-not-located` and `mbi-prospectus-1889-90-not-located`). No incorporation
document, statute, or prospectus text exists in `text/` to abridge. This is logged as **BLOCKED**:
a follow-up Scout/Fetcher pass is needed against the Moody Bible Institute Archives
(library.moody.edu/collections/archives) or the Illinois Secretary of State's historical
corporate-charter records before a true abridged charter, in the sense the book design calls
for, can be produced.

**What follows instead**, on the precedent set by the `geneva-academy` chapter (BLOCKED for the
same reason: a partial primary-adjacent text stood in for an unrecovered original), is a
**partial charter-in-substance**: passages of Moody's own recorded words, quoted verbatim with
quotation marks in his son's 1900 biography (document_id `lifedwight00mood`), covering the
founding need, the chartering act, the course of study, and the classes of students admitted.
These are not the 1887 legal charter; they are the closest verified substitute located this
session, all in the same document so that every kept passage below can be checked verbatim
against a single `text/` file (`python check_quotes.py research/moody-bible-institute --charter`).

A second, independent institutional source — Norman H. Camp's 1915 *Brief Historical Sketch*,
document_id `moodybibleinstit00camp` — supplies the Institute's own standing doctrinal-basis and
"objective" statements, quoted in full in `quotes.jsonl` (moody-bible-institute-q014,
moody-bible-institute-q015) and used directly in the chapter's Founding and Emphasis sections;
it is not reproduced below only to keep this file's verbatim check against one source document.

## Structure of this partial abridgement

| # | Content | Source (quote_id) | KEEP/CUT |
|---|---|---|---|
| 1 | The founding need, in Moody's own recorded expression ("gap men") | moody-bible-institute-q001 | KEEP (a, b) |
| 2 | The chartering act itself, as narrated by Moody's son | moody-bible-institute-q002 | KEEP (a) |
| 3 | The two-year, twelve-month "circle" course | moody-bible-institute-q004 | KEEP (d) |
| 4 | The daily division between class-room study and practical work | moody-bible-institute-q005 | KEEP (d, e) |
| — | The three classes of students the Institute aimed to serve (admission) | moody-bible-institute-q008 | KEEP in chapter body only (c) — the fetched OCR reads "the [nstitute" (a stray unmatched bracket from a damaged capital "I"); reproducing it here breaks the mechanical `[...]`-cut-marker parsing that `check_quotes.py --charter` relies on, so this passage is quoted in the chapter's Admission section (checked independently via `quotes.jsonl`/`check_quotes.py`, no `--charter`) rather than in this file's single continuous verbatim block |
| — | The doctrinal-basis and objective statements (1915) | moody-bible-institute-q014, moody-bible-institute-q015 | KEEP in chapter body (b, d) — omitted here only for single-document verbatim checking, see note above |
| — | The actual 1887 articles of incorporation / charter | NOT LOCATED | CUT — BLOCKED, not fetched |
| — | The actual 1889-90 prospectus/catalog | NOT LOCATED | CUT — BLOCKED, not fetched |

Word count of the substitute text below: approximately 210 words. Well under the book design's
1,200-2,500 word target because this is a substitute for an unrecovered document, not an
abridgement of one in hand; no further passages were withheld to reach a target length.

## Headnote

No single founding document survives in this partial abridgement. What follows is assembled, in
Moody's own quoted words, from William R. Moody, *The Life of Dwight L. Moody* (New York:
Fleming H. Revell Company, 1900), chapter XXX, "The Bible Institute for Home and Foreign
Missions" — English, as printed, digitized by the Internet Archive from a Princeton Theological
Seminary copy. Each passage below is copied verbatim from `text/lifedwight00mood.txt`, in the
document's own original order, with `[...]` marking every cut between passages.

**Note on the text below:** reproduced exactly as OCR'd from the digitized 1900 edition,
including the period scanner's misreadings (e.g. "gOl t«>" for "got to," "arc" for "are,"
"[nstitute" for "Institute," "hook" for "book," "wdio" for "who," "the}'" for "they," "he" for
"be") per the book's rule against silently correcting a fetched transcription. No corrections
are inserted in the text itself; the intended reading in each case is plain from context.

## Text

" 1 believe we have gOl t«> have ' gap men - men who arc trained to stand between the laity and
the ministers," was a common expres sion of Mr. Moody's.

[...]

Responses to this appeal came heartily, the money was pledged, the preliminary steps were taken,
and the new- enterprise was chartered under the name of " The Chicago Evan- gelization Society."

[...]

The system embraces a thorough doctrinal, analyti cal, and hook study of the English Bible under
the tuition of resident instructors. Added to this, lectures are given by the best Bible teachers
from both sides of the water on topics to which the}' have individually given the closest
attention. While spiritual exposition is emphasized, all is based upon the most careful and
scholarly study of the Word. Two years of twelve months each are required for the course, but,
as it proceeds in a circle, students can enter at any time and by remaining two years complete
the full course.

[...]

The morning hours are spent in the class-room, and the afternoons and evenings are divided
between study and practical work among the unconverted. Rescue mission work, house-to-house
visitation, children's meetings, women's meetings, jail work, inquiry-meeting work, church
visitation — every form of effort which can be developed in the heart of a great and wicked city
is here supplied.
