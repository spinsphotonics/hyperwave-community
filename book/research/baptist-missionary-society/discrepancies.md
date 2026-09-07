# Discrepancies — baptist-missionary-society

1. **"Founding resolution and minutes (1792)" as a distinct roster item vs. the copy used.**
   The roster's "Key primary documents" cell lists "founding resolution and minutes (1792)"
   separately from the *Periodical Accounts*. This session did not locate a standalone 1792
   minute-book leaf or broadside; the founding resolutions of 2 October 1792 were instead
   read and quoted from their reprint in *Periodical Accounts relative to the Baptist
   Missionary Society*, Vol. I (1800), document_id `pa1800`, which explicitly presents itself
   as reproducing "the following resolutions" agreed at Kettering. No contradiction is known
   between the roster's two document descriptions; they most likely name the same textual
   content (the founding minute) as preserved in two possible physical forms (an original
   manuscript minute vs. its 1800 printing). See `sources.csv` row
   `periodicalaccounts-founding-resolution-1792-standalone`.

2. **Committee member name spelling: "John Sutrchff" vs. "John Sutcliff."**
   The OCR of `pa1800` (Resolution 5, 2 Oct. 1792) renders the name as "John Sutrchff"
   [bms-q010]. Every other source consulted for this book (including the roster itself,
   `02_institution_roster.md`, row `baptist-missionary-society`) gives "Sutcliff." This is
   judged an OCR misreading, not a second spelling attested in a source; left uncorrected in
   the quote record per Rule 9 and the Fetcher instruction not to correct source text, but
   the dossier's own prose uses the standard spelling "Sutcliff" outside quotation marks.

3. **Resolution 1's text has a visible OCR gap.**
   The OCR of Resolution 1 (2 Oct. 1792) reads "we ... do ſolemnly _ to act in ſociety
   together for that purpoſe" [bms-q008] — an underscore where a verb (almost certainly
   "agree") should stand. Left as OCR gives it; not corrected or guessed at.

4. **OCR error rate on `pa1800` (see `sources.csv`).** The founding-minute passages,
   hand-checked over a three-page sample, run at approximately 4-7 errors per 100 words
   (chiefly long-s "ſ" misreadings), at or slightly above the 5/100 threshold specified in
   Procedure F step 3. No re-transcription tool (e.g. tesseract with historical-font
   training) was available this session to produce a cleaner OCR pass. Flagged here rather
   than silently accepted. Every string actually quoted in `quotes.jsonl` was independently
   confirmed, character-by-character, to appear verbatim in the fetched text file (and
   `check_quotes.py` confirms this programmatically); the elevated error rate affects the
   surrounding unquoted narrative more than the specific short quoted spans, which were
   chosen partly because they were legible.

5. **Carey's Enquiry: two different digitizations, one superseded.** See `sources.csv`.
   A 1961 photographic-facsimile OCR scan (archive.org id `enquiryobligat00careuoft`) was
   fetched first but its OCR misreads the original's long-s typography so heavily that it
   was judged unusable for quotation (a sample far exceeded the 5-errors-per-100-words
   threshold — e.g. "commiffion," "fubjeft," "Chrift" for "commission," "subject," "Christ").
   It is retained at `raw/enquiryobligat00careuoft_djvu.txt` for the record but is not the
   `carey1792` document_id used in `quotes.jsonl`; the Project Gutenberg proofread plain-text
   transcription of the identical 1792 work was used instead. No difference in wording
   between the two is asserted or implied; the switch is solely for legibility of the
   digitization.
