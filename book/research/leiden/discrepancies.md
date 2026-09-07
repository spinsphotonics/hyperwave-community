# Discrepancies — leiden

1. **The roster's description conflates two different institutions.** The roster row for
   `leiden` reads: "Founding charter 1575 (Dutch/Latin); Staten College statutes 1592
   [VERIFY]" and describes the whole entry as "the Dutch Reformed seminary that trained
   ministers for the Indies." The material read this session indicates these are two, or
   arguably three, distinct things:
   - Leiden University itself (charter 1575), a general university with a theological
     faculty among four faculties [leiden-q001, leiden-q003, leiden-q011].
   - The Statencollege (Staten Collegie / Collegium Theologicum), founded 1592, a
     scholarship boarding house for theology students, set up by the States of Holland
     and West-Friesland to address a shortage of ministers for the home Dutch Reformed
     Church — with no mention in the sources read this session of the East Indies
     [leiden-q024, leiden-q025].
   - The Seminarium Indicum ("Indian Seminary"), founded 1622 and closed 1633, a separate,
     later institution, run by the Leiden professor Antonius Walaeus out of his own house,
     specifically to train ministers for service in the Dutch East Indies under the VOC
     [leiden-q014, leiden-q015, leiden-q018, leiden-q027].
   The phrase "trained ministers for the Indies" in the roster describes the Seminarium
   Indicum, not the Statencollege. Status: UNVERIFIED whether the roster intended the two
   as one continuous institution (the Statencollege continuing under another name) or
   conflated them. The chapter treats the two as related but distinct, per the sources
   read, and states this discrepancy openly rather than silently correcting the roster.

2. **Founding-date discrepancy for the University itself.** Leiden University's own page
   states the University's inauguration was 8 February 1575, but the charter document
   itself is dated 6 January 1575 despite being drawn up only after the inauguration
   [leiden-q001]. Both dates are given in the same primary-adjacent source; the roster's
   single figure "1575" is consistent with either but does not specify which.

3. **Regent's name spelling: Waleus vs. Walaeus.** The Dutch Wikipedia article on the
   Seminarium Indicum spells the rector's name "Waleus" [leiden-q027], while the
   Christian Study Library biographical article (translating/summarizing Dutch sources)
   spells it "Walaeus" throughout, as does English-language scholarship generally
   (Antonius Walaeus, 1573-1639). The chapter uses "Walaeus," the form fixed by English
   scholarship, per the roster/style-guide convention of using the form most common in
   English scholarship, and flags the variant spelling here rather than treating it as an
   error.

4. **Numbers sent from the Seminarium Indicum: "a dozen" vs. "twelve."** The Christian
   Study Library article states Walaeus "delivered to the churches in the Dutch East
   Indies a dozen ministers of the Word" [leiden-q020]. The Dutch Wikipedia article gives
   an exact breakdown: twenty-five students studied at the seminary, six did not complete
   their studies, fourteen became ministers, of whom twelve served the VOC
   [leiden-q028]. These two independent secondary accounts corroborate each other closely
   ("a dozen" / "twelve... in dienst van de VOC") but are not word-for-word identical and
   neither is a primary source (the seminary's own enrollment register was not located
   this session). Both figures are reported in the chapter with their sources; neither is
   presented as more authoritative than the other.

5. **Primary charter and statutes texts not obtained.** The actual 1575 charter and
   1592/1625-era statutes texts (Molhuysen's published transcriptions) are held at
   resources.huygens.knaw.nl, a host blocked by this session's network egress policy (see
   sources.csv rows `leiden-charter-molhuysen-transcription` and
   `leiden-statutes-molhuysen-transcription`). This is the single largest gap in this
   chapter's research: no full charter or statute text was read this session, only
   short quoted fragments reproduced in secondary sources (the university's own website,
   the Deddens biographical article, and the two Wikipedia articles). A follow-up Fetcher
   ticket should re-attempt this host, or try Google Books
   (books.google.com/books?id=S_FKAAAAYAAJ) or HathiTrust
   (catalog.hathitrust.org/Record/011724854) for a digitized copy of Molhuysen's
   *Bronnen tot de geschiedenis der Leidsche universiteit*.
