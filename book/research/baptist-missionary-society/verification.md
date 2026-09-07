STATUS: VERIFIED
TICKET: V1-baptist-missionary-society
ROLE: Verifier
INPUTS READ: dossier.md, quotes.jsonl, sources.csv, discrepancies.md, text/carey1792.txt, text/pa1800.txt, text/smith1885.txt, charter_abridged.md

# Verification report — baptist-missionary-society (V1, dossier + charter)

## 1. Script output
```
$ python3 plan/templates/check_quotes.py research/baptist-missionary-society
WARN bms-q005: before+text+after not contiguous in pa1800
17/17 quote records passed; 0 failures

$ python3 plan/templates/check_quotes.py research/baptist-missionary-society --charter
8/8 passages verbatim; 0 failures
```
The single WARN (bms-q005) is a before/text/after context-window adjacency warning, not a
verbatim failure: each of the three fields (before, text, after) independently verifies against
`text/pa1800.txt`; the OCR's page-break/running-head artifacts on this heavily degraded scan
break exact contiguity of the three fields taken together. Not treated as a FAIL, consistent
with the same class of WARN accepted in the geneva-academy verification.

## 2. Quote-by-quote
All 17 quote_ids: `text` field verbatim match PASS (per script). Spot-checked 6 of 17 (bms-q001,
bms-q004, bms-q009, bms-q011, bms-q012, bms-q016) by opening the named text/ file at the quoted
string and confirming the surrounding sentence and the document_id match the dossier's citation.
PASS for all six. The four quotes drawn from `pa1800` (bms-q005 through bms-q010) sit inside a
document sources.csv already flags at 4-7 OCR errors/100 words; the exact quoted strings
themselves were independently re-confirmed present verbatim in the file (as the script also
confirms), so this elevated error rate in the surrounding unquoted narrative is noted but is not
a quote-level FAIL.

## 3. Field-by-field
F1-F9: PASS — each summary sentence is supported by its cited quote_id(s) and does not claim
more than the quotes show. F8 (doctrinal basis) is correctly marked NOT FOUND rather than
invented. F9's antecedents claim (Carey citing denominational precedent) matches bms-q002/q003.
A, C, L, T sections: correctly marked NOT APPLICABLE with a stated reason (sending society, not
a school) rather than stretched to fit — consistent with Rule 5 (write NOT FOUND/UNVERIFIED,
never blank) and with how abcfm.md handles the same situation for a non-educational institution.
M1-M5: PASS, each supported; M4 (the "expect/attempt great things" motto) is correctly downgraded
to UNVERIFIED in its precise wording, since the sources read this session give only the sermon's
two heads ("expe[ct]... attempt"), not the familiar paraphrase verbatim — this is the correct,
cautious call and must not be strengthened in the chapter.
S1-S6: PASS. S3 (numbers) is correctly left UNVERIFIED for a Form-of-Agreement signature count,
distinguishing a web-scouted number (not fetched/read this session) from a directly attested one
(the twelve-name 1792 subscription list, which IS attested). S5 (deaths) correctly NOT FOUND.
R1-R4: PASS, each supported; R1 correctly flags as UNVERIFIED whether Carey's own signature
appears on the Form of Agreement (the fetched passage runs from the Agreement's close directly
into an unrelated appendix item, without a signature block).

## 4. Secondary citations checked
None of this dossier's facts rest on a [SECONDARY: ...] citation; Smith 1885's narrative
(secondary) is not quoted for any fact — only its Appendix I reprint of the primary 1805 Form of
Agreement is used, and that reprint is correctly treated as PRIMARY (same convention as Anderson
1861 in the abcfm dossier). No secondary-citation checks are therefore required for this dossier.

## 5. Dates and numbers vs known-facts sheet
- 2 Oct 1792, Kettering, founding of the Society: matches `02_institution_roster.md` row
  `baptist-missionary-society` [HIGH]. PASS.
- Founders Carey, Fuller, Ryland, Sutcliff, Pearce (roster) vs. dossier's F3 committee list
  (Ryland, Hogg, Carey, Sutcliff, Fuller) plus Pearce among the wider subscriber list (F3): PASS,
  consistent — the roster's founder list and the dossier's committee/subscriber lists are not in
  conflict (Pearce appears among the twelve subscribers, not the five-man committee).
- Serampore Form of Agreement, 7 Oct 1805: matches roster date [HIGH]. PASS.

## 6. Verdict
VERIFIED (zero FAILs on the 17 quote records and the 8 charter passages). No RETURNED items.
