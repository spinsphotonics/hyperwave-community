STATUS: DONE
TICKET: S-omf
ROLE: Scout
INPUTS READ: plan/02_institution_roster.md (row `omf`); plan/05_repositories_and_search_strings.md (`omf` section)

# Scout notes -- omf

This is a coda ticket (roster Tier A "as coda to CIM chapter"). Per the task instruction, its job is narrower than a full institution scout: confirm the rename year and locate what changed in name, structure, and *Principles and Practice* when CIM became OMF, plus where OMF operates today.

## Target documents (from the roster row)

1. OMF *Principles and Practice* (1960s revision) -- roster's only named "key primary document."

## Searches run (roster's two search strings, plus follow-on searches needed to find anything usable)

- `"Overseas Missionary Fellowship" "Principles and Practice"` -- WebSearch. No archive.org, HathiTrust, or Google Books full-text hit for a digitized post-1964 P&P. One lead: an omf.org-hosted thesis page ("...Hong Kong Council of the Overseas Missionary Fellowship...") whose search snippet mentions Council members signing "the last page of the 'Principles and Practice of the OMF'" -- BLOCKED, omf.org would not serve the page to curl (see sources.csv row `omf-org-mrt-hongkong-council`).
- `China Inland Mission renamed Overseas Missionary Fellowship 1964` -- WebSearch. Multiple secondary hits (Wikipedia, AIM25/Archives Hub, Wheaton finding aid, omf.org's own "our story" page). Fetched what could be fetched; see sources.csv.
- Follow-on: `archive.org "Overseas Missionary Fellowship" "Principles and Practice"` -- no hit.
- Follow-on: `"China's Millions" 1964/1965 archive.org` -- Internet Archive holds no post-1935 issues of *China's Millions* in the open collection (all digitized volumes found are 1875-1935; a 1965-dated "China's Millions" item, `bwb_Y0-AHD-568`, is an unrelated access-restricted book by Anna Louise Strong that happens to share the title).
- Follow-on: `archive.org title search "Passion for the Impossible"` -- found Leslie Lyall's 1965 CIM/OMF centenary history, the most promising lead this session, but it is access-restricted (controlled digital lending); see the `.REQUEST.md` file in `text/`.
- Follow-on: `archive.org title search "Story of Faith Missions" Fiedler` -- found (roster's Secondary source), also access-restricted; not read for content (Scout does not read for content).

## What is BLOCKED

- **omf.org** (every page tried: `/about-us/our-story/`, the Hong Kong Council thesis page, and regional variants) returns an interactive `sgcaptcha` bot-check redirect to every `curl` request this session, with or without a browser-spoofed User-Agent. Cannot be fetched by any method available this session.
- **archiveshub.jisc.ac.uk** (the direct SOAS Archives Hub finding aid `gb102-cim`, both the HTML page and its raw EAD XML export) returns a Cloudflare interactive challenge page ("Just a moment...").
- **archives.wheaton.edu** (Billy Graham Center Archives finding aid for the North American OMF/CIM records) returns a Cloudflare Turnstile human-verification gate.
- **archives.soas.ac.uk** name-authority page for OMF loads (no bot-check) but its content is rendered entirely client-side by JavaScript; the fetched HTML is empty of substance.
- Internet Archive's **`passionforimposs0000lesl`** (Lyall, 1965) and **`storyoffaithmiss0000fied`** (Fiedler, 1994) are both controlled-digital-lending, access-restricted items; their listed OCR text files return HTTP 401 when downloaded directly.
- **web.archive.org** (the Wayback Machine) is not in this environment's network egress allowlist; direct fetches to it fail immediately with "Host not in allowlist."

## What was found and used

Despite the above, three independent, openly accessible secondary sources were fetched in full and used: the Wikipedia article "OMF International," the AIM25 archival catalogue description for the CIM/OMF records held at SOAS (which mirrors the blocked Archives Hub page's own narrative), and the Wikipedia article on J. O. Sanders (CIM/OMF General Director 1954-1969). These three independently corroborate a rename year of 1964 (AIM25 dates the specific Overseas Council meeting to October 1964). See `dossier.md` and `discrepancies.md` for a discrepancy between two dates given inside the Wikipedia article itself (its own 1950s-chronology entry says "14 October 1954" for what reads as the same restructuring meeting AIM25 dates to October 1964).

No primary OMF-published document (a P&P revision, an annual report, a *China's Millions* closing issue, or Lyall's 1965 centenary history) could be opened this session. Every dossier field that would require one is marked NOT FOUND or is filled from a labeled SECONDARY source, per the rules.
