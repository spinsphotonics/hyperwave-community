# Template-compliance spot-check: procedure cards (not a full re-verification)

**Scope:** 8 procedure cards, chosen to cover all four requested WP groups, checked against three
template rules from the Procedure Card template in RESEARCH_PLAN.md Section 5: (1) safety warnings
appear before the numbered steps, (2) no step longer than 25 words, (3) the card cites real source
IDs — meaning IDs that trace to an actual file in `sources/`.

Cards checked: `procedures/WP-03-1.md`, `WP-03-2.md`, `WP-05-2.md`, `WP-05-4.md`, `WP-06-1.md`,
`WP-06-2.md`, `WP-07-1.md`, `WP-07-3.md`.

## Results

| Card | Safety warnings before steps? | All steps ≤25 words? | Real source IDs? |
|---|---|---|---|
| WP-03-1 (Open the hood) | PASS | PASS | PASS — `S-001` (sources/S-001.md exists) |
| WP-03-2 (Adjust seat/mirrors) | PASS | PASS (longest step is exactly 25 words) | PASS — `S-001` |
| WP-05-2 (Tall Mode) | PASS | **FAIL** — the "return to normal seating" step is 54 words | **FAIL** — `S-HONDA-OM2007` has no matching file in `sources/` |
| WP-05-4 (Refresh Mode) | PASS | **FAIL** — the "return to normal driving position" step is 39 words | **FAIL** — `S-HONDA-OM2007` has no matching file in `sources/` |
| WP-06-1 (Hill start, manual) | PASS | PASS | **FAIL** — `S-HFITINFO-MT`, `S-HFITINFO-PARK` have no matching files in `sources/` |
| WP-06-2 (Drive a manual, first time) | PASS | PASS | **FAIL** — `S-HFITINFO-START`, `S-HFITINFO-MT`, `S-HFITINFO-DRIVTOC` have no matching files in `sources/` |
| WP-07-1 (Change a flat tire) | PASS | PASS | PASS — `S-001` |
| WP-07-3 (Car is overheating) | PASS | PASS | PASS — `S-001` |

**Safety warnings before steps: 8/8 PASS.** Every card lists its "Safety warnings:" block before
the "Steps:" block, with no exceptions.

**Steps ≤25 words: 6/8 PASS, 2/8 FAIL.** WP-05-2 and WP-05-4 each fold their entire "how to undo
this mode" sub-procedure into one long numbered step (54 words and 39 words respectively) instead
of breaking it into separate numbered steps. This also violates the template's "one action per
step" rule, not just the word count — each of those two steps actually bundles 3-4 distinct
actions (e.g., WP-05-4 step 7 bundles: hold the seat-back and pivot it up, control the motion by
hand, reinstall both head restraints, and lock both seats — four actions in one step).

**Real source IDs: 4/8 PASS, 4/8 FAIL.** This is the more significant finding. WP-05-2, WP-05-4,
WP-06-1, and WP-06-2 all cite source IDs in a mnemonic style (`S-HONDA-OM2007`,
`S-HFITINFO-MT`, `S-HFITINFO-PARK`, `S-HFITINFO-START`, `S-HFITINFO-DRIVTOC`) that do not
correspond to any file in `sources/` — I checked with `ls sources/` and a recursive grep and found
no card for any of them, under either the plain numbered scheme (`sources/S-###.md`) or the
WP-scoped collision-safe scheme (`sources/S-WP##-<letter>.md`) that ADAPT-3 establishes for new
mid-task sources. **This is very likely a broken citation trail rather than an invented fact**:
the URLs implied by the mnemonic names (hofi-470, hofi-471, hofi-474, hofi-469 — all in
hfitinfo.com's "Driving" chapter range) fall squarely inside the page range that
`sources/S-021.md` already documents and cites as reachable, real, Tier 3 content verbatim-
matching the actual Honda manual. So the underlying material these procedure cards point to is
almost certainly real — but as written, a reader (or a future verifier) cannot actually trace
`S-HFITINFO-MT` back to a Source Card the way the template requires ("Every fact gets a source URL
and access date. No URL, no fact" — Section 1, Rule 5). The same pattern (`S-HONDA-FEATURES`,
`S-HONDA-OVERVIEW`, `S-FITFREAK-MAGICSEAT`) also appears in the underlying `facts/WP-05.md` and
`facts/WP-06.md`, not just in the procedure cards derived from them — this is a WP-05/WP-06
research-time habit, not a one-off procedure-writing slip.

I did not edit `facts/WP-05.md`, `facts/WP-06.md`, or the four affected procedure cards — this
task asked for a spot-check, not a fix, and this finding affects the underlying fact files as
much as the procedure cards. Recommend a follow-up pass on WP-05 and WP-06 (fact files and their
procedure cards together) to either (a) create real, WP-scoped Source Cards for these mnemonic IDs
(they clearly map to real hfitinfo.com/Honda pages already fetched and quoted, so this should be
a quick backfill, not new research) or (b) re-point every citation to the existing `sources/S-021.md`
where the URL range already covers it.

## Bottom line

- **Safety-warnings-first: 8/8 compliant.**
- **Step length/one-action-per-step: 6/8 compliant** (WP-05-2, WP-05-4 each have one oversized,
  multi-action step — an easy fix: split the "how to undo this" tail into its own separately
  numbered steps).
- **Real, traceable source IDs: 4/8 compliant** (WP-05-2, WP-05-4, WP-06-1, WP-06-2 cite IDs with
  no corresponding Source Card file — a citation-trail gap, not a fabricated-fact problem, but a
  real template violation that should be fixed before these chapters are considered done).
