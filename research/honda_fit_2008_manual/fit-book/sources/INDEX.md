# WP-00 Source Library Index

Built 2026-09-07. All URLs below are real, live URLs surfaced by WebSearch results (not invented).
Note: WebFetch (direct page loading) was blocked by this session's network egress policy for every
domain tested, including owners.honda.com, helminc.com, nhtsa.gov, fueleconomy.gov, iihs.org,
fitfreak.net, hondapartsnow.com, en.wikipedia.org, and even google.com — so every card below was
verified via a live WebSearch query and its returned result snippet, not a direct WebFetch page load.
This should be re-attempted with a working fetch tool before Phase 2+ facts are finalized.

| Source ID | Title | Tier | Type | Reachable / Blocked | Paywalled |
|---|---|---|---|---|---|
| S-001 | Owners Manual for 2008 Honda Fit (owners.honda.com) | 1 | owner's manual | Reachable (search-confirmed; WebFetch blocked) | No |
| S-002 | 2007-2008 Fit (KA) Service Manual (Helm Inc.) | 1 | service manual | Reachable listing (search-confirmed); manual itself paid | Yes |
| S-003 | Honda Fit 07-13 Haynes Repair Manual 42030 | 3 | service manual | Reachable listing (search-confirmed); manual itself paid | Yes |
| S-004 | NHTSA Vehicle Detail Search - 2008 Honda Fit (Recalls) | 2 | recall | Reachable (search-confirmed; WebFetch blocked) | No |
| S-005 | NHTSA Vehicle Detail Search - 2008 Honda Fit (Complaints/Investigations) | 2 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-006 | IIHS 2008 Honda Fit 4-door wagon ratings | 2 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-007 | fueleconomy.gov - Fuel Economy of the 2008 Honda Fit | 2 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-008 | HondaPartsNow.com - 2008 Fit 5dr Base KA 5AT Parts | 1 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-009 | Wheel-Size.com - Honda Fit GD [2001-2008] fitment | 3 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-010 | FitFreak.net - 1st Generation (GD 01-08) subforum | 4 | forum post | Reachable (search-confirmed; WebFetch blocked) | No |
| S-011 | FitFreak.net - "Things to know before buying a FIT" | 4 | forum post | Reachable (search-confirmed; WebFetch blocked) | No |
| S-012 | FitFreak.net - "Which GD do I have?" | 4 | forum post | Reachable (search-confirmed; WebFetch blocked) | No |
| S-013 | SCCA National Solo Rules (2025 edition) | 2 | rulebook | Reachable (search-confirmed; WebFetch blocked) | No |
| S-014 | NASA Club Codes and Regulations (2026.3 edition) | 2 | rulebook | Reachable (search-confirmed; WebFetch blocked) | No |
| S-015 | HondaNews - 2008 Honda Fit Specifications press release | 1 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-016 | Wikipedia - Honda Fit (first generation) | 3 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-017 | Edmunds - 2008 Honda Fit review/specs | 3 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-018 | Honda-Tech.com - Honda Fit forum | 4 | forum post | Reachable (search-confirmed; WebFetch blocked) | No |
| S-019 | HondaNews - Master Power Window Switch Recall statement (13V-260) | 1 | recall | Reachable (search-confirmed; WebFetch blocked) | No |
| S-020 | Center for Auto Safety - 2008 Honda Fit recalls/complaints | 3 | article | Reachable (search-confirmed; WebFetch blocked) | No |
| S-021 | Honda Fit 2001-2008 Owners Manual mirror, "Instruments and Controls"/"Driving" chapters (hfitinfo.com) | 3 | owner's manual | Reachable, directly read via curl (WP-04) | No |
| S-022 | HONDA 2008 FIT OWNER'S MANUAL, page-image/OCR viewer (manualslib.com) | 1 | owner's manual | Reachable, directly read via curl (WP-04) | No |
| S-023 | 2008 Fit Owner's Manual (AAA0808OM.pdf) — Maintenance/Technical Information chapters (techinfo.honda.com) | 1 | owner's manual | Reachable via r.jina.ai reader proxy (WP-09) | No |
| S-024 | workshop-manuals.com - Fit L4-1.5L (2008) fluids/spark plug/valve clearance specs | 3 | service manual | Reachable via r.jina.ai reader proxy (WP-09) | No |
| S-025 | Boslla - 2007-2018 Honda Fit bulb size guide, 2007-2008 table | 3 | article | Reachable via r.jina.ai reader proxy (WP-09) | No |
| S-026 | AutoPadre - Honda Fit battery size and wiper blade size (2007-2020) | 3 | article | Reachable via r.jina.ai reader proxy (WP-09) | No |
| S-027 | HondaPartsNow - genuine air filter (17220-PWA-505) and cabin filter (80291-SAA-J01) pages | 1 | article | Reachable via r.jina.ai reader proxy (WP-09) | No |

## Totals
- Total Source Cards: 27 (target was 15+; 20 built in WP-00, S-021/S-022 added in WP-04, S-023 through S-027 added in WP-09)
- Reachable (real live URL/content confirmed): 27 / 27
- Directly loaded via WebFetch: 0 / 27 — WebFetch itself stayed blocked all session, but from WP-01 onward direct `curl` and the `https://r.jina.ai/` reader proxy (ADAPT-2) successfully read real page/PDF content directly for many sources, including S-001 (WP-07), S-021/S-022 (WP-04), and S-015/S-023 through S-027 (WP-09) — see queue/blocked.md and queue/decisions_needed.md
- Fully paywalled with no free equivalent for the primary content: 2 (S-002 Helm factory service manual, S-003 Haynes manual) — both logged to queue/blocked.md

### Tier breakdown
- Tier 1: S-001, S-002, S-008, S-015, S-019, S-022, S-023, S-027 (8 sources)
- Tier 2: S-004, S-005, S-006, S-007, S-013, S-014 (6 sources)
- Tier 3: S-003, S-009, S-016, S-017, S-020, S-021, S-024, S-025, S-026 (9 sources)
- Tier 4: S-010, S-011, S-012, S-018 (4 sources)

## Notes for next work packages
- WP-01 should re-derive engine/spec numbers directly from S-015 (Honda press specs) and S-017 (Edmunds), not from this index's summarized snippets.
- WP-02 should treat S-004, S-005, S-019, S-020 as its primary input set, and confirm the Takata airbag inflator recall and any ignition-switch complaints separately (search turned up a Takata fact sheet at hondanews.com but no confirmed 2008 Fit-specific NHTSA campaign number yet — flag as UNVERIFIED until found).
- S-013 (SCCA) and S-014 (NASA) are general rulebooks, not Fit-specific; WP-20 must locate the exact Street-category class table inside each before citing a class for the stock 2008 Fit.
- Two independent Tier 4 community threads are on file (S-011, S-012) plus two Tier 4 forum hubs (S-010, S-018) to seed WP-02/WP-13 "community-reported" facts, which per the plan require two independent Tier 4 sources each.
