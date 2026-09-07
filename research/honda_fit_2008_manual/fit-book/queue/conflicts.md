# Conflicts

Two sources disagree and neither was picked as a winner (per RESEARCH_PLAN.md Rule 6).
Resolve these during the verification pass, ideally with a direct (non-WebSearch-snippet)
read of the primary source.

## WP-02 Recalls, TSBs, and known defects (2026-09-07)

### C-001: NHTSA campaign 10V624 (headlight low-beam wiring) exact start date — RESOLVED (verify pass, 2026-09-07)
- Source A says the recall was "expected to begin on or before January 21, 2011."
- Source B says the recall "began January 24, 2011."
- Both agree on the campaign number (10V624000), the 143,083-unit count, and the defect
  description. Only the exact start date conflicts, by 3 days.
- **Resolution:** Checked directly against NHTSA's own campaign record
  (`api.nhtsa.gov/recalls/campaignNumber?campaignNumber=10V624000`). The record's own remedy
  text states "THE SAFETY RECALL BEGAN ON JANUARY 24, 2011." Source B is correct.
- Related fact card: F-WP02-003

### C-002: NHTSA campaign 10V033 (power window switch, first campaign) — alternate ID
- Most sources confirm campaign number 10V033000, announced ~January 29, 2010.
- One single source snippet referenced this same recall as "12V-073" instead. This was not
  corroborated anywhere else and is logged as an unresolved oddity, not a real conflict
  (3 independent sources agree on 10V033 vs. 1 outlier on 12V-073).
- Related fact card: F-WP02-001

### C-003: Honda Service Bulletin 13-021 (rocker arm oil pressure switch) — applicable models — RESOLVED (verify pass, 2026-09-07)
- Source A: covers "2007-2011 Honda Fit" (among other models).
- Source B: covers "2012-13 Civic models (except Si and Hybrid)" and does not mention Fit.
- These cannot both be accurate as stated for the same bulletin number. Do not treat this
  TSB as confirmed-applicable to the 2008 Fit until resolved.
- **Resolution:** Downloaded and read the actual bulletin PDF directly
  (`static.nhtsa.gov/odi/tsbs/2015/SB-10098946-5233.pdf`, Honda Service Bulletin 13-021,
  version 3, Sept 16 2015). Its AFFECTED VEHICLES table lists eight rows including
  "2007–11 Fit — ALL — ALL" AND "2012-13 Civic — ALL except Si and Hybrid — ALL" as two
  *separate* rows of the same multi-model bulletin. Source A was reading the Fit row; Source B
  was reading the Civic row and incorrectly reported it as the bulletin's only scope. **The
  bulletin does cover the 2008 Fit** (within the confirmed 2007-11 range) — the fact card's own
  "unclear, do not use" conclusion should be upgraded to "confirmed applicable."
- Related fact card: F-WP02-009

### C-004: 2008 Fit total TSB count — STILL UNRESOLVED (verify pass, 2026-09-07)
- Source A: "19 technical service bulletins issued for the 2008 Honda Fit."
- Source B: "42 TSBs" for the 2008 Honda Fit.
- Likely different counting methodology (e.g., Fit-specific bulletins vs. all bulletins that
  happen to list the Fit among covered models), but unresolved.
- **Verification attempt:** Both source domains (obd-codes.com, carcomplaints.com) return
  Cloudflare 403 blocks to this session's network egress, both via direct curl and via the
  r.jina.ai reader proxy. NHTSA has no public TSB-count-by-vehicle API endpoint (only
  individual TSB PDFs, which can be read directly once you have the exact filename). Genuinely
  needs a session with different network access — logged to queue/blocked.md.
- Related fact card: F-WP02-010

### C-005: NHTSA campaign 20V770 (driveshaft corrosion) — model year range covered — RESOLVED (verify pass, 2026-09-07)
- Source A: "2007-2013" Fit.
- Source B: "2007-2014" Fit.
- Both agree the range starts at 2007 (so 2008 is covered either way) and that the recall is
  manual-transmission-only, but the end year conflicts.
- **Resolution:** NHTSA's own campaign summary (`api.nhtsa.gov/recalls/campaignNumber?campaignNumber=20V770000`)
  states the recall covers "2007-2008 Honda Fit vehicles with a manual transmission" (all
  states) plus "2009-2013 Honda Fit vehicles" (limited to specific salt-belt states/registration
  history). The correct end year is **2013**, matching Source A. Source B's "2007-2014" is not
  supported by the primary record.
- Related fact card: F-WP02-005

## WP-01 Identify the car exactly (2026-09-07)

### C-007: Cargo volume, rear seats folded down
- Honda's own 2008 US press release (Tier 1): 56.8 cu ft.
- A general web-search-aggregator summary (sources not clearly separated by generation):
  41.9 cu ft — likely a conflated 2009+ GE8 figure, treated as YEAR-UNCONFIRMED and not used.
- CarTimeline's GD3-specific spec page (Tier 3): 59.2 cu ft (1,677 L), which the source itself
  flags as using a "method [that] varies by region."
- Used the Honda Tier 1 US press release value (56.8 cu ft) as the book's primary value.
- Related fact card: F-WP01-020

### C-008: 2008 Fit exterior color list — 7 vs. more colors
- CarsDirect (Tier 3) lists exactly 7 exterior colors for the 2008 Fit: Milano Red, Storm
  Silver Metallic, Tidewater Blue Metallic, Vivid Blue Pearl, Blackberry Pearl, Blaze Orange
  Metallic, Nighthawk Black Pearl.
- A touch-up-paint vendor's color list (possibly a sitewide/non-year-filtered dropdown, not
  reliably confirmed as 2008-Fit-specific) also surfaced "Taffeta White" and "Alabaster Silver
  Metallic" as 2008 Fit colors, which do not appear on the CarsDirect list.
- Honda's own overview press release (Tier 1) only confirms the two colors NEW for 2008
  (Tidewater Blue Metallic, Blackberry Pearl), which are consistent with, but do not fully
  validate, either full list above.
- Related fact card: F-WP01-030

### C-009: Base-trim auxiliary audio jack, tilt steering wheel, intermittent wipers — RESOLVED in WP-05 (2026-09-07)
- Honda's US press release (Tier 1) describes the auxiliary audio jack as part of the Fit
  Sport's premium audio system only, implying the US base Fit's 4-speaker AM/FM/CD system did
  NOT have an aux jack.
- auto123.com's Canadian-market "Base/DX" page (Tier 3) lists "AM/FM stereo radio with
  auxiliary audio jack" as a standard feature, along with a tilt steering wheel and
  intermittent wipers.
- This may be a genuine US-vs-Canada trim equipment difference (plausible, since Canadian DX
  and US "Fit" base are not guaranteed to be identically equipped) rather than a factual
  error, but it was not resolved with the sources available this pass. The tilt-wheel and
  intermittent-wiper claims for the US base trim are recorded at medium confidence (single
  Tier 3 source; a second, US-specific source could not be fetched — Edmunds and Cars.com
  feature pages returned HTTP 403 to automated fetch).
- Related fact card: F-WP01-027
- **RESOLUTION (WP-05):** Confirmed as a genuine US-vs-Canada equipment difference, not an
  error. The official 2007 Honda Fit owner's manual (Tier 1, read in full via r.jina.ai —
  techinfo.honda.com/rjanisis/pubs/om/AA0707/AA0707OM.pdf) states the auxiliary input jack is
  fitted to "U.S. Sport and all Canadian models" — meaning ALL Canadian-market Fits (including
  the Canadian base/DX trim auto123.com described) get the aux jack, while the US base Fit does
  not. Independently corroborated by Honda's official "2008 Honda Fit - Features" press release
  (Tier 1, hondanews.com), whose equipment table places "MP3/Auxiliary Input Jack" with a bullet
  ONLY in the "Fit Sport" column, not the base "Fit" column. So: US base Fit has NO aux jack
  (auto123's claim does not apply to it); the Canadian DX/base-equivalent trim DOES have one
  (auto123's claim is correct for its own market). The tilt steering wheel and intermittent
  wipers are also now confirmed at HIGH confidence (upgraded from medium) via the same two Tier 1
  sources: "Adjustable Steering Column" and "2-Speed/Intermittent Windshield Wipers" both carry a
  bullet under the base "Fit" column in Honda's press equipment table, and the owner's manual's
  own steering-wheel-adjustment steps describe a tilt-only (no telescoping) mechanism available
  on all US models. See facts/WP-05.md F-WP05-007, F-WP05-024, F-WP05-025, F-WP05-027.

### C-006: 2008 Fit top NHTSA complaint category counts
- Two different search summaries of what appears to be the same underlying CarComplaints.com
  /NHTSA complaint data gave different counts for the same categories (e.g., Air Bags 55 vs.
  61; Engine 22 vs. 41; Body/Paint not in top 10 vs. 43 complaints; Brakes 13 vs. 24).
  The relative ranking (air bags highest, engine/powertrain and brakes prominent) was
  consistent; exact counts were not.
- Related fact card: F-WP02-011

## WP-09 The maintenance schedule (2026-09-07)

### C-010: Generic (non-Fit-specific) Maintenance Minder code-table content doesn't match the Fit's own manual
- A KBB "2008 Honda Fit Service Schedules & Maintenance Pricing" page, and other dealer-site
  explainers of "Honda Maintenance Minder codes," describe a generic Code A/B/1-7 system used
  across many Honda models, including "Code 4: Replace spark plugs & timing belt; Inspect
  water pump, and valve clearance" and "Code 6: Replace rear differential fluid (if
  applicable)."
- The 2008 Fit's own owner's manual (S-023, read directly) has NO timing belt (the L15A1 uses
  a timing chain — see F-WP09-044) and, being front-wheel-drive only in the US market, has no
  rear differential fluid item either. The Fit's own maintenance item table also does not
  print the traditional dual "Normal Conditions / Severe Conditions" mileage chart that some
  generic explainers imply exists for every current Honda (see F-WP09-001, F-WP09-015).
- Resolution: the Fit-specific Tier 1 owner's manual (S-023) was used as the source of record
  for all WP-09 facts; the generic dealer-explainer content was not used for any Fit-specific
  numeric claim, only as background on how the Maintenance Minder A/B/number system works in
  general. Recorded here, not silently discarded, per Rule 6.
- Related fact cards: F-WP09-001, F-WP09-015, F-WP09-044

### C-011: AutoPadre battery-size table gives a different group size for 2007 vs. 2008, same generation
- autopadre.com's "Honda Fit Battery Size (2007-2020)" table lists the 2007 Fit as group size
  51R but the 2008 Fit (same GD3 generation, same engine) as group size 151R.
- A first-generation Fit should not plausibly have changed OEM battery group size between its
  first and second (final) US model year with no other spec change noted anywhere else found
  this session — this looks like a data-entry inconsistency in that one aggregator table
  rather than a real running change, but it was not resolved with the sources available this
  pass.
- The 2008-specific value (151R) is used on the fact card because it is the row that names
  2008 directly, and because it is independently corroborated by a Tier 4 community source
  (qlmotorsport.com "Honda Fit Battery Swap – 151R to 51R," describing 151R as the factory
  size and 51R as a popular higher-CCA upgrade) and by a WebSearch AI synthesis of
  oreillyauto.com/autozone.com OEM-fitment listings for the 2008 Fit specifically.
- Related fact card: F-WP09-046

## WP-08 Paperwork and buying used (2026-09-07)

### C-012: Typical used 2008 Fit price/value figures differ sharply across KBB, Edmunds, and CarGurus
- KBB (direct page read, S-029): dealer retail range $4,575 (Sport)-$5,850 (base Hatchback
  4D); KBB's own trade-in estimate $2,550-$2,825; KBB's own private-party estimate
  $3,925-$5,125.
- Edmunds (search-snippet-sourced, S-030): appraisal value range $763-$2,189; example
  "Clean"-condition trade-in ~$1,022, private-party ~$1,446 — noticeably lower than KBB's
  trade-in/private-party figures for what should be the same general metric.
- CarGurus (search-snippet-sourced, S-030): average asking price ~$6,521 across current
  listings, with metro-level ranges as wide as $2,600-$15,200+ — noticeably higher than either
  appraisal tool, because these are real asking prices, not an appraisal estimate.
- **Not resolved to a single number.** The three sources are not all measuring the same thing
  (trade-in estimate vs. private-party estimate vs. dealer retail estimate vs. real current
  asking prices), so the disagreement is partly explainable by definition differences rather
  than a straightforward factual contradiction — but KBB's and Edmunds' *trade-in* figures
  specifically ($2,550-$2,825 vs. ~$1,022) do disagree on what should be the same metric, and
  that gap was not resolved this pass (Edmunds could not be directly fetched to check whether
  its lower number reflects a specific default mileage/condition assumption).
- Recorded in facts/WP-08.md as F-WP08-016 (KBB), F-WP08-017 (Edmunds/CarGurus), and F-WP08-018
  (reconciliation/reader guidance, not a forced single number).
- Related fact cards: F-WP08-016, F-WP08-017, F-WP08-018.

## WP-12 Diagnostics (2026-09-07)

### C-013: Honda code P2649 — two sources give different definitions
- autocodes.com states P2649 is "Rocker Arm Oil Control Solenoid Circuit High Voltage" — the
  same VTEC oil-control-solenoid family as P2646/P2647 (rocker arm oil PRESSURE SWITCH), used
  in facts/WP-12.md F-WP12-012.
- A separate generic code-lookup aggregator (dot.report) surfaced in an early search titled
  P2649 as "'B' Rocker Arm Actuator System Stuck On (Bank 1)" — a different fault type
  (a stuck mechanical actuator state, not a solenoid circuit-voltage fault) and a different
  bank-letter convention ("B" vs. the "A"-only system the Fit's SOHC VTEC uses).
- Not resolved this pass. F-WP12-012 uses the autocodes.com definition because it is
  consistent with the same rocker-arm-oil-pressure-switch family already confirmed applicable
  to this exact car via Honda's own TSB 13-021 (F-WP12-010/011), and because a real FitFreak.net
  GD3 thread's fix for P2649 (a small O-ring/gasket) matches TSB 13-021's own remedy parts list
  for this code family — but the dot.report definition was not independently disproven, just
  not used.
- Related fact card: F-WP12-012

### C-014: P0420 Fit-specific corroboration is weaker than the other misfire/sensor codes
- Not a source-vs-source disagreement, but flagged per the plan's spirit of recording
  weaknesses rather than hiding them: F-WP12-009 (P0420, catalytic converter efficiency) relies
  on one thread explicitly tagged to the "1st Generation (GD 01-08)" sub-forum, plus a second
  thread in FitFreak's general "Fit-talk" sub-forum whose exact model year could not be
  independently confirmed from the snippet alone (it is not GE8-tagged either — it is simply
  unlabeled by generation).
- This does not currently contradict anything, so it is not a true "two sources disagree"
  conflict, but it means F-WP12-009 is weaker than most of the other 19 codes in the same file
  and should be the first one re-checked if a future session regains fuller access to
  fitfreak.net.
- Related fact card: F-WP12-009

## WP-11 Maintenance procedures (2026-09-07)

### C-011: Total engine coolant system capacity
- The 2008 Fit owner's manual's own Specifications table (S-001/S-023, Tier 1) gives capacity
  figures in this range for engine coolant: approximately 3.7 to 5.4 L, depending on how the
  table's columns (M/T vs A/T, "change" vs "total") are read -- the table itself extracts from
  the PDF with jumbled/interleaved cell order, a caution already noted on S-023's own source
  card.
- hfitinfo.com's repair-manual-style "Coolant Replacement" page (S-WP11-A, Tier 3, generation
  label unconfirmed for GD3) instead gives: M/T model, at coolant change, 4.37 L; A/T model, at
  coolant change, 4.47 L; after an engine overhaul, 4.86 L (M/T) / 4.96 L (A/T) -- all noticeably
  higher than the low end of the owner's manual's own range.
- Not resolved this pass. Used in procedures/WP-11-15.md as a stated range with a note to check
  Appendix B/WP-09 for the verified number, rather than picking a winner.
- Related procedure: P-WP11-15 (Drain and fill the coolant)

## WP-10 Torque specs and fitment numbers (2026-09-07)

### C-015: Spark plug torque source's own generation label doesn't match GD3
- The only source giving a specific numeric spark plug torque with full procedure text
  (S-WP10-F, hfitinfo.com "Ignition Coil and Spark Plug Removal/Installation") files itself
  under hfitinfo's own "Second generation (2007-2026)" repair-manual bucket — not the "First
  generation (2001-2008)" bucket this project otherwise uses for the GD3 (2007-2008 US Fit).
- A second, independent source (S-WP10-E, a JustAnswer Q&A) names a "2007 Honda Fit" (which IS
  GD3) directly and cites the identical number (13 lbf-ft / 18 N-m), so the VALUE is not really
  in dispute — the open question is only whether the two Fit generations' L15A-family SOHC
  engines share the exact same spark plug thread/torque spec, which was not independently
  confirmed by a document naming GD3/2008/first-generation explicitly for this specific number.
- Not resolved this pass. F-WP10-004 uses 13 lbf-ft (18 N-m) at Confidence: medium rather than
  high, with this caution stated on the card itself.
- Related fact card: F-WP10-004

### C-016: Caliper slide-pin torque's second-best corroborating source names the wrong generation
- F-WP10-007 (caliper slide/guide pin torque) has one clean GD3-relevant source (S-WP10-B,
  FitFreak.net, front ~26 lbf-ft / rear 17 lbf-ft) and one additional source giving a very
  similar front figure (~25 lbf-ft), but that second source (S-WP10-K, JustAnswer) is explicitly
  about a "2015 Honda Fit" — the third-generation GK chassis, a different car from the GD3 per
  D2/D8.
- The two numbers (25 vs. 26 lbf-ft) are close enough that they may reflect the same real
  fastener spec carried across Fit generations, but D8 does not allow treating a GK-only source
  as confirmation for a GD3 fact, so S-WP10-K was NOT counted as the second independent Tier 4
  source required by RESEARCH_PLAN.md Section 3 for the front slide-pin figure.
- Not resolved this pass. F-WP10-007's front slide-pin value is recorded as single-GD3-sourced,
  low confidence, rather than upgraded using the GK source; S-WP10-K is kept on file as context
  only, not cited as a Fact Card source.
- Related fact card: F-WP10-007
