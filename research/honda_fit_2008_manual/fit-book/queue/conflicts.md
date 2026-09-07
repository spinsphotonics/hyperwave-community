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
