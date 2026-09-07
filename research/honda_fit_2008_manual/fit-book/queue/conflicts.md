# Conflicts

Two sources disagree and neither was picked as a winner (per RESEARCH_PLAN.md Rule 6).
Resolve these during the verification pass, ideally with a direct (non-WebSearch-snippet)
read of the primary source.

## WP-02 Recalls, TSBs, and known defects (2026-09-07)

### C-001: NHTSA campaign 10V624 (headlight low-beam wiring) exact start date
- Source A says the recall was "expected to begin on or before January 21, 2011."
- Source B says the recall "began January 24, 2011."
- Both agree on the campaign number (10V624000), the 143,083-unit count, and the defect
  description. Only the exact start date conflicts, by 3 days.
- Related fact card: F-WP02-003

### C-002: NHTSA campaign 10V033 (power window switch, first campaign) — alternate ID
- Most sources confirm campaign number 10V033000, announced ~January 29, 2010.
- One single source snippet referenced this same recall as "12V-073" instead. This was not
  corroborated anywhere else and is logged as an unresolved oddity, not a real conflict
  (3 independent sources agree on 10V033 vs. 1 outlier on 12V-073).
- Related fact card: F-WP02-001

### C-003: Honda Service Bulletin 13-021 (rocker arm oil pressure switch) — applicable models
- Source A: covers "2007-2011 Honda Fit" (among other models).
- Source B: covers "2012-13 Civic models (except Si and Hybrid)" and does not mention Fit.
- These cannot both be accurate as stated for the same bulletin number. Do not treat this
  TSB as confirmed-applicable to the 2008 Fit until resolved.
- Related fact card: F-WP02-009

### C-004: 2008 Fit total TSB count
- Source A: "19 technical service bulletins issued for the 2008 Honda Fit."
- Source B: "42 TSBs" for the 2008 Honda Fit.
- Likely different counting methodology (e.g., Fit-specific bulletins vs. all bulletins that
  happen to list the Fit among covered models), but unresolved.
- Related fact card: F-WP02-010

### C-005: NHTSA campaign 20V770 (driveshaft corrosion) — model year range covered
- Source A: "2007-2013" Fit.
- Source B: "2007-2014" Fit.
- Both agree the range starts at 2007 (so 2008 is covered either way) and that the recall is
  manual-transmission-only, but the end year conflicts.
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

### C-009: Base-trim auxiliary audio jack, tilt steering wheel, intermittent wipers
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

### C-006: 2008 Fit top NHTSA complaint category counts
- Two different search summaries of what appears to be the same underlying CarComplaints.com
  /NHTSA complaint data gave different counts for the same categories (e.g., Air Bags 55 vs.
  61; Engine 22 vs. 41; Body/Paint not in top 10 vs. 43 complaints; Brakes 13 vs. 24).
  The relative ranking (air bags highest, engine/powertrain and brakes prominent) was
  consistent; exact counts were not.
- Related fact card: F-WP02-011
