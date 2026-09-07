# Research Plan: "The Complete 2008 Honda Fit Book"

A foolproof, step-by-step plan for researching, verifying, writing, and publishing an
ultra-comprehensive owner's book for the **2008 Honda Fit, base trim**. The book must be
readable by a 16-year-old new driver and a parent who knows nothing about cars.

This plan is written so that **small models with limited reasoning** can execute it.
Every task is atomic, has a fixed output template, and has a binary "done when" check.
No task requires judgment beyond "does the source say this, yes or no."

---

## 0. Locked decisions (read before doing anything)

These decisions are fixed. Do not re-decide them. If a task seems to need a different
decision, stop and write the question in `queue/decisions_needed.md`.

| ID | Decision | Value |
|----|----------|-------|
| D1 | Vehicle | 2008 Honda Fit, **US market**, base trim (not Sport) |
| D2 | Generation | **First generation (chassis code GD3)**, sold in the US for model years 2007–2008 only |
| D3 | Engine | 1.5 L L15A1 SOHC VTEC, 4-cylinder (verify exact horsepower/torque in WP-01) |
| D4 | Transmissions covered | Both: 5-speed manual and 5-speed automatic |
| D5 | Units | US customary first, metric in parentheses. Example: `29 psi (200 kPa)` |
| D6 | Reading level | Flesch-Kincaid grade 8 or lower. Sentences under 20 words on average |
| D7 | Copyright | Never copy more than 10 consecutive words from any source. Paraphrase everything. No images from any source; all diagrams are drawn fresh |
| D8 | Year confusion rule | The 2009–2014 Fit (chassis GE8) is a **different car**. Any fact from a GE8 source is **invalid** unless a second source confirms it applies to the 2007–2008 GD3 |
| D9 | Safety stance | The book never tells a reader to skip a safety step. When a procedure can hurt someone, the book says so before the steps, not after |
| D10 | Output language | English, US spelling |

---

## 1. How a small model must behave on every task

Print these rules at the top of every task prompt. They are not optional.

1. **Do one task at a time.** Finish it, save the output file, then stop.
2. **Never guess.** If a source does not say it, write `UNVERIFIED` in the value field. An
   `UNVERIFIED` fact is fine. An invented fact is a failure.
3. **Never generalize across model years.** Every source must be checked for the phrase
   "2008" or "2007-2008" or "first generation" or "GD3". If the source does not name the
   year, mark the fact `YEAR-UNCONFIRMED`.
4. **Copy the fact card template exactly.** Do not add, remove, or rename fields.
5. **Every fact gets a source URL and access date.** No URL, no fact.
6. **When two sources disagree, record both.** Write the conflict in
   `queue/conflicts.md`. Do not pick a winner.
7. **Do not summarize forum threads.** Extract one fact per card, with the post URL.
   Forum facts are Tier 4 (see Section 3) and must be marked as such.
8. **Do not write prose during research phases.** Research outputs are tables and cards only.
   Prose is written in Phase 5 by a different task.
9. **Stop conditions.** Stop and write to `queue/blocked.md` if: a source is paywalled and
   no free equivalent exists, a task template is missing, or you have spent the task's time
   budget.
10. **No part numbers from memory.** Part numbers come only from Tier 1 or Tier 2 sources,
    and each must be copied character-by-character and then re-read against the source.

---

## 2. Folder layout for the project

```
fit-book/
  plan/                 this document, and the master ledger
  sources/              one markdown file per source (see Source Card template)
  facts/                one file per work package, containing fact cards
  procedures/           one file per procedure card
  queue/                decisions_needed.md, conflicts.md, blocked.md
  verify/               verification reports, one per work package
  draft/                book chapters, one file per chapter
  review/               readability, safety, and legal review reports
  final/                assembled manuscript
```

The **master ledger** `plan/ledger.csv` has one row per work package:

```
wp_id,title,phase,owner,status,facts_count,unverified_count,verified_by,notes
```

Status is exactly one of: `todo`, `in_progress`, `done`, `blocked`, `verified`, `written`, `reviewed`.

---

## 3. Source tiers and the approved source list

### Tiers

| Tier | What it is | Trust rule |
|------|-----------|------------|
| 1 | Honda's own documents: 2008 Fit Owner's Manual, Honda service manual (Helm), Honda Technical Service Bulletins, Honda parts catalog, official Honda press kit/specs | One Tier 1 source is enough for a fact |
| 2 | Government and standards bodies: NHTSA (recalls, complaints, crash tests), IIHS, EPA fuel economy, state DMV pages, SCCA/NASA rulebooks | One Tier 2 source is enough |
| 3 | Published repair manuals and reputable references: Haynes, Chilton, ALLDATA/Mitchell excerpts, tire and parts manufacturer spec sheets, Edmunds/KBB/Consumer Reports for 2008 | Need one Tier 3 plus one other source of any tier |
| 4 | Community: FitFreak forum, Reddit r/hondafit, Honda-Tech, YouTube walkthroughs, blogs | Never sufficient alone. Need two independent Tier 4 sources **and** the fact must be labeled "community-reported" in the book |

### Approved starting sources (task WP-00 expands this list)

- Honda Owners site: `owners.honda.com` – download the 2008 Fit Owner's Manual PDF
- Helm Inc. – official Honda service manual for 2007-2008 Fit (paid; log in `queue/blocked.md` if unavailable and use Haynes as the fallback)
- Haynes manual 42030 (Honda Fit 2007–2013) – note it covers both generations; apply rule D8 on every page
- NHTSA: `nhtsa.gov/recalls` and `nhtsa.gov/vehicle/2008/HONDA/FIT` for recalls, complaints, investigations, TSB summaries
- IIHS: `iihs.org` crash test ratings for the 2008 Fit
- EPA: `fueleconomy.gov` for 2008 Fit ratings
- Honda parts diagrams: any dealer-run online parts catalog (e.g., hondapartsnow.com, bernardiparts.com) – Tier 1 for part numbers because they mirror Honda's catalog
- Tire and wheel fitment: tire manufacturer sites, plus `tiresize.com` style calculators (Tier 3)
- FitFreak (`fitfreak.net`) – largest first-gen Fit community, Tier 4
- SCCA Solo rulebook and NASA HPDE rules – Tier 2 for track prep

### Source Card template (`sources/S-###.md`)

```
Source ID: S-###
Title:
Author/Publisher:
URL:
Type: (owner's manual | service manual | TSB | recall | forum post | video | article | rulebook)
Tier: (1|2|3|4)
Covers model year 2008 GD3? (yes | no | unclear)
Date published:
Date accessed:
Paywalled? (yes | no)
Notes: one line, what this source is good for
```

---

## 4. Target book outline

Research work packages (Section 5) feed these chapters one-to-one. Do not change this
outline during research. Proposed changes go in `queue/decisions_needed.md`.

**Part I – Meet Your Fit**
1. What this car is (history, why the Fit exists, first generation vs second generation)
2. Identify your exact car (VIN decoding, trim differences, build plate, paint code, what "base" got and did not get)
3. Specifications at a glance

**Part II – The Absolute Basics**
4. Keys, locks, and getting in (mechanical key, remote, if the base trim has one, child locks, trunk, hood release, fuel door)
5. Sitting right (seat adjustment, steering wheel, mirrors, headrests, seat belts)
6. Starting and stopping the car (both transmissions; what every sound and light means during startup)
7. The dashboard explained (every gauge, every warning light, what to do for each one)
8. Controls you will use every day (lights, wipers, horn, turn signals, hazards, defrost, HVAC, windows, the radio)
9. The Magic Seat and cargo tricks
10. Fuel (which grade, how to fill, what the low-fuel light means, range)

**Part III – Driving It**
11. First drive for a new driver (manual and automatic, separate walkthroughs)
12. Driving a manual transmission from zero (clutch, stalling, hills, downshifting)
13. Parking, reversing, tight spaces
14. Highway, rain, snow, heat, and night
15. Getting good fuel economy
16. What to do when something goes wrong on the road (flat tire, dead battery, overheating, warning light appears, accident checklist)

**Part IV – Owning It (the paperwork)**
17. Title, registration, insurance, inspection and emissions basics (general, with a "check your state" pointer)
18. Recalls and service bulletins for this exact car, and how to check for new ones
19. Documents to keep in the glovebox and at home
20. Buying a used 2008 Fit: inspection checklist and known trouble spots

**Part V – Maintenance**
21. The maintenance schedule (what, when, how much it costs to DIY vs shop)
22. Tools and supplies for a starter garage
23. Safety first: jacking, jack stands, wheel chocks, hot parts, fluids
24. Oil and filter change
25. Air filter and cabin filter
26. Tires: pressure, rotation, wear, replacement, the spare
27. Brakes: inspection, pads, rotors, fluid
28. Battery: testing, cleaning, replacing, jump starting
29. Spark plugs
30. Coolant, transmission fluid, brake fluid, power steering (if hydraulic; verify), washer fluid
31. Belts, hoses, timing chain (note: not a belt; verify)
32. Lights and bulbs, wipers, fuses
33. Keeping it clean inside and out

**Part VI – Diagnosing Problems**
34. How to describe a problem (noise, when, where, how fast)
35. The check engine light and OBD-II codes: reading, common codes for this engine, clearing
36. Won't start / won't crank / cranks but won't start flowchart
37. Noises, vibrations, smells, leaks: lookup tables
38. Known weak spots on the 2007–2008 Fit (from recalls, TSBs, and community consensus)
39. When to go to a shop, and how to not get ripped off

**Part VII – Repairs You Can Do at Home**
40. Brake job, full
41. Suspension: struts, shocks, sway bar links, bushings
42. Cooling system: thermostat, radiator, water pump
43. Starter, alternator, ignition switch
44. Exhaust and O2 sensors
45. Interior fixes: door handles, window regulators, seat repair
46. Body: dents, rust, paint touch-up, headlight restoration

**Part VIII – Souping It Up**
47. Modding philosophy: goals, budget, legality, insurance, warranties, what not to do
48. Stage 0: baseline (fresh fluids, good tires, brakes, alignment)
49. Tires and wheels: sizes that fit, offset, weight
50. Suspension: springs, coilovers, sway bars, alignment specs for street and track
51. Brakes: pads, fluid, lines, rotors, cooling
52. Intake, header, exhaust: what they actually do on an L15A1
53. Engine management: what tuning options exist for the GD3 (verify carefully; do not assume GE8 tools work)
54. Power adders: turbo/supercharger kits, and the honest list of what breaks
55. Engine swaps that have been done (L15A to what, cost, difficulty)
56. Weight reduction and aerodynamics
57. Seats, harnesses, roll bars: safety gear and when it becomes mandatory

**Part IX – Getting Ready for the Track**
58. Autocross: what it is, what class a base Fit runs in, first-event checklist
59. HPDE / track days: rules, tech inspection, what the car needs
60. Track-day prep checklist (the day before, the morning of, between sessions)
61. Driver skills: lines, braking, looking ahead, listening to the car
62. After the track: inspection and recovery

**Part X – Appendices**
- A. Full specifications
- B. Fluid capacities and types
- C. Torque specifications
- D. Fuse and relay maps
- E. Tire and wheel fitment table
- F. Part numbers for common maintenance items
- G. Glossary (every term used in the book, defined in one sentence)
- H. Resources (communities, parts sellers, rulebooks)
- I. Recall and TSB list

---

## 5. Research work packages

Every work package (WP) uses the same task prompt structure:

```
TASK: <WP id and title>
RULES: <paste Section 1>
VEHICLE: 2008 Honda Fit, US market, base trim, GD3, L15A1
INPUTS: <list of source IDs to use, from sources/>
QUESTIONS: <numbered list, below>
OUTPUT: facts/<WP id>.md containing one Fact Card per answer, and
        procedures/<WP id>-<n>.md for each procedure
TIME BUDGET: <minutes>
DONE WHEN: every question has a fact card (value may be UNVERIFIED),
           every card has a source ID and URL, ledger row updated
```

### Fact Card template (`facts/WP-##.md`, one card per fact)

```
Fact ID: F-WP##-###
Chapter: <chapter number from Section 4>
Question: <the exact question from the WP>
Value: <the answer, with units> | UNVERIFIED
Applies to: 2008 GD3 base trim (yes | no | unclear)
Source ID: S-###
Source URL:
Exact location in source: (page number, section, timestamp, or post number)
Tier: (1|2|3|4)
Confidence: (high = Tier 1 or 2 | medium = Tier 3 confirmed | low = Tier 4 or single Tier 3)
Conflicts with: (Fact ID or "none")
Date accessed:
```

### Procedure Card template (`procedures/WP-##-n.md`)

```
Procedure ID: P-WP##-n
Chapter:
Title: (verb first: "Change the engine oil")
Skill level: (1 = anyone | 2 = needs basic tools | 3 = needs jack stands and care | 4 = shop recommended)
Time: <minutes for a first-timer>
Tools: (one per line)
Parts and supplies: (one per line, with part number and Fact ID that sourced it)
Safety warnings: (one per line; each starts with "Do not" or "Always")
Steps: (numbered; one action per step; no step longer than 25 words)
How you know it worked:
Common mistakes:
Source IDs:
```

---

### Phase 1 – Foundations (do these first, in order)

**WP-00 Build the source library**
- Questions: Which of the approved sources are reachable? What is the URL for the 2008 Fit Owner's Manual PDF? Is a Honda service manual obtainable? What is the Haynes manual ISBN? What are the top five community threads titled like "GD3 FAQ", "first-gen Fit sticky", or "new owner guide"?
- Output: at least 15 Source Cards in `sources/`.
- Time: 90 min.
- Done when: every approved source in Section 3 has a Source Card marked reachable or blocked.

**WP-01 Identify the car exactly** (Chapters 1–3)
- Questions: Exact engine code, displacement, horsepower and torque with RPM, compression ratio, bore/stroke, fuel type, redline. Transmission options and gear ratios. Curb weight per transmission. Dimensions (length, width, height, wheelbase, ground clearance). Cargo volume seats up/down. Fuel tank size. EPA MPG city/highway per transmission. Trim list for 2008 US Fit and what base lacks vs Sport (wheels, cruise control, fog lights, spoiler, paddle shifters, audio, keyless entry, power windows? verify each). Factory tire size and wheel size. Paint codes and color names. VIN decoding: which characters mean what for this car. Where the VIN plate, door jamb sticker, and emissions label are. Production plant and country. IIHS and NHTSA crash ratings. Original MSRP.
- Time: 120 min.

**WP-02 Recalls, TSBs, and known defects** (Chapters 18, 38, Appendix I)
- Questions: Every NHTSA recall for the 2008 Fit with campaign number, date, component, description, and remedy. Every TSB summary available on NHTSA for 2008 Fit. Top ten NHTSA complaint categories by count. From community sources: the top ten "everyone with a first-gen Fit eventually deals with this" problems, each with two independent Tier 4 sources.
- Time: 120 min.
- Note: starting points to confirm (do not assume): power window master switch water intrusion recall; Takata airbag inflator recalls; headlight low-beam wiring harness issues; ignition switch complaints.

### Phase 2 – Basics and driving (Chapters 4–16)

**WP-03 Getting in and getting comfortable** (Chapters 4–5)
- Questions: How the key works, whether base trim has remote keyless entry, how to lock/unlock all doors, child safety lock location, hood release and prop rod location, fuel door release, trunk/hatch release, seat adjustment controls and ranges, steering wheel tilt (and telescoping? verify), mirror adjustment (manual or power on base? verify), headrest adjustment, seat belt adjustment, seat belt height adjuster presence.
- Source: Owner's Manual primarily.
- Output: facts plus procedure cards for: "Open the hood", "Adjust your seat and mirrors before driving".

**WP-04 Starting, stopping, dashboard** (Chapters 6–7)
- Questions: Starting procedure for manual and automatic (clutch pedal requirement, brake requirement, shifter position). Every indicator light on the instrument cluster: name, icon description in words, meaning, what to do, and whether it is safe to keep driving. Gauge list and normal readings. What the immobilizer light means. Shift interlock release location on automatics.
- Output: a full table of warning lights (this becomes the most-used page in the book).

**WP-05 Everyday controls** (Chapters 8–10)
- Questions: Location and operation of headlights, high beams, turn signals, hazards, wipers and washer, rear wiper, defrost front and rear, HVAC controls, air conditioning, recirculation, window controls, door locks, dome lights, 12 V outlet location, audio system basics (does base trim have an aux jack? verify), clock setting. Magic Seat: every configuration (utility, long, tall, refresh modes) with steps. Fuel: recommended octane, tank capacity, low-fuel light trigger amount, fuel door and cap procedure, "check fuel cap" light behavior.
- Output: procedure cards for each Magic Seat mode and for refueling.

**WP-06 Driving** (Chapters 11–15)
- Questions: Manual transmission shift pattern and reverse lockout method. Recommended shift points for economy. Automatic gear positions and what D3, 2, 1 do (verify which exist on the base auto). Hill start method. Break-in recommendations from Honda. Fuel economy tips specific to this car (from owner's manual and EPA). Winter tire guidance and snow chain permissibility from the owner's manual. Towing capacity (likely "not recommended"; verify).
- Output: procedure cards for "Start on a hill (manual)", "Drive a manual for the first time".

**WP-07 Roadside emergencies** (Chapter 16)
- Questions: Spare tire type and location, jack and tool location, jacking points from the owner's manual, spare tire pressure and speed limit, jump-start procedure and terminal locations, overheating procedure, what to do for each red warning light, hazard light and reflector guidance, towing hook locations and flat-tow rules for automatic and manual.
- Output: procedure cards for "Change a flat tire", "Jump start the car", "Car is overheating".

### Phase 3 – Ownership and maintenance (Chapters 17–33)

**WP-08 Paperwork and buying used** (Chapters 17, 19, 20)
- Questions: What documents come with a car in the US (general). How to check a recall by VIN. What a pre-purchase inspection should cover for this car (built from WP-02 results). Typical 2008 Fit used prices in the current year from two Tier 3 sources. Common odometer ranges seen and what wears by then.
- Keep state-specific rules generic: "check your state DMV site."

**WP-09 The maintenance schedule** (Chapter 21)
- Questions: Does the 2008 Fit use Honda's Maintenance Minder or a fixed mileage schedule (verify from Owner's Manual). Every service item with its interval. Severe vs normal service definitions. Recommended oil viscosity and spec. Oil capacity with and without filter. Every fluid type and capacity (engine oil, coolant, manual transmission fluid, automatic transmission fluid, brake fluid, power steering if applicable, washer). Spark plug type, gap, interval. Air filter and cabin filter part numbers and intervals. Timing chain or belt, and any interval. Valve adjustment interval (SOHC L15A has adjustable valves; verify). Battery group size. Wiper blade sizes. Bulb sizes for every exterior and interior light.
- Output: one master maintenance table (this becomes Appendix B and part of Appendix F).

**WP-10 Torque specs and fitment numbers** (Appendices C, E)
- Questions: Wheel lug nut torque, oil drain plug torque, oil filter torque, spark plug torque, caliper bracket bolts, caliper slide pins, strut top nuts, battery hold-down. Lug pattern, center bore, factory wheel offset, tire size, recommended pressures front and rear (door jamb), largest tire that fits without rubbing (Tier 4 only; label as community-reported).
- Source: service manual first, Haynes second, community last.

**WP-11 Maintenance procedures** (Chapters 22–33)
- One procedure card each, following the Procedure Card template: oil and filter change; engine air filter; cabin air filter; check and set tire pressure; rotate tires; inspect brake pads through the wheel; replace brake pads (front); replace brake pads (rear; drum or disc on base? verify); bleed brakes; test battery with a multimeter; replace battery; clean terminals; replace spark plugs; check coolant level and top up; drain and fill coolant; check and change manual transmission fluid; check and change automatic transmission fluid; replace headlight bulb; replace tail/brake bulbs; replace wiper blades; replace a fuse; jack up the car and set it on jack stands; wash and wax; clean interior.
- Each card is a separate task so a small model handles one procedure per run.
- Time: 30 min per card.

### Phase 4 – Troubleshooting and repairs (Chapters 34–46)

**WP-12 Diagnostics** (Chapters 34–37)
- Questions: OBD-II port location. The twenty most-reported codes for the L15A1 (Tier 3 and 4; each code needs two sources) with plain-English meaning and first thing to check. Symptom lookup tables: noises (by when they happen), vibrations (by speed), smells, leak colors and locations, each with "likely cause" and "urgency" columns. A "won't start" decision tree with no more than three branches at each node.
- Output: tables only; the decision tree is written as a numbered list with "if yes go to step N".

**WP-13 Known weak spots and shop guidance** (Chapters 38–39)
- Questions: For each defect from WP-02, what the symptoms are, cost to fix at a shop (two Tier 3 or 4 sources), and whether it is DIY-able. Typical labor rates and how to read an estimate (general, Tier 3).

**WP-14 Home repairs** (Chapters 40–46)
- One procedure card per repair listed in Chapters 40–46. Skill level must be filled in honestly; level 4 items get a "what the shop will do" summary instead of full steps.
- Time: 45 min per card.

### Phase 5 – Modifications and track (Chapters 47–62)

**WP-15 Modding ground rules** (Chapters 47–48)
- Questions: Federal emissions tampering rules (EPA, Tier 2). CARB exemption numbers (EO numbers) concept and how to look one up. Typical insurance implications (Tier 3). Baseline checklist items with references to WP-11 procedures.

**WP-16 Wheels, tires, suspension, brakes** (Chapters 49–51)
- Questions: Wheel sizes and offsets the community runs without rubbing (two Tier 4 sources each). Popular tire sizes for autocross on the GD3. Lowering springs and coilovers made for the GD3 (brand, model, drop, spring rate if published). Rear sway bar options. Front camber adjustment options. Alignment specs for street and track (community-reported). Brake pad compounds for track, brake fluid specs and boiling points (Tier 3 manufacturer data), stainless lines.
- Rule: every product must be confirmed as fitting the **GD3**, not GE8, by product listing text.

**WP-17 Engine: bolt-ons and tuning** (Chapters 52–53)
- Questions: Documented dyno gains for intake, header, exhaust on the L15A1 (Tier 4, two sources, and mark as "reported"). What engine management options exist for the 2007–2008 US Fit ECU (verify: is Hondata FlashPro GE8-only? are there piggyback or standalone options for GD3?). Whether the GD3 ECU can be reflashed at all.
- This WP is the highest risk for year confusion. Apply D8 on every fact.

**WP-18 Power adders and swaps** (Chapters 54–55)
- Questions: Turbo and supercharger kits ever sold for the GD3 L15A1, whether still available, reported power, reported failures (two sources). Documented engine swaps into a GD3 (engine, who did it, thread URL, cost estimate, key obstacles). Fuel system, clutch, and axle upgrades reported necessary.

**WP-19 Safety gear and weight** (Chapters 56–57)
- Questions: Seat and harness rules from SCCA Solo and a representative HPDE organization (Tier 2). Roll bar requirements by event type. Weight reduction items commonly removed and how much they weigh (Tier 4). Aerodynamic parts commonly used.

**WP-20 Autocross and track days** (Chapters 58–62)
- Questions: Current SCCA Solo class for a stock 2008 Fit and what modifications move it to which class (Tier 2, rulebook year noted). NASA/HPDE typical tech inspection checklist items. Event-day checklist items. Beginner driving skill sources (Tier 3 books or org guides; paraphrase only). Post-event inspection items.

### Phase 6 – Appendices

**WP-21 Fuse and relay maps** (Appendix D)
- Questions: Location of each fuse box, every fuse position with amperage and circuit, from the owner's manual. Output as a table.

**WP-22 Glossary and resources** (Appendices G, H)
- Questions: Collect every technical term used in all fact and procedure cards. Define each in one sentence at grade-8 reading level. List all Tier 4 communities and Tier 1/3 parts sources with URLs.

---

## 6. Verification pass (separate role, separate model run)

Every WP with status `done` gets verified before writing. The verifier never sees the
researcher's reasoning, only the fact cards and sources.

**Verifier task prompt:**
```
TASK: Verify facts/WP-##.md
For each fact card:
  1. Open the Source URL. Confirm the source exists.
  2. Find the exact location given. Confirm the value matches. Write PASS or FAIL.
  3. Confirm the source names 2008, 2007-2008, first generation, or GD3. Write YEAR-OK or YEAR-FAIL.
  4. If Tier 3 or 4, confirm the second source exists. Write CORROBORATED or SINGLE.
Output: verify/WP-##.md with one line per fact: Fact ID, PASS/FAIL, YEAR-OK/FAIL, CORROBORATED/SINGLE
Any FAIL or YEAR-FAIL sets the fact to UNVERIFIED. Do not fix it yourself.
```

A WP is `verified` when: zero FAIL, zero YEAR-FAIL, and all Tier 3/4 facts are CORROBORATED
or explicitly marked "community-reported."

Facts that stay UNVERIFIED after two research passes are written in the book as
"We could not confirm this. Check your owner's manual." That sentence is acceptable.
A confident wrong number is not.

---

## 7. Writing pass (Chapters, one task per chapter)

**Writer task prompt:**
```
TASK: Write draft/ch##.md
INPUTS: all verified fact cards and procedure cards tagged with this chapter
RULES:
  - Use only facts from the cards. No new facts.
  - Every number in the text must be followed by its Fact ID in a comment: <!-- F-WP01-003 -->
  - Grade 8 reading level. Average sentence under 20 words. No sentence over 30 words.
  - Define every technical term the first time it appears, in the same sentence.
  - Start the chapter with "What you will learn" (3-5 bullets) and end with "Quick check" (3-5 questions).
  - Every procedure follows the procedure card exactly. Safety warnings go before step 1.
  - Use "you." Never say "simply," "just," or "obviously."
  - Community-reported facts get the phrase "Owners report that" in front of them.
  - UNVERIFIED facts get: "We could not confirm this for the 2008 Fit. Check your owner's manual."
OUTPUT: the chapter file, plus a list of Fact IDs used.
DONE WHEN: every fact card tagged with this chapter is either used or listed as "not used, because ..."
```

**Chapter structure template:**
```
# Chapter ##: Title
## What you will learn
## The short version (3 sentences a 16-year-old can repeat back)
## The full explanation
## Step by step (if any procedures)
## What can go wrong
## Quick check
```

---

## 8. Review passes (three separate tasks per chapter)

**R1 Readability review**
- Compute Flesch-Kincaid grade for the chapter. Must be 8.0 or lower.
- List every sentence over 30 words. Writer must fix them.
- List every technical term not defined on first use.
- Output: `review/ch##-readability.md`, PASS or FAIL with the list.

**R2 Safety review**
- For every procedure: confirm the safety warnings are before step 1, confirm jack stands are mentioned any time the car is lifted, confirm "engine off and cool" is stated for any under-hood fluid task, confirm eye protection is listed for brake and battery work.
- Output: `review/ch##-safety.md`, PASS or FAIL with line numbers.

**R3 Fact and legal review**
- Confirm every number has a Fact ID comment and the Fact ID exists and is verified.
- Search the chapter for any 10-word run that appears verbatim in a source (spot-check 20 random sentences against their source).
- Confirm no images are copied.
- Output: `review/ch##-facts.md`, PASS or FAIL.

A chapter is `reviewed` only when all three say PASS.

---

## 9. Assembly and publication

1. Concatenate chapters in outline order into `final/manuscript.md`.
2. Generate the glossary from WP-22 and confirm every bolded term in the manuscript has an entry.
3. Build the index: every Fact ID comment becomes an endnote citing the Source Card title and URL.
4. Add the front matter: title, disclaimer ("This book is not affiliated with Honda. Working on cars can injure you. Follow your owner's manual when it disagrees with this book."), and a one-page "How to use this book."
5. Commission fresh diagrams for: dashboard warning lights, engine bay labeled, fuse box maps, jacking points, Magic Seat modes, brake assembly, OBD-II port location. Each diagram has a caption and is drawn from the described facts, not traced from a source.
6. Export to PDF and EPUB. Print one proof copy and have a real 16-year-old and a real parent each complete three procedures from it. Log every point where they got stuck in `review/user-test.md` and fix those pages.
7. Publish.

---

## 10. Effort estimate

| Phase | Tasks | Minutes per task | Total hours |
|-------|-------|------------------|-------------|
| 1 Foundations | 3 | 90–120 | 5.5 |
| 2 Basics and driving | 5 | 60–90 | 6.5 |
| 3 Ownership and maintenance | 4 WPs + 24 procedure cards | 30–90 | 18 |
| 4 Troubleshooting and repairs | 3 WPs + ~15 procedure cards | 45–90 | 15 |
| 5 Mods and track | 6 | 90 | 9 |
| 6 Appendices | 2 | 60 | 2 |
| Verification | 23 | 30 | 11.5 |
| Writing | 62 chapters | 45 | 46.5 |
| Review | 62 × 3 | 15 | 46.5 |
| Assembly | 1 | – | 8 |
| **Total** | | | **~170 hours of model time** |

---

## 11. Why this plan is foolproof for small models

- **No task requires synthesis.** Research tasks copy values into cards. Verification tasks compare two strings. Writing tasks rephrase cards. Review tasks count words and check lists.
- **Every output has a template.** A model that cannot reason can still fill in a form.
- **Errors are caught by a different model run.** Research, verification, writing, and review are separate prompts with separate inputs, so a single hallucination has to survive three independent checks to reach the book.
- **Year confusion has a dedicated rule (D8) and a dedicated check (YEAR-OK).** This is the single most likely mistake for this car, because the 2009 Fit shares a name and looks similar.
- **UNVERIFIED is a valid answer.** The plan removes the incentive to guess.
- **Blocked is a valid state.** Paywalls and missing sources go to a queue instead of being worked around with invention.
- **Done-when checks are binary.** A task is done or it is not; there is no "mostly done."

---

## 12. Quick-start: the first five tasks to run

1. Run WP-00 (source library). Do not start anything else until 15 Source Cards exist.
2. Run WP-01 (identify the car). This produces the facts every other WP depends on.
3. Run WP-02 (recalls and defects). This shapes the buying, troubleshooting, and safety chapters.
4. Run the verifier on WP-01 and WP-02.
5. Run WP-09 (maintenance schedule). It is the spine of the book's second half.

After these five, all remaining WPs can run in any order and in parallel.
