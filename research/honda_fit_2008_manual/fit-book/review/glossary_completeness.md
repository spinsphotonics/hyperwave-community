# Glossary Completeness Check (Appendix G vs. Chapters 1-62)

Per RESEARCH_PLAN.md Section 9, step 2: "Generate the glossary from WP-22 and confirm every bolded term in the manuscript has an entry."

## Method

1. Read the existing glossary at `fit-book/draft/appendix_g.md` (165 alphabetized terms, "**Term**: definition." format).
2. Extracted every `**bolded**` span from `ch01.md` through `ch62.md` with a systematic grep sweep (`grep -noP '\*\*[^*\n]+\*\*' ch*.md`), not a manual sample. 39 of the 62 chapters contain no bold markup at all; the remaining 23 chapters produced 364 raw bolded spans.
3. Filtered the 364 raw spans by hand, chapter by chapter, discarding non-glossary bold usage: structural headers ("Safety warnings:", "Steps:", "How you know it worked:", "Step 1." flowchart labels), flowchart answer options ("yes"/"no"/"single click"), table column headers ("Source", "Low/Medium/High"), numbered "common mistakes" list items (full gerund-phrase sentences like "Skipping Step 2."), numbered "weak spot" section headers (e.g. "**1. Motor mounts.**" — the actual jargon inside those paragraphs, like "motor mount" or "torque converter," was not itself bolded), and plain-English words/warning emphasis ("**Always**", "**not**", "title" as a table label) that a 16-year-old would already know.
4. What remained were genuine technical/automotive noun phrases that the book itself explicitly defines in-line at first use. Each was checked case-insensitively, and for wording variants, against the 165 existing glossary terms.
5. For every genuinely new term, wrote a one-sentence, grade-8-level definition drawn only from how that term is explained in its own chapter — no outside facts were added.
6. Merged the new entries into `appendix_g.md` alphabetically with a small Python script (verified via `git diff` that no existing entry's text changed — the only removed line was the old footer note, replaced with an updated one).

## Counts

- **Terms before:** 165
- **Bolded spans found across all 62 chapters (raw):** 364
- **Bolded spans that were genuine, undefined glossary candidates:** 50
- **Terms after:** 215

## New terms added (50)

Aftermarket; Alignment; Alignment rack; Aspect ratio; Bill of sale; Bleeder screw; Brake fluid; Brake shoes; Brake-parts cleaner; Breaker bar; Buyers Guide; C-clamp; Cabin air filter; Caliper; CARB (California Air Resources Board); Clean Air Act; Cranks but won't start; Damper; Defeat device; Diagnostic fee; Differential; Directional tire; Disc brake; Drum brake; Dry boiling point; Endorsement (insurance); Engine air filter; Flat-tip (flathead) screwdriver; Floor jack; Jack stands; Lug wrench; Milliamp (mA); Modding; Odometer disclosure statement; Oil filter wrench; Pre-purchase inspection (PPI); Ratchet; Registration; Rotor; Rub (tire rub); Self-adjuster (adjuster); Spring (suspension); Stage 0; Staggered (fitment); Title (vehicle title); Torque wrench; Unsprung weight; Wet boiling point; Wheel chocks; Won't crank.

Sources, by chapter, for the largest clusters:
- Ch. 22 (tools): Ratchet, Breaker bar, Torque wrench, Oil filter wrench, Lug wrench, Flat-tip (flathead) screwdriver, Floor jack, Jack stands, Wheel chocks, C-clamp, Brake-parts cleaner, Bleeder screw
- Ch. 27 (brakes): Disc brake, Rotor, Caliper, Drum brake, Brake shoes, Self-adjuster (adjuster), Brake fluid
- Ch. 47 (mod law): Modding, Clean Air Act, Defeat device, CARB, Aftermarket, Endorsement
- Ch. 48-50 (mod baseline/tires/suspension): Stage 0, Alignment, Alignment rack, Aspect ratio, Rub (tire rub), Staggered, Unsprung weight, Damper, Spring
- Ch. 51 (brake upgrades): Dry boiling point, Wet boiling point
- Ch. 17/19/20 (paperwork): Buyers Guide, Bill of sale, Odometer disclosure statement, Title, Registration, Pre-purchase inspection (PPI), Milliamp
- Ch. 25/26/30/36/39: Engine air filter, Cabin air filter, Directional tire, Differential, Won't crank, Cranks but won't start, Diagnostic fee

## Notes / judgment calls

- A handful of already-bolded acronyms/terms turned out to already be covered by an existing entry under a slightly different wording and were **not** duplicated: e.g. "TSB", "recall", "campaign number", "NHTSA", "DTC", "MIL", "EPS", "Takata", "Salt-belt states", "H Street (HS)", "Autocross", "Coilover", "Sway bar", "Camber/Caster/Toe", "Torsion beam" (matches "Torsion-beam axle"), "Stainless steel brake line" (matches "Stainless brake lines"), "DOT rating" (matches "DOT brake fluid rating"), "Wheel offset" (matches bolded "Offset"), "Turbocharger", "Street Touring FWD (STF)" (already covered by the broader "Street Touring" entry), and the bolded "Executive Order (EO)" in Ch. 47 (already covered by the existing "CARB EO number" entry, which explains the same EO/CARB relationship).
- Excluded as not genuine jargon a 16-year-old wouldn't know, even though bolded: "flashlight", "Phillips screwdriver", "wire brush", "tire pressure gauge", "jack" (alone), "Drive (D)", "Second (2)", "First (1)" (common words or already explained by the existing "D3" entry's context), and generic list headers like "Labor.", "Parts.", "Tax." in Ch. 39.
- No existing glossary entry's meaning was altered. The only edit to prior content was updating the trailing HTML-comment note (previously stating "165 terms") to record the new total and this review pass.

## Result

`fit-book/draft/appendix_g.md` now contains 215 alphabetized terms and every bolded technical term found across Chapters 1-62 that was genuine jargon has a corresponding glossary entry.
