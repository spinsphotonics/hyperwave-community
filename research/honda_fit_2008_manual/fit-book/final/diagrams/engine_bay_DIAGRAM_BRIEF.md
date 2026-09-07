# Diagram Brief: Engine Bay, Labeled

**Caption (for the finished diagram):** A labeled top-down view of the 2008 Honda Fit
(GD3, L15A1) engine compartment, showing the battery, fuse boxes, and fluid-check points
a new owner needs to find — for Chapter 8 (Controls) and Chapter 16 (Roadside
Emergencies).

**Status: BRIEF ONLY — do not attempt this as an SVG.** Unlike the fuse box maps or
jacking points, an engine bay diagram has to get the *relative position* of a dozen-plus
real parts right (what's on the left vs. right, what's near the firewall vs. the
radiator, what's visible immediately vs. under a cover). Guessing at that layout from
partial facts would produce a diagram that looks authoritative but could send a reader's
hand to the wrong place under the hood — exactly the kind of confident-wrong diagram
Rule D9 and this project's own instructions rule out. This brief exists so a human
illustrator (or a future research pass with a real under-hood photo or the Honda service
manual's own diagram, redrawn rather than traced) can do this correctly later.

---

## What we can confirm today (anchor points for the illustrator)

Use these as fixed, sourced landmarks. Everything else on the diagram must come from a
verified source before it is drawn — do not fill gaps by guessing what "looks right" for
a small hatchback engine bay.

| Confirmed item | Location detail | Fact ID |
|---|---|---|
| Battery | In the engine compartment (exact side not stated in the cited fact) | F-WP07-011 |
| Secondary fuse box | Mounted directly on the battery's positive terminal; holds one 80 A fuse | F-WP21-004, F-WP07-011 |
| Primary under-hood fuse box | Engine compartment, **driver's side**; opens by pushing tabs on the lid | F-WP21-003 |
| Jump-start "stay" (ground point) | A metal ground point in the engine bay, shown in the manual's own illustration but not named as a specific bracket in the text we could read — do **not** invent a part name for it | F-WP07-012 |
| Radiator + radiator cap | Somewhere in the engine bay, accessed for the overheating procedure; exact position not confirmed this pass | F-WP07-016, F-WP07-017 |
| Coolant reserve tank | Separate from the radiator itself; checked/topped for the overheating procedure | F-WP07-016 |
| Electric Power Steering (EPS) system | Present on all 2008 US Fit trims (not Sport-exclusive); no hydraulic power-steering pump or reservoir exists on this car | F-WP04-018, Section F of facts/WP-12.md |
| Engine | 1.5 L L15A1 SOHC VTEC 4-cylinder | RESEARCH_PLAN.md D3 |

## What is still missing before this can be drawn accurately

None of the following component positions were confirmed by a Tier 1/2 source in the
work packages read for this task. **Do not guess at any of these** — get them from the
owner's manual's own under-hood diagram (paraphrased layout, not traced), the Haynes
manual, or a verified photo before drawing:

- Engine orientation (transverse, which side the transmission sits on)
- Air intake box / air filter housing position (needed for Chapter 25's air filter
  procedure)
- Engine oil fill cap and dipstick locations
- Brake fluid reservoir location (needed since it's shared with the clutch on manual
  cars — see the Leaks table in facts/WP-12.md, Section F)
- Windshield washer fluid reservoir location
- Alternator location (relevant to the Charging System Indicator, F-WP04-012, and the
  "whining from the front of the engine" noise entry in facts/WP-12.md)
- Drive belt / serpentine belt routing and tensioner location
- Exact hood prop-rod or hood-strut location and hood release mechanism (belongs to
  WP-03; not in the fact files read for this task — check `facts/WP-03.md` if it exists)
- Underhood emissions/VIN sticker location (relevant to Chapter 2 and Appendix I)
- Precise left/right side of the battery and radiator relative to the driver's seat

## Labeling requirements for the finished diagram

1. Every label must trace to a Fact ID the way this brief's table does. If a future
   researcher adds a new fact for one of the missing items above, cite it.
2. Use plain-English labels a 16-year-old can read at a glance ("Battery," not "12V
   SLI lead-acid unit").
3. Mark anything not confirmed as "approximate" or omit it rather than drawing it with
   false confidence.
4. Show the primary under-hood fuse box and the secondary (battery-mounted) fuse box
   as two visually distinct boxes — readers regularly confuse these (see
   facts/WP-21.md).
5. Orient the diagram with the front of the car clearly marked, and state the viewing
   angle (top-down, standing at the open hood) in the caption.

## Style guidance

- Simple, schematic, labeled shapes — the same honest-simplification style used in this
  book's other fresh diagrams (fuse box maps, jacking points). Not a photorealistic
  rendering.
- Once the missing items above are sourced, this brief can be converted into an SVG
  using the same method as `fuse_box_maps.svg` and `jacking_points.svg` in this folder.

## What NOT to do

- Do not invent a plausible-looking engine bay layout from general automotive knowledge
  of "what small cars usually look like under the hood." The Fit's actual layout may
  differ, and a wrong diagram is worse than no diagram.
- Do not trace or closely copy Honda's own under-hood illustration (Rule D7) — redraw
  from verified facts once they exist.
