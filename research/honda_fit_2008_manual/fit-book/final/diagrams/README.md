# Fresh Diagrams — 2008 Honda Fit Book

Produced per `RESEARCH_PLAN.md` Section 9, step 5: "Commission fresh diagrams for:
dashboard warning lights, engine bay labeled, fuse box maps, jacking points, Magic Seat
modes, brake assembly, OBD-II port location. Each diagram has a caption and is drawn
from the described facts, not traced from a source."

No image-generation tool and no real photos of this car were available for this task.
Every item below is either (a) a real SVG, built only from geometric/layout facts that
were already fully known and textual, or (b) a written brief for a future illustrator,
used whenever accurate real-world part shapes or positions would be needed that no
available source confirmed. Nothing here was traced from Honda's own diagrams or any
other copyrighted image (Rule D7) — every SVG shape is an original, honestly simplified
schematic (a rounded rectangle, a grid of boxes, a labeled circle), not a redrawing of
someone else's illustration.

## The 7 items

| # | Item | Format | Caption | Source Fact IDs |
|---|---|---|---|---|
| 1 | Fuse box maps | **SVG** — `fuse_box_maps.svg` | The three fuse boxes of a 2008 Honda Fit base trim, with every numbered position's amperage and protected circuit, for Chapter 32 (Fuses) and Appendix D. | F-WP21-001 through F-WP21-014, plus the Interior/Under-hood/Secondary fuse tables in `facts/WP-21.md` |
| 2 | Jacking points | **SVG** — `jacking_points.svg` | Approximate locations of the 2008 Fit's four jacking points (one per wheel), for Chapter 16 (Roadside Emergencies). | F-WP07-008 |
| 3 | Magic Seat modes | **SVG** — `magic_seat_modes.svg` | The Fit's four Magic Seat cargo/rest configurations, simplified to side-view rectangles, for Chapter 9 (The Magic Seat and Cargo Tricks). | `procedures/WP-05-1.md` through `WP-05-4.md` |
| 4 | OBD-II port location | **SVG** — `obd_port_location.svg` | The OBD-II diagnostic port's approximate location in the driver's footwell, for Chapter 35 (Diagnostics). | F-WP12-001 |
| 5 | Dashboard warning lights | **Markdown reference sheet** — `dashboard_warning_lights.md` | A plain-language reference sheet for all 26 instrument-panel indicators, showing which 20 appear on a US base-trim dash, for Chapter 7 (The Dashboard Explained). | F-WP04-009 through F-WP04-032; F-WP07-018 through F-WP07-022 |
| 6 | Engine bay, labeled | **Diagram brief** — `engine_bay_DIAGRAM_BRIEF.md` | *(no finished diagram yet — see brief)* | F-WP07-011, F-WP07-012, F-WP07-016/017, F-WP21-003/004, F-WP04-018 |
| 7 | Brake assembly (front disc + rear drum) | **Diagram brief** — `brake_assembly_DIAGRAM_BRIEF.md` | *(no finished diagram yet — see brief)* | `procedures/WP-11-7.md`, `procedures/WP-11-8.md`, F-WP07-009 |

## Why SVGs for 1–4 but briefs for 6–7 (and a table, not an SVG, for 5)

The task's own instructions drew this line, and the facts confirmed it holds for every
item:

- **Fuse box maps, jacking points, Magic Seat modes, and the OBD-II port** are all
  cases where the *entire* layout needed for an honest diagram was already sitting in
  the fact cards as numbers, positions, or simple described locations — a numbered grid
  of fuse positions, four corner-points on a generic car outline, a handful of
  seat-position rectangles, one dashboard-area callout. Drawing these as schematic SVGs
  adds nothing invented; it just gives the already-known facts a visual form.
- **Dashboard warning lights** could technically be forced into an SVG grid of icon
  glyphs, but that would mean drawing 26 Honda icon shapes from scratch — the exact
  glyph geometry Honda used, which no source here provided precisely enough to redraw
  honestly without effectively tracing it. A Markdown reference-sheet table, using a
  generic placeholder shape (🔴/🔵/🔺) plus the icon's shape *described in words* (taken
  directly from the fact cards, e.g., "oil can," "engine-block outline"), says everything
  the SVG would have said, without pretending to a level of icon-accuracy nothing here
  supports. This is explicitly a reference sheet, not a picture of a real dashboard —
  labeled as such throughout.
- **Engine bay and brake assembly** are the two cases flagged in the task instructions
  as needing real photographic/mechanical realism to avoid being misleading — an actual
  engine bay has a dozen-plus parts whose *relative position* matters (which side the
  battery is on, where the air box sits), and a brake assembly has real part shapes
  (caliper casting, rotor profile, drum shape, spring geometry) that no fact card in
  this task's source set specified precisely enough to draw without guessing. Both
  became detailed diagram briefs instead: they list every confirmed anchor fact, name
  exactly what is still missing and where to get it, specify labeling and style rules,
  and explicitly warn against inventing geometry — so a human illustrator, or a future
  research-and-draw pass with real reference photos or the service manual's own
  diagrams (redrawn, not traced), can finish them correctly.

## Fact-to-diagram traceability

Every shape, number, and label in the four finished SVGs and the reference sheet
traces to a Fact ID already verified in `facts/WP-04.md`, `facts/WP-07.md`,
`facts/WP-12.md`, `facts/WP-21.md`, or the Magic Seat procedure cards
(`procedures/WP-05-1.md` through `WP-05-4.md`). The only geometry that is *not* a direct
fact-to-pixel copy is the honest simplification the task instructions explicitly
invited — for example:

- The fuse-box "grid" layout is a schematic numbered arrangement, not the real physical
  layout of the fuse box lid (the manual's own layout was not extracted as a diagram,
  only as an ordered list of position/amperage/circuit — Fact F-WP21-014's method note
  explains how that ordered list was reconstructed).
- The jacking-point car outline is a generic rounded-rectangle silhouette with wheels,
  explicitly because Fact F-WP07-008 itself says the manual gives no numeric measurement
  and instructs against inventing one — the diagram shows "one point per corner,
  roughly at the rocker panel between the wheels," which is what the fact supports and
  no more.
- The Magic Seat panels are side-view rectangles standing in for cushions and
  seat-backs, with dashed outlines for the normal position and solid shapes for each
  mode — a simplification the task instructions asked for directly ("simple
  side-view schematic rectangles ... clearly simplified/schematic, not
  photorealistic"). The Long Mode panel is explicitly marked lower-confidence in its own
  caption, matching `procedures/WP-05-3.md`'s single-Tier-4-source caveat.
- The OBD-II diagram's footwell/steering-column shapes are a simplified interior
  side-view, not a scaled technical drawing — the port's marked position (under the
  dash, driver's side, near the column base, roughly pedal height) is the one detail
  that is fact-sourced; the seat, wheel, and pedal shapes around it are only there for
  orientation.

## Files in this folder

```
fuse_box_maps.svg                  Real SVG
jacking_points.svg                 Real SVG
magic_seat_modes.svg               Real SVG
obd_port_location.svg              Real SVG
dashboard_warning_lights.md        Reference-sheet Markdown (not an SVG — see rationale above)
engine_bay_DIAGRAM_BRIEF.md        Brief for a future illustrator (no SVG produced)
brake_assembly_DIAGRAM_BRIEF.md    Brief for a future illustrator (no SVG produced)
README.md                          This file
```

All four SVGs are self-contained, valid XML, and theme-aware (they render sensibly in
both a light and a dark viewer without any external assets).
