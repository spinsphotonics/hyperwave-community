Source ID: S-WP15-B
Title: "Select a type of part to check" — Aftermarket Parts Lookup tool
Author/Publisher: California Air Resources Board (CARB)
URL: https://ww2.arb.ca.gov/aftermarket-parts-lookup
Type: article (official CARB public web tool / landing page)
Tier: 2
Covers model year 2008 GD3? unclear — this is CARB's general public lookup tool for Executive Order (EO) numbers, not specific to any one vehicle. A reader would use it to check whether a specific aftermarket part carries an EO covering their own 2008 Fit.
Date published: unknown (live CARB web application; content is maintained/current, not a dated document)
Date accessed: 2026-09-07
Paywalled? no
Notes: Retrieved via direct `curl` (HTTP 200) per ADAPT-2 — confirmed live and reachable, not a dead link or invented URL. Page presents five categories a reader can pick to search Executive Orders by device type: "Performance & Add on Parts," "Catalytic Converters," "Motorcycles," "Auxiliary Fuel Tanks," "Alternate Fuel Retrofit Systems." The "Performance & Add on Parts" link goes to `https://ww2.arb.ca.gov/applications/aftermarket-parts-database` (also confirmed HTTP 200 via curl; page shell loads but the actual search form is a JavaScript application not renderable through curl's static fetch — expected for an interactive search tool, not a broken-link problem). Reached this page by following CARB's own site navigation from its top-level "Aftermarket, Performance, and Add-on Parts" program page (see S-WP15-D) rather than guessing a URL, so this is confirmed as CARB's real, current public-facing EO lookup entry point.
