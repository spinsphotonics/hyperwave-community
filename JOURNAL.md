# Journal

Shared memory between context windows. Read this at the start of every session.

---

## GC Inverse Design Notebook (`examples/gc_inverse_design.ipynb`)

### Reference
The notebook's structure and layer stack matches the reference GC inverse design
from the sibling hyperwave repo:
- **Template:** `hyperwave/devices/gc/template_57pct/gc_template_uniform_20260129_1343.py`
- **Full inverse design:** `hyperwave/devices/gc/inverse_design/gc_inverse_design.py`

### Structure: Intentional Differences from Reference

The notebook is a **cloud-first** rewrite. All FDTD runs on cloud GPU; only 2D
filtering and plotting run locally. This means:

1. **`recipe_from_params()` instead of `create_structure()`**: No 3D permittivity
   array is built locally. The cloud reconstructs from the lightweight recipe dict.
   (Same as reference template.)

2. **Power loss (`power_axis=0`) instead of mode overlap**: The notebook uses
   Poynting vector Sx through the waveguide output monitor. This avoids needing
   a pre-computed waveguide mode and simplifies the workflow for users. The
   reference uses mode overlap loss (requires `mode_converter` + `hws.mode()`).

3. **`generate_gaussian_source()` on cloud GPU**: Instead of the local
   `create_gaussian_source()` which needs the full hyperwave solver, FDTD runs
   in free space on cloud GPU, then wave equation error is computed locally
   (~1.5 GB peak, fine for Colab).

4. **`mode_converter` reverted to solver-only**: Cloud fallback was removed.
   The notebook doesn't use `mode_converter` at all (power loss, not mode overlap).

### Structure: What Matches the Reference Exactly

- **8-layer SOI stack** (pad/air/clad/etch/dev/BOX/sub/pad)
- **Two design layers**: etch (`density_top` from theta) and dev (`density_bottom = where(theta > 0, 1.0, 0.0)`)
- **Tuple permittivity** `(eps_clad, eps_si)` on both design layers
- **Uniform layers** use `slab_density = hwc.density(jnp.zeros(...), pad_width=0)`
- **Density filtering**: `radius=3, alpha=0.8, pad_width=0`
- **`vertical_radius=0`** (no z-blurring)
- **Float thicknesses** in pixels (subpixel averaging for thin layers)

### Current State (2026-02-08)

- Branch: `feb-inverse-design-notebook`
- Notebook fully rewritten for cloud-only execution
- `__init__.py`: Added `recipe_from_params`, `generate_gaussian_source` exports
- `sources.py`: Added `generate_gaussian_source()` cloud function + helpers,
  reverted `mode_converter` cloud fallback
- Ready to commit and push

### Lessons Learned

1. **Always match the reference structure** from `hyperwave/devices/gc/`. When
   in doubt, read the template script. The structure has TWO design layers, not
   one solid slab.
2. **Document design decisions** in this journal. Context windows forget
   everything not written down.
3. **`recipe_from_params` must be exported from `__init__.py`** for the notebook
   to use `hwc.recipe_from_params()`.
4. **Float layer thicknesses** are critical for subpixel averaging. Rounding to
   int shifts thin layer interfaces by up to 0.5 cells.
