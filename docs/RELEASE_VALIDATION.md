# Showcase release validation

## V3 serving upgrade

The dashboard and profile API now execute `oceanembed_v3_qc_full/best.pt` with
its saved feature schema, missing-value indicators and target normalization.
Both temperature and salinity are predicted. Its 15 physical depth labels end
at 900 m; 1000 m is not relabeled or extrapolated. Existing browser selections
are migrated to the supported range. Historical v2 artifacts remain available
in a labeled expander.

Final upgrade run: **82 tests passed, 22 subtests passed**, in 65.69 seconds.
Two dependency deprecation warnings remain. New checks compare serving output
directly with checkpoint inference, verify both map heads, API/dashboard
agreement, metadata, unsupported-depth rejection and old-session migration.
Live browser checks confirmed v3 provenance, both themes, map/profile rendering
and the validation page without exception panels or horizontal overflow.

The saved v3 test scores remain limited to held-out day 5 of the five-day
GLORYS experiment. Serving additional dates does not expand that validation
claim. Grid temperature reference at 900 m is interpolated from 700/1000 m;
no measured salinity reference is fabricated.

## Previous showcase release

Validated on 15 September 2026 using the project's Windows virtual environment.

## Automated checks

Command: `.venv/Scripts/python.exe -m pytest tests -q`

Result: **78 passed, 22 subtests passed**, in 40.64 seconds. Two upstream
Starlette/AnyIO deprecation warnings remain; there are no test failures.

Coverage includes all six dashboard pages, light/dark round trips, retained
controls and comparisons, all eight map layers, both transect directions,
date/depth boundaries, regional and event presets, isolated saved-location
operations, profile values, CSV/NetCDF exports, mission diagnostics, API
compatibility, and existing model/physics tests.

## Browser checks

- Inspected all six pages in light and dark mode at desktop width.
- Confirmed ocean-cell selection updates the active location and profile.
- Checked dark table cells, dropdowns, status badges, chart legends and axes.
- Verified the temperature/salinity/sound-speed overlay has separate axes.
- Checked a 390-pixel mobile viewport for horizontal page overflow.
- Confirmed navigation resets the main scroll position.
- Final browser check reported no Streamlit exception panels or page overflow;
  the browser error log was empty.

These checks validate the application and existing computations. They do not
constitute new model training or independent oceanographic skill validation.
See `SHOWCASE_WALKTHROUGH.md` for the demo flow and model limitations.
