# Showcase release validation

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
