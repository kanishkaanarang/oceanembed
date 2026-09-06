# OceanEmbed frontend demo

A self-contained SIH concept prototype for exploring subsurface ocean temperature profiles in the Bay of Bengal. Built with React, TypeScript, Vite, Tailwind CSS, Recharts, and Lucide.

**All ten scenarios are synthetic. No trained model is connected.** Existing Python scripts and research notebooks in the parent repository are preserved.

## Start locally

Use Node.js 22.18+ (Node 24 recommended) and npm. In this directory:

```powershell
npm install
npm run dev
```

Open http://127.0.0.1:5173. For a reproducible install from the committed lockfile, use `npm ci` instead of `npm install`.

```powershell
npm test
npm run build
npm run preview
```

The production preview defaults to http://127.0.0.1:4173. The build output is `dist/`, which can be served by any static web server. No backend, API keys, authentication, Python environment, GPU, or database is required.

The initial package download needs internet. Once installed, the app can be served locally without internet. Fonts use the system font stack; maps, icons, scripts, and scenarios are bundled locally. Keep the local server running during the presentation. This is not a service-worker/PWA installation: opening a never-loaded hosted URL while completely offline is not supported.

## What is included

- **Explorer:** five clickable ocean locations, two seasonal examples each, surface inputs, prepared profile loading, depth inspection, reference toggle, MAE/RMSE, data table, and CSV export.
- **Compare:** two independently selectable scenarios, profile overlays, signed temperature differences at each depth, and a swap action.
- **How It Works:** the proposed scientific workflow, existing research foundation, planned work, and a plain-language glossary.
- Keyboard-operable native controls, SVG map buttons, focus indicators, reduced-motion support, and responsive layouts.
- Cancellation of stale scenario loads. Changing location or date clears the old result and export. Reset also cancels pending work.

A labelled prepared Central Bay example appears on first load so the dashboard immediately demonstrates the idea. The generate action reloads a local fixture with a short presentation delay, not model inference. Explorer selection and results persist across navigation within the page; reloading the page restores defaults.

## Scenarios and units

Five locations: Central Bay, Northern Bay, Western Bay, Andaman Basin, and Southern Bay. Each has a pre-monsoon example dated 15 April 2024 and a post-monsoon example dated 15 November 2024. These dates describe fictional scenarios, not observations collected on those dates.

Depths are 0, 10, 25, 50, 75, 100, 150, 200, 300, 400, and 500 metres. Temperature is °C; sea-level anomaly is metres; surface salinity uses the conventional PSU display label. Depth spacing is numeric rather than categorical, with the surface at the top of the chart.

The hand-authored local SVG coastline is a simplified geographic illustration using the same equirectangular transform for land and markers. It is not a navigational chart, satellite image, bathymetry map, or live observation layer. Decorative contours carry no measured depth information.

The estimates and reference profiles are separately authored synthetic arrays. Neither is actual Argo data. MAE and RMSE are calculated over valid paired depths, equally weighted per prepared depth, with missing pairs excluded. They describe agreement within the fictional scenario, not model validation or independent accuracy.

CSV export contains scenario identity, date, coordinates, season, depth, temperatures, surface inputs, units in the column headers, and synthetic provenance on every data row. The browser performs the download through a local Blob; no server upload occurs.

## Code layout

```text
src/
  App.tsx                         Navigation and persistent Explorer state
  types.ts                        Scenario and profile contracts
  data/scenarios.ts               Ten deterministic fixtures
  services/scenarioService.ts     Typed, cancellable local scenario loader
  lib/calculations.ts             Metrics, interpretation, formatting and CSV
  components/
    Explorer.tsx                  Selection, cancellation and profile workflow
    OceanMap.tsx                  Bundled SVG geography and accessible markers
    ProfileChart.tsx              Shared numeric temperature-depth chart
    Compare.tsx                   Two-scenario exploration
    SurfaceInputs.tsx             Reusable input cards
    HowItWorks.tsx                Concept, roadmap and glossary
  styles.css                      Tailwind integration and responsive design
tests/core.test.ts                Data, calculation, export and cancellation checks
verification/                    Browser check scripts and local screenshots
PRESENTATION.md                   Two-minute walkthrough
```

## Future model integration

1. Build quality-checked matched surface/Argo training data. Record product provenance, units, location/time tolerances, and missing-data handling.
2. Train a baseline and evaluate on separated test profiles. Keep surface reanalysis comparisons distinct from learned-model accuracy.
3. Replace the local scenario loader with a genuine typed service. Return profile values, model version, input provenance, supported region/date coverage, and explicit failures.
4. Extend the `provenance` contract before adding observed data; the current type deliberately accepts only `synthetic`.
5. Update the UI labels only when backed by the returned provenance. Keep observations, reanalysis, predictions, and uncertainty distinct.
6. Add uncertainty and explanation panels only after they have been computed and assessed.

The research README mentions learned embeddings, ConvLSTM/Transformer models, and SHAP. Those are not implemented by this frontend. The existing reference-comparison errors are not reused as model performance.

## Verification

Run `npm test` for deterministic checks and `npm run build` for strict TypeScript compilation and production bundling. See `verification/VERIFICATION.md` for the observed browser checks and limitations. Browser automation is optional and is not an application dependency.

Implementation references: [Vite](https://vite.dev/guide/), [Tailwind with Vite](https://tailwindcss.com/docs/installation/using-vite), [Recharts](https://recharts.github.io/en-US/api/Scatter/).
