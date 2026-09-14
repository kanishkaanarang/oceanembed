# OceanEmbed showcase

Open http://localhost:8501 and refresh if a previously open tab shows an old page.

## Suggested five-minute flow

1. **Overview (30 seconds):** “Satellites observe the ocean surface. OceanEmbed reconstructs the vertical water column so we can explore what lies below.” Choose a regional scenario.
2. **Ocean Explorer (90 seconds):** Show a temperature map, select a depth, and inspect the adjacent vertical profile. Switch to the vertical transect and demonstrate both slice directions. Change to salinity or sound speed. The same location, date, depth, and layer remain selected when changing pages.
3. **Profiles & Data (45 seconds):** Inspect all 15 layers, enable a second location in the sidebar, and download a CSV or NetCDF sounding. Open the surface-forcing section or the multivariable chart if asked.
4. **Mission Intelligence (60 seconds):** Explain D26 and heat integrated above 26°C, then show the acoustic profile and stratification diagnostics. A heat-potential band alone is not a cyclone-category forecast.
5. **Validation (45 seconds):** Explain the serving CNN evaluation and the separate v3 experiment. The v3 result is 0.271°C / 0.103 PSU on a five-day GLORYS split ending at 900 m; it is not the live map model's score or independent satellite-to-Argo validation.
6. **About & Saved Places (30 seconds):** Explain the seven-channel CNN inputs, depth outputs, and local saved locations.

## Feature locations

| Page | Features |
| --- | --- |
| Overview | Regional scenario buttons, selected water-column summary, interpretation, shortcuts |
| Ocean Explorer | Eight map layers, selectable map, vertical profiles, zonal/meridional transects, numeric inspection, grid CSV and sounding NetCDF |
| Profiles & Data | Complete depth table, selected-depth metrics, surface forcing, two-point comparison, reference casts, T–S–c overlay, CSV/NetCDF |
| Mission Intelligence | Existing heat, acoustic and stratification diagnostics, previews, comparison, methodology and analysis exports |
| Validation | V3 candidate report, serving CNN depth metrics, evaluation figures and feature attribution |
| About & Saved Places | Architecture, input channels, output depths, saved-location list and clearing |

The sidebar retains date, location, depth, layer, presets, comparison controls, save location, and light/dark mode. Only the selected page renders its panels. Hidden preview selections persist, and page navigation resets the main scroll position.

## Accurate technical answers

- The serving engine currently loads the **v2 CNN**. The v3 SE-ResNet checkpoint is a separately evaluated candidate.
- Salinity in the serving dashboard is an existing diagnostic estimate; density and sound speed are derived from temperature and estimated salinity.
- Confidence is a heuristic, not a calibrated predictive probability.
- Nearby reference casts are illustrative records, not a live Argo feed.
- Horizontal fields and transects now use the same calculations and units for every layer, including Confidence and Embedding.

## Starting the local app

From the `oceanembed` directory:

```powershell
.venv/Scripts/python.exe -m streamlit run app.py --server.port 8501
```

If port 8501 is already serving OceanEmbed, use the running application. No deployment is required for the local showcase.
