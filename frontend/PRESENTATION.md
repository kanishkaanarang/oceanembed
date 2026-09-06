# Two-minute OceanEmbed walkthrough

Before presenting: run `npm run dev`, open http://127.0.0.1:5173, reload the page, and keep the local server running. Prefer a desktop browser. No internet connection is needed after dependencies are installed. Record a short backup walkthrough if your presentation setup permits it.

## 0:00–0:20 — The problem and scope

“OceanEmbed explores how surface ocean conditions might help us estimate temperatures below the surface. Surface observations cover broad areas, while direct subsurface measurements are sparse. Our focus is the Bay of Bengal. This is an interactive concept prototype using clearly labelled synthetic scenarios; it does not yet run a trained model.”

Point to the persistent Demo mode badge.

## 0:20–0:50 — Explore a scenario

Select **Northern Bay** on the map. Keep the April pre-monsoon scenario. Point to the temperature, sea-level anomaly, and salinity input cards, then click **Generate demo profile**.

“A researcher chooses a location and date, reviews the surface inputs, and explores a profile below the surface. Here we load a prepared example to show the intended workflow.”

## 0:50–1:15 — Read the profile

Move the depth slider. Explain that depth increases downward and temperature is on the horizontal axis. Toggle the synthetic reference and open **Explore the numbers**.

“This curve illustrates how temperature changes with depth. We can inspect any prepared depth and compare the estimate with a reference. Both curves here are synthetic. The displayed agreement metrics are computed from these pairs; they are not validated AI accuracy.”

Show the CSV export control and its provenance-bearing output if browser downloads are permitted in the presentation environment.

## 1:15–1:40 — Compare seasons

Open **Compare**. Set scenario A to **Central Bay · Pre-monsoon** and B to **Central Bay · Post-monsoon**. At 100 m, the prepared estimates are 21.5°C and 23.3°C, a signed A-minus-B difference of −1.8°C.

“The comparison view shows how users could investigate differences across seasons and locations. These examples illustrate the interface; they do not establish measured seasonal change.”

## 1:40–2:00 — Explain what comes next

Open **How It Works** and point to the research workflow and three status cards.

“Our repository contains initial data acquisition and exploration. The next milestone is a matched training dataset, a trained baseline, and evaluation on held-out Argo profiles. This frontend provides the workflow into which those results can be integrated.”

## If asked whether the AI is working

“The interactive software workflow is working. Its current outputs are prepared synthetic fixtures. We have not connected or validated a trained prediction model, and we keep that distinction visible throughout the demo.”
