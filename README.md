# OceanEmbed 🌊
> **Deep Learning 3D Subsurface Ocean State Reconstruction from Surface Satellite Observations**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

**OceanEmbed** is a deep learning platform that reconstructs complete 3D subsurface ocean temperature fields and physical states from surface satellite remote sensing observations. 

While direct subsurface in-situ measurements (Argo floats) are sparse in space and time, satellite constellations provide continuous surface coverage. OceanEmbed bridges this gap by learning the nonlinear vertical stratification physics across the **North Indian Ocean (Arabian Sea & Bay of Bengal: 5°N–30°N, 45°E–105°E)** on a **0.25° grid (24,000 spatial nodes)** across **15 standard depth layers (0–1000 m)**.

---

## ✨ Key Features

- **🚀 Live Deep Learning Inference**: Powered by `OceanEmbedNet v2` (Encoder-Decoder CNN with 60,495 trainable parameters), running real-time 3D forward passes directly on CPU/GPU.
- **🛰️ 7 Surface Satellite Input Channels**:
  1. `SST`: Sea Surface Temperature (°C)
  2. `SSS`: Sea Surface Salinity (PSU)
  3. `SLA/SSH`: Sea Level Anomaly from satellite altimetry (m)
  4. `uwnd`: Zonal Surface Wind Velocity (m/s)
  5. `vwnd`: Meridional Surface Wind Velocity (m/s)
  6. `ucurr`: Zonal Surface Geostrophic Current (m/s)
  7. `vcurr`: Meridional Surface Current (m/s)
- **🌊 15 Standard Depths Inverted**: `0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000` meters.
- **🔬 Physical Oceanographic Quantities**:
  - Predicted Temperature (°C) & Ground Truth Reanalysis Reference
  - Subsurface Salinity Stratification (PSU)
  - Seawater Potential Density ($\rho$ in $\text{kg/m}^3$) & $\sigma_\theta$
  - Seawater Speed of Sound ($c$ in $\text{m/s}$) via Mackenzie 9-term equation
  - Mixed Layer Depth (MLD) & Thermocline Location ($\text{°C/m}$)
- **🗺️ Interactive Web Dashboard**:
  - Real-time basin heatmap with land masking and coastal reference points.
  - One-click **Regional Hotspots** (Central Bay of Bengal, Ganga Plume, EICC Boundary Current, Andaman Sea, Central Arabian Sea, Oman Upwelling, Equatorial Indian Ocean).
  - Sounding table with full numerical readouts across all 15 depths.
  - Dual-point comparison mode with signed difference metrics ($\Delta T, \Delta S, \Delta \rho, \Delta c$).
  - Automated AI physical oceanographic narrative summary.
  - Validation benchmarks, error distributions, and SHAP explainability.

---

## 📊 Model Performance & Benchmarks

The model was trained on multi-month satellite observations and validated against held-out test splits from the **Copernicus GLORYS12V1** reanalysis and independent **Argo float** observations:

- **Overall Ocean Test Set MAE**: `0.703 °C`
- **Overall Ocean Test Set RMSE**: `0.948 °C`

### Depth-Wise Performance Breakdown

| Depth (m) | RMSE (°C) | MAE (°C) | Bias (°C) | $R^2$ Correlation | Regime |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0 m** | 0.763 | 0.557 | -0.080 | 0.856 | Surface Layer |
| **5 m** | 0.734 | 0.538 | -0.126 | 0.870 | Surface Layer |
| **10 m** | 0.683 | 0.491 | -0.012 | 0.884 | Mixed Layer |
| **20 m** | 0.762 | 0.550 | -0.096 | 0.862 | Mixed Layer |
| **30 m** | 0.880 | 0.610 | -0.019 | 0.824 | Upper Thermocline |
| **50 m** | 1.202 | 0.894 | +0.083 | 0.685 | Core Thermocline |
| **75 m** | 1.384 | 1.059 | +0.236 | 0.593 | Core Thermocline |
| **100 m** | 1.328 | 1.034 | +0.070 | 0.694 | Lower Thermocline |
| **125 m** | 1.243 | 0.988 | +0.064 | 0.805 | Lower Thermocline |
| **150 m** | 1.150 | 0.916 | -0.023 | 0.841 | Intermediate Water |
| **200 m** | 0.896 | 0.678 | -0.029 | 0.886 | Intermediate Water |
| **300 m** | 0.768 | 0.539 | -0.061 | 0.882 | Deep Water |
| **500 m** | 0.587 | 0.411 | -0.044 | 0.891 | Deep Water |
| **700 m** | 0.611 | 0.433 | -0.015 | 0.876 | Abyssal Water |
| **1000 m** | 0.628 | 0.451 | -0.044 | 0.831 | Abyssal Water |

---

## 🏗️ Architecture

```
                       Input Satellite Observation Grid (7, 100, 240)
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   Conv2d (7 -> 16, 3x3)      Conv2d (16 -> 32, 3x3)
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                        Latent Representation Embedding (32, 100, 240)
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   Conv2d (32 -> 16, 3x3)     Conv2d (16 -> 15, 3x3)
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                      Reconstructed 3D Temperature Field (15, 100, 240)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/kanishkaanarang/oceanembed.git
cd oceanembed
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to interact with the model.

> **💡 Out-of-the-Box Ready**: The repository includes the trained PyTorch weights (`notebooks/results/oceanembed_cnn_best_v2.pt`) and an embedded compact reference dataset (`data/processed/oceanembed_reference_compact.npz`), so the application runs immediately upon cloning without any additional downloads!
>
> **Optional Full Reanalysis Arrays**: To use the complete 150-day Copernicus satellite arrays (250MB), simply place `training_arrays_v2_normalized.npz` into `training_arrays_v2_normalized/` or `data/processed/`. The application detects and switches to it automatically.
>
> **🎨 Light / Dark Theme**: OceanEmbed launches in a crisp, clean Light theme by default, and includes a one-click **Dark mode** toggle in the sidebar for nighttime oceanographic analysis.

---

## 📂 Project Structure

```
oceanembed/
├── app.py                     # Streamlit web application & interactive UI
├── backend.py                 # Production model adapter, physics & data service
├── model_adapter.py           # Verification contract & schema validation
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Protected ignore rules (prevents large data pushes)
├── src/
│   └── models/
│       ├── oceanembed_net.py  # OceanEmbedNet PyTorch CNN architecture & engine
│       └── baseline_xgb.py    # XGBoost tabular baseline model
├── results/
│   ├── depth_wise_metrics.csv # Official per-depth validation statistics
│   ├── validation_comparison_bar.png
│   ├── validation_error_map.png
│   ├── validation_scatter_agreement.png
│   └── shap_summary_v2.png    # SHAP feature importance plot
├── notebooks/
│   └── results/
│       ├── oceanembed_cnn_best.pt     # Checkpoint (v1)
│       └── oceanembed_cnn_best_v2.pt  # Production model weights (v2)
└── tests/
    └── test_backend.py        # Automated test suite
```

---

## 🧪 Running Tests

Run the test suite to verify backend contracts and model inference:
```bash
python -m unittest tests/test_backend.py
```

---

## 📄 License
This project is licensed under the MIT License.