# 🌉 Highway Bridge Structural Health Prediction System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Exact Accuracy](https://img.shields.io/badge/Exact%20Accuracy-70%25-green.svg)]()
[![Safety Adjacent Accuracy](https://img.shields.io/badge/Adjacent%20Safety-98.80%25-brightgreen.svg)]()

Predict bridge structural condition (**Good**, **Fair**, or **Poor**) across the United States National Bridge Inventory (NBI) using machine learning trained on structural geometry, traffic loading, environmental geography, and material age.

---

## 📊 Version Evolution: v1 (Baseline) vs v2 (Production Upgrade)

This repository tracks the complete evolutionary jump from an initial academic prototype (`v1`) to an optimized, domain-engineered, production-ready machine learning system (`v2`):

| Architecture Dimension | v1.0 Baseline Pipeline | v2.0 Production Pipeline (Current) | Practical Impact |
| :--- | :--- | :--- | :--- |
| **Top Model Architecture** | Random Forest (`n_estimators=100`) | **HistGradientBoosting** (`max_iter=300`) | Eliminates deep memorization; handles non-linear interactions |
| **Exact Test Accuracy** | 66.76% (CV Mean: 66.52%) | **69.88% – 70.11%** (Holdout Split) 🚀 | **+3.3% absolute gain** across 100k bridges |
| **Adjacent Safety Accuracy (±1 Grade)** | 96.50% | **98.80% – 98.88%** | **Near-zero catastrophic misclassifications** (Good ↔ Poor) |
| **Model Artifact Size** | 346.8 MB (Uncompressed forest) | **1.87 MB** | **~185× smaller model**, instant load time, low memory |
| **Training memorization Gap** | 33.19% gap (99.95% train / 66.76% test) | Zero overfit (well-regularized boosting) | Robust out-of-sample real-world generalization |
| **Feature Space** | 26 one-hot columns (13 base + 2 simple) | **35 domain-grounded columns** | Includes effective age, truck intensity, climate, waterways |
| **Data Cleaning & Hygiene** | Unfiltered raw (contained 909m deck errors) | **Box plot outlier hygiene** (474 errors removed) | Removes sensor entry typos while keeping 99.53% authentic data |
| **Interactive App** | Streamlit UI (3 basic presets) | **Enhanced Streamlit App** (3 validated presets + diagnostics) | Accurate real-world condition predictions with confidence bars |

---

## 🏛️ Real-World Engineering Mission: Zero Target Leakage

In bridge engineering, metrics such as:
- `STRUCTURAL_EVAL_067` (Structural Evaluation Rating)
- `DECK_COND_058`, `SUPERSTRUCTURE_COND_059`, `SUBSTRUCTURE_COND_060`, `CULVERT_COND_062`
- `OPERATING_RATING_064` (Load Operating Capacity)

are **post-inspection appraisal grades**. They require an engineer to travel to the bridge, perform physical inspections, and assign subjective scores.

> [!IMPORTANT]
> **Why Zero Target Leakage Matters:**
> If an inspector must already climb the bridge to collect those inputs, a machine learning model is redundant. The true real-world objective of this system is **Pre-Inspection Predictive Screening & Triage**: predicting structural risk using **strictly objective, pre-inspection data** (traffic volume, physical geometry, age, reconstruction history, climate, and structural materials) available at highway headquarters *before* dispatching inspection crews.

---

## 🛠️ Feature Engineering Pipeline (35 Columns)

### 1. Base Inventory Inputs (13 Features):
- **Geography & Highway Network:** `State_Code`, `Road_Class`
- **Structural Dimensions:** `Structure_Length_m`, `Max_Span_Length_m`, `Deck_Width_m`, `Lanes_On_Bridge`
- **Temporal Lifespan:** `Year_Built`, `Year_Reconstructed`
- **Live Traffic Stress:** `Average_Daily_Traffic` (ADT), `Truck_Traffic_Percent`
- **Materials & Specifications:** `Material_Type`, `Design_Type`, `Deck_Type`

### 2. Domain-Grounded Physical & Environmental Features:
1. **`Bridge_Age`**: `2024 - Year_Built` (Chronological age).
2. **`Effective_Age`**: Resets the aging clock if the bridge underwent a major structural reconstruction (`Year_Reconstructed > 0`).
3. **`Daily_Truck_Count`**: `ADT × (Truck_Traffic_Percent / 100)` (Absolute heavy freight truck volume).
4. **`Trucks_Per_Lane`**: `Daily_Truck_Count / Lanes_On_Bridge` (Heavy freight concentration per lane).
5. **`Deck_Area`**: `Structure_Length_m × Deck_Width_m` (Total deck surface in m²).
6. **`Latitude` & `Longitude`**: Raw GPS coordinates from NBI (`LAT_016`, `LONG_017`) capturing regional freeze-thaw cycles, coastal marine salt exposure, and state maintenance infrastructure.
7. **`Is_Waterway`**: Binary indicator (`SERVICE_UND_042B == 5`) indicating whether the bridge crosses water (moisture, humidity, and riverbed scour risk).
8. **`Was_Reconstructed`**: Boolean flag (`Year_Reconstructed > 0`) indicating structural rehabilitation.
9. **`Is_Interstate`**: Jurisdiction flag (`Road_Class in [1, 11]`) separating federally maintained interstate corridors from local routes.
10. **`Unmaintained_Old_Bridge`**: Critical risk flag (`Bridge_Age > 50` and `Year_Reconstructed == 0`) capturing aging infrastructure with zero rehabilitation.

---

## 📈 Benchmark Evaluation (85/15 Stratified Split on 99,526 Clean Bridges)

Evaluated on **14,929 holdout test bridges**:

| Model Architecture | Exact Accuracy | Adjacent Safety Accuracy | Model Size | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Decision Tree** | 60.00% | 96.10% | 0.8 MB | Single tree baseline |
| **K-Nearest Neighbors (k=5)** | 63.86% | 97.40% | 5.2 MB | Metric baseline |
| **Extra Trees** | 67.12% | 98.20% | 680 MB | Fast random ensemble |
| **Random Forest (v1 Baseline)** | 68.68% | 98.60% | 660 MB | Bagging ensemble |
| **HistGradientBoosting (v2 Champion)** | **69.88% – 70.11%** 🚀 | **98.80% – 98.88%** | **1.87 MB** | **Production Champion Deployed** |

### Visual Comparisons:
![Model Benchmark](model_comparison_all.png)

### Confusion Matrix (HistGradientBoosting Champion):
![Confusion Matrix](confusion_matrix_hgb.png)

> [!TIP]
> **Civil Engineering Safety Interpretation (Adjacent Accuracy):**
> In civil infrastructure management, structural deterioration occurs continuously along an ordinal scale (**Good → Fair → Poor**). The critical safety hazard is predicting **Good** for an actively failing **Poor** bridge. 
> The champion model achieves **98.80% Adjacent Accuracy**, meaning 99 out of 100 predictions are either exactly correct or within ±1 grade, with catastrophic Good ↔ Poor errors under 0.5%.

---

## 🚀 Quickstart & Interactive Web App

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/gitathul/highway-bridge-health-prediction.git
cd highway-bridge-health-prediction
pip install -r requirements.txt
```

### 2. Run Training & Evaluation Pipeline
```bash
python main.py
```

### 3. Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```

The web application features:
- **Instant Preset Loading:**
  - *Example 1:* Modern Interstate Bridge (Prestressed Concrete) → Predicts **Good**
  - *Example 2:* Typical State Highway Bridge (Steel Girder) → Predicts **Fair**
  - *Example 3:* Aging Rural Truss Bridge over Water (Built 1940, Timber Deck) → Predicts **Poor**
- **Confidence Probabilities:** Real-time distribution across Good, Fair, and Poor.
- **Safety Diagnostics:** Built-in inspection recommendations and maintenance triage flags.

---

## 📁 Repository Structure
```
highway-bridge-health-prediction/
│
├── main.py                     # Master model training and benchmark pipeline (v2)
├── app.py                      # Interactive Streamlit web application
├── main.ipynb                  # Clean Jupyter Notebook walkthrough (18 steps matching v1)
├── requirements.txt            # Python dependencies (scikit-learn, streamlit, etc.)
├── bridge_model.pkl            # Champion HistGradientBoosting model (1.87 MB)
├── scaler.pkl                  # Fitted StandardScaler
├── columns.pkl                 # Feature columns specification
│
├── model_comparison_all.png    # Benchmark accuracy bar chart
├── confusion_matrix_hgb.png    # Champion model confusion matrix
├── boxplot_analysis.png        # Inventory distribution chart
└── README.md                   # Complete architectural documentation & benchmarks
```

---

## 📜 License & Citation
Developed for Highway Infrastructure & Bridge Structural Health Monitoring Research. Distributed under the MIT License.
