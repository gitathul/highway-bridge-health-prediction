# Highway Bridge Health Prediction System (v1 - Baseline)

Predict bridge structural condition (**Good**, **Fair**, or **Poor**) based on National Bridge Inventory (NBI) physical attributes and traffic stress.

## 1. Feature Architecture (v1 Baseline)
- **13 Physical Inventory Inputs:** `State_Code`, `Road_Class`, `Year_Built`, `Year_Reconstructed`, `Lanes_On_Bridge`, `Average_Daily_Traffic`, `Truck_Traffic_Percent`, `Structure_Length_m`, `Max_Span_Length_m`, `Deck_Width_m`, `Material_Type`, `Design_Type`, `Deck_Type`.
- **2 Engineered Features:**
  - `Bridge_Age = 2024 - Year_Built`
  - `Daily_Truck_Count = Average_Daily_Traffic * (Truck_Traffic_Percent / 100)`
- Total One-Hot Encoded Features: **26 columns**.

## 2. Model Performance Benchmark (80/20 Stratified Split)
Evaluated on **20,000 holdout test bridges**:

| Model Architecture | Accuracy | Key Observations |
| :--- | :---: | :--- |
| **Decision Tree** | 58.31% | Single decision tree baseline |
| **K-Nearest Neighbors (k=5)** | 62.26% | Distance-based baseline |
| **Random Forest (Champion)** | **66.76%** | **Cross-Validation Mean: 66.52%** |

> [!NOTE]
> *v1 Limitations:* The baseline Random Forest model exhibits high training memorization (99.95% train accuracy vs 66.76% test accuracy) and has a heavy 346.8 MB model artifact.
