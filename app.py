import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page Configuration & Header
# --------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Highway Bridge Health Prediction",
    page_icon="🌉",
    layout="wide"
)

st.title("🌉 Highway Bridge Health Prediction System")
st.write("Predict bridge structural condition (Good, Fair, or Poor) based on physical attributes, traffic stress, and construction materials.")

# --------------------------------------------------
# 2. Example Presets Configuration
# --------------------------------------------------
# Preset bridges to auto-fill realistic values for quick testing
presets = {
    "Custom Bridge (Enter your own values)": {
        "state": 29, "road": 2.0, "built": 1985, "recon": 0, "lanes": 2,
        "adt": 3500, "truck": 7.0, "len": 40.0, "span": 18.0, "width": 10.0,
        "mat": "Steel", "des": "Stringer/Girder", "deck": "Concrete Cast-in-Place"
    },
    "Example 1: Modern Interstate Bridge (Prestressed Concrete)": {
        "state": 29, "road": 1.0, "built": 2018, "recon": 0, "lanes": 4,
        "adt": 45000, "truck": 12.0, "len": 120.0, "span": 35.0, "width": 18.0,
        "mat": "Prestressed concrete", "des": "Box Beam", "deck": "Concrete Precast"
    },
    "Example 2: Typical State Highway Bridge (Steel Girder)": {
        "state": 26, "road": 2.0, "built": 1975, "recon": 1995, "lanes": 2,
        "adt": 8500, "truck": 9.0, "len": 55.0, "span": 20.0, "width": 11.0,
        "mat": "Steel", "des": "Stringer/Girder", "deck": "Concrete Cast-in-Place"
    },
    "Example 3: Aging Rural Bridge (Old Timber)": {
        "state": 48, "road": 9.0, "built": 1935, "recon": 0, "lanes": 1,
        "adt": 350, "truck": 4.0, "len": 18.0, "span": 8.0, "width": 6.0,
        "mat": "Timber", "des": "Stringer/Girder", "deck": "Timber"
    }
}

# Dropdown to choose a preset
preset_choice = st.selectbox(
    "⚡ Quick Load Preset (Choose a sample bridge or customize below):",
    list(presets.keys())
)
data = presets[preset_choice]

st.divider()

# --------------------------------------------------
# 3. User Input Form (2 Columns)
# --------------------------------------------------
col1, col2 = st.columns(2)

# --- Column 1: Location, Age & Traffic Load ---
with col1:
    st.subheader("📍 Location, Age & Traffic")
    
    # State code (1 - 78)
    state_code = st.slider(
        "State Code",
        min_value=1, max_value=78,
        value=int(data["state"]),
        key=f"state_{preset_choice}"
    )
    
    # Road functional classification
    road_classes = [1.0, 2.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 14.0, 16.0, 17.0, 19.0]
    road_class_labels = {
        1.0: "1.0 - Rural Interstate Highway",
        2.0: "2.0 - Rural Principal Arterial",
        6.0: "6.0 - Rural Minor Arterial",
        7.0: "7.0 - Rural Major Collector",
        8.0: "8.0 - Rural Minor Collector",
        9.0: "9.0 - Rural Local Road",
        11.0: "11.0 - Urban Interstate Highway",
        12.0: "12.0 - Urban Freeway / Expressway",
        14.0: "14.0 - Urban Other Principal Arterial",
        16.0: "16.0 - Urban Minor Arterial",
        17.0: "17.0 - Urban Collector",
        19.0: "19.0 - Urban Local Street"
    }
    road_class = st.selectbox(
        "Road Functional Class",
        road_classes,
        index=road_classes.index(data["road"]),
        format_func=lambda x: road_class_labels.get(x, str(x)),
        key=f"road_{preset_choice}"
    )

    # Age and reconstruction years
    year_built = st.slider("Year Built", min_value=1880, max_value=2024, value=int(data["built"]), key=f"built_{preset_choice}")
    year_reconstructed = st.slider("Year Reconstructed (0 = never reconstructed)", min_value=0, max_value=2024, value=int(data["recon"]), key=f"recon_{preset_choice}")
    lanes_on_bridge = st.slider("Number of Traffic Lanes", min_value=1, max_value=12, value=int(data["lanes"]), key=f"lanes_{preset_choice}")
    
    # Traffic volume & heavy truck percentage
    average_daily_traffic = st.number_input("Average Daily Traffic (vehicles/day)", min_value=0, max_value=500000, value=int(data["adt"]), step=500, key=f"adt_{preset_choice}")
    truck_traffic_percent = st.slider("Truck Traffic Percentage (%)", min_value=0.0, max_value=100.0, value=float(data["truck"]), step=0.5, key=f"truck_{preset_choice}")

# --- Column 2: Dimensions & Construction Materials ---
with col2:
    st.subheader("🏗️ Structure & Materials")
    
    # Bridge geometry in meters
    structure_length_m = st.number_input("Total Bridge Length (meters)", min_value=5.0, max_value=5000.0, value=float(data["len"]), step=5.0, key=f"len_{preset_choice}")
    max_span_length_m = st.number_input("Maximum Span Length (meters)", min_value=2.0, max_value=1000.0, value=float(data["span"]), step=2.0, key=f"span_{preset_choice}")
    deck_width_m = st.slider("Deck Width (meters)", min_value=3.0, max_value=40.0, value=float(data["width"]), step=0.5, key=f"width_{preset_choice}")
    
    # Categorical material and structural design types
    materials = ['Steel', 'Concrete', 'Prestressed concrete', 'Concrete continuous', 'Steel continuous', 'Timber']
    material_type = st.selectbox("Material Type", materials, index=materials.index(data["mat"]), key=f"mat_{preset_choice}")
    
    designs = ['Stringer/Girder', 'Tee Beam', 'Slab', 'Girder Floorbeam', 'Truss', 'Box Beam']
    design_type = st.selectbox("Design Type", designs, index=designs.index(data["des"]), key=f"des_{preset_choice}")
    
    decks = ['Concrete Cast-in-Place', 'Concrete Precast', 'Open Steel Grid', 'Timber']
    deck_type = st.selectbox("Deck Type", decks, index=decks.index(data["deck"]), key=f"deck_{preset_choice}")

st.divider()

# --------------------------------------------------
# 4. Model Prediction & Processing
# --------------------------------------------------
if st.button("🔍 Predict Bridge Health", use_container_width=True):
    # Load saved models from pickle
    model = pickle.load(open(APP_DIR / "bridge_model.pkl", "rb"))
    scaler = pickle.load(open(APP_DIR / "scaler.pkl", "rb"))
    columns = pickle.load(open(APP_DIR / "columns.pkl", "rb"))

    # Feature Engineering (matching Cell 6 of notebook)
    bridge_age = 2024 - year_built
    daily_truck_count = average_daily_traffic * (truck_traffic_percent / 100)

    # Build input DataFrame
    input_df = pd.DataFrame([{
        "State_Code": state_code,
        "Road_Class": road_class,
        "Year_Built": year_built,
        "Year_Reconstructed": year_reconstructed,
        "Lanes_On_Bridge": lanes_on_bridge,
        "Average_Daily_Traffic": average_daily_traffic,
        "Truck_Traffic_Percent": truck_traffic_percent,
        "Structure_Length_m": structure_length_m,
        "Max_Span_Length_m": max_span_length_m,
        "Deck_Width_m": deck_width_m,
        "Bridge_Age": bridge_age,
        "Daily_Truck_Count": daily_truck_count,
        "Material_Type": material_type,
        "Design_Type": design_type,
        "Deck_Type": deck_type
    }])

    # One-hot encode categorical features and align to 26 trained columns
    input_encoded = pd.get_dummies(input_df, drop_first=True, dtype=int)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale inputs using the fitted StandardScaler
    input_scaled = scaler.transform(input_encoded)

    # Model inference: predicted condition and class probabilities
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    # --------------------------------------------------
    # 5. Structural Health Assessment Report
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Structural Assessment & Health Report")

    # Key metric summary cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if prediction == "Good":
            st.metric("Predicted Condition", "GOOD 🟢", delta="Structurally Sound", delta_color="normal")
        elif prediction == "Fair":
            st.metric("Predicted Condition", "FAIR 🟡", delta="Minor Deterioration", delta_color="off")
        else:
            st.metric("Predicted Condition", "POOR 🔴", delta="Action Required", delta_color="inverse")

    with m2:
        st.metric("Confidence Score", f"{max(probs)*100:.1f}%", delta=f"{prediction} Probability", delta_color="off")

    with m3:
        st.metric("Calculated Bridge Age", f"{bridge_age} Years", delta=f"Built in {year_built}", delta_color="off")

    with m4:
        st.metric("Daily Truck Traffic", f"{int(daily_truck_count):,} Trucks", delta=f"{truck_traffic_percent}% of ADT", delta_color="off")

    st.write("")

    # Visual Probability Breakdown
    st.markdown("#### 🎯 Model Probability Breakdown")
    
    # 3 Metrics side-by-side
    p1, p2, p3 = st.columns(3)
    for i, c in enumerate(classes):
        [p1, p2, p3][i].metric(label=c, value=f"{round(probs[i] * 100, 1)}%")

    # Native Streamlit Probability Bar Chart
    prob_df = pd.DataFrame({
        "Condition": classes,
        "Probability (%)": [round(p * 100, 1) for p in probs]
    })
    st.bar_chart(prob_df, x="Condition", y="Probability (%)")
