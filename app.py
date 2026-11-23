import pickle
from pathlib import Path
import numpy as np
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
presets = {
    "Custom Bridge (Enter your own values)": {
        "state": 29, "road": 2.0, "built": 1985, "recon": 0, "lanes": 2,
        "adt": 3500, "truck": 7.0, "len": 40.0, "span": 18.0, "width": 10.0,
        "lat": 38.5, "lon": -92.2, "waterway": True,
        "mat": "Steel", "des": "Stringer/Girder", "deck": "Concrete Cast-in-Place"
    },
    "Example 1: Modern Interstate Bridge (Prestressed Concrete) [Expected: Good]": {
        "state": 29, "road": 1.0, "built": 2018, "recon": 0, "lanes": 4,
        "adt": 45000, "truck": 12.0, "len": 120.0, "span": 35.0, "width": 18.0,
        "lat": 38.9, "lon": -92.3, "waterway": False,
        "mat": "Prestressed concrete", "des": "Box Beam", "deck": "Concrete Precast"
    },
    "Example 2: Typical State Highway Bridge (Steel Girder) [Expected: Fair]": {
        "state": 26, "road": 2.0, "built": 1975, "recon": 1995, "lanes": 2,
        "adt": 8500, "truck": 9.0, "len": 55.0, "span": 20.0, "width": 11.0,
        "lat": 44.3, "lon": -85.6, "waterway": True,
        "mat": "Steel", "des": "Stringer/Girder", "deck": "Concrete Cast-in-Place"
    },
    "Example 3: Aging Rural Bridge (1940 Steel Truss over Water) [Expected: Poor]": {
        "state": 29, "road": 9.0, "built": 1940, "recon": 0, "lanes": 1,
        "adt": 50, "truck": 5.0, "len": 30.0, "span": 15.0, "width": 5.5,
        "lat": 38.5, "lon": -92.2, "waterway": True,
        "mat": "Steel", "des": "Truss", "deck": "Timber"
    }
}

# Sidebar System Diagnostics
with st.sidebar:
    st.subheader("ℹ️ System Diagnostics")
    st.success("Active Model: `bridge_project_v2`")
    st.write(f"• **Artifact Path:** `{APP_DIR / 'bridge_model.pkl'}`")
    st.write(f"• **Architecture:** HistGradientBoosting")
    st.write(f"• **Test Accuracy:** 69.88% (Exact) / 98.80% (Adjacent)")
    st.write(f"• **Feature Pipeline:** 35 Domain Features")
    st.write(f"• **Model File Size:** 1.87 MB")
    st.divider()
    st.caption("Zero Target Leakage Pipeline • Strict Remote Inventory Screening")

# Dropdown to choose a preset
preset_choice = st.selectbox(
    "⚡ Quick Load Preset (Choose a sample bridge or customize below):",
    list(presets.keys())
)
data = presets[preset_choice]

st.divider()

# --------------------------------------------------
# 3. User Input Form (2 Columns) - Exact Original Layout
# --------------------------------------------------
col1, col2 = st.columns(2)

# --- Column 1: Location, Age & Traffic Load ---
with col1:
    st.subheader("📍 Location, Age & Traffic")
    
    # State code slider (1 - 78)
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
        index=road_classes.index(data["road"]) if data["road"] in road_classes else 0,
        format_func=lambda x: road_class_labels.get(x, str(x)),
        key=f"road_{preset_choice}"
    )

    # Age and reconstruction years (Sliders)
    year_built = st.slider(
        "Year Built",
        min_value=1880, max_value=2024,
        value=int(data["built"]),
        key=f"built_{preset_choice}"
    )
    year_reconstructed = st.slider(
        "Year Reconstructed (0 = never reconstructed)",
        min_value=0, max_value=2024,
        value=int(data["recon"]),
        key=f"recon_{preset_choice}"
    )
    lanes_on_bridge = st.slider(
        "Number of Traffic Lanes",
        min_value=1, max_value=12,
        value=int(data["lanes"]),
        key=f"lanes_{preset_choice}"
    )
    
    # Traffic volume & heavy truck percentage (Number input + Slider)
    average_daily_traffic = st.number_input(
        "Average Daily Traffic (vehicles/day)",
        min_value=0, max_value=500000,
        value=int(data["adt"]),
        step=500,
        key=f"adt_{preset_choice}"
    )
    truck_traffic_percent = st.slider(
        "Truck Traffic Percentage (%)",
        min_value=0.0, max_value=100.0,
        value=float(data["truck"]),
        step=0.5,
        key=f"truck_{preset_choice}"
    )

    # Environmental & Location Details
    with st.expander("🌐 Geographic Coordinates & Waterway Exposure", expanded=False):
        c_lat, c_lon = st.columns(2)
        with c_lat:
            latitude = st.number_input("Latitude", value=float(data.get("lat", 38.5)), format="%.3f", key=f"lat_{preset_choice}")
        with c_lon:
            longitude = st.number_input("Longitude", value=float(data.get("lon", -92.2)), format="%.3f", key=f"lon_{preset_choice}")
        is_waterway = st.checkbox("Bridge crosses over water / river (Scour & moisture risk)", value=bool(data.get("waterway", True)), key=f"water_{preset_choice}")

# --- Column 2: Dimensions & Construction Materials ---
with col2:
    st.subheader("🏗️ Structure & Materials")
    
    # Bridge geometry in meters
    structure_length_m = st.number_input(
        "Total Bridge Length (meters)",
        min_value=5.0, max_value=5000.0,
        value=float(data["len"]),
        step=5.0,
        key=f"len_{preset_choice}"
    )
    max_span_length_m = st.number_input(
        "Maximum Span Length (meters)",
        min_value=2.0, max_value=1000.0,
        value=float(data["span"]),
        step=2.0,
        key=f"span_{preset_choice}"
    )
    deck_width_m = st.slider(
        "Deck Width (meters)",
        min_value=3.0, max_value=40.0,
        value=float(data["width"]),
        step=0.5,
        key=f"width_{preset_choice}"
    )
    
    # Categorical material and structural design types
    materials = ['Concrete', 'Concrete continuous', 'Prestressed concrete', 'Prestressed concrete continuous', 'Steel', 'Steel continuous', 'Timber']
    material_type = st.selectbox(
        "Material Type",
        materials,
        index=materials.index(data["mat"]) if data["mat"] in materials else 0,
        key=f"mat_{preset_choice}"
    )
    
    designs = ['Box Beam', 'Girder Floorbeam', 'Slab', 'Stringer/Girder', 'Tee Beam', 'Truss']
    design_type = st.selectbox(
        "Design Type",
        designs,
        index=designs.index(data["des"]) if data["des"] in designs else 0,
        key=f"des_{preset_choice}"
    )
    
    decks = ['Concrete Cast-in-Place', 'Concrete Precast', 'Open Steel Grid', 'Timber']
    deck_type = st.selectbox(
        "Deck Type",
        decks,
        index=decks.index(data["deck"]) if data["deck"] in decks else 0,
        key=f"deck_{preset_choice}"
    )

st.divider()

# --------------------------------------------------
# 4. Model Prediction & Processing
# --------------------------------------------------
if st.button("🔍 Predict Bridge Health", use_container_width=True):
    # Load saved model artifacts
    with open(APP_DIR / "bridge_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(APP_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(APP_DIR / "columns.pkl", "rb") as f:
        columns = pickle.load(f)

    # Feature Engineering (Calculated automatically from inputs)
    bridge_age = 2024 - year_built
    effective_age = (2024 - year_reconstructed) if year_reconstructed > 0 else bridge_age
    daily_truck_count = average_daily_traffic * (truck_traffic_percent / 100.0)
    trucks_per_lane = daily_truck_count / max(lanes_on_bridge, 1)
    deck_area = structure_length_m * deck_width_m
    was_reconstructed = 1 if year_reconstructed > 0 else 0
    is_interstate = 1 if road_class in [1.0, 11.0] else 0
    unmaintained_old_bridge = 1 if (bridge_age > 50 and year_reconstructed == 0) else 0

    # Build input DataFrame
    input_data = {
        "State_Code": state_code,
        "Lanes_On_Bridge": lanes_on_bridge,
        "Road_Class": road_class,
        "Structure_Length_m": structure_length_m,
        "Year_Built": year_built,
        "Year_Reconstructed": year_reconstructed,
        "Average_Daily_Traffic": average_daily_traffic,
        "Truck_Traffic_Percent": truck_traffic_percent,
        "Deck_Width_m": deck_width_m,
        "Max_Span_Length_m": max_span_length_m,
        "Bridge_Age": bridge_age,
        "Effective_Age": effective_age,
        "Daily_Truck_Count": daily_truck_count,
        "Trucks_Per_Lane": trucks_per_lane,
        "Deck_Area": deck_area,
        "Latitude": latitude,
        "Longitude": longitude,
        "Is_Waterway": 1 if is_waterway else 0,
        "Was_Reconstructed": was_reconstructed,
        "Is_Interstate": is_interstate,
        "Unmaintained_Old_Bridge": unmaintained_old_bridge,
        "Material_Type": material_type,
        "Design_Type": design_type,
        "Deck_Type": deck_type
    }
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features and align to trained columns
    input_encoded = pd.get_dummies(input_df, columns=["Material_Type", "Design_Type", "Deck_Type"], drop_first=False, dtype=int)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale inputs using the fitted StandardScaler
    input_scaled = scaler.transform(input_encoded)

    # Model inference
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    # --------------------------------------------------
    # 5. Structural Health Assessment Report (Exact Original Report Layout)
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
        st.metric("Calculated Bridge Age", f"{bridge_age} Years", delta=f"{bridge_age - effective_age} yrs saved by recon" if year_reconstructed > 0 else f"Built in {year_built}", delta_color="off")

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
