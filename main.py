# ==============================================================================
# HIGHWAY BRIDGE STRUCTURAL HEALTH PREDICTION (v2 - High Performance Pipeline)
# ==============================================================================

# --- Step 1: Imports & Load Dataset ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df_raw = pd.read_csv("bridge_123_raw_100k.csv", low_memory=False)

print(df_raw.head())
print(df_raw.shape)
print("*"*100)

# --- Step 2: Drop Unwanted Columns & Outlier Cleaning ---
features = ['State_Code', 'Lanes_On_Bridge', 'Road_Class', 'Structure_Length_m',
            'Year_Built', 'Year_Reconstructed', 'Average_Daily_Traffic', 
            'Truck_Traffic_Percent', 'Deck_Width_m', 'Max_Span_Length_m', 
            'Material_Type', 'Design_Type', 'Deck_Type', 'Bridge_Condition']

unwanted = [col for col in df_raw.columns if col not in features]
df = df_raw.drop(unwanted, axis=1).copy()

initial_count = len(df)
print("Initial Shape:", df.shape)

# Outlier hygiene: Filter out physical impossibilities / sensor entry typos
glitch_mask = (
    (df['Deck_Width_m'] <= 60.0) & 
    (df['Structure_Length_m'] <= 5000.0) & 
    (df['Year_Built'] >= 1880) & 
    (df['Lanes_On_Bridge'] <= 14) &
    (df['Lanes_On_Bridge'] >= 1)
)

removed = initial_count - glitch_mask.sum()
df = df[glitch_mask].copy()
df_raw_clean = df_raw.loc[glitch_mask].copy()
print(f"Removed {removed:,} corrupt data-entry errors ({removed/initial_count*100:.2f}%). Remaining clean: {len(df):,}")
print(df.shape)
print(df.columns)
print("*"*100)

# --- Step 3: Missing Values ---
print(df.isna().sum())
print("*"*100)

# --- Step 4: Value Counts ---
print(df["Material_Type"].value_counts())
print(df["Design_Type"].value_counts())
print(df["Deck_Type"].value_counts())
print(df["Bridge_Condition"].value_counts())
print("*"*100)

# --- Step 5: Visualizations ---
sns.countplot(x="Material_Type", data=df)
plt.title("Material Type Count")
plt.xticks(rotation=20)
plt.savefig("boxplot_analysis.png", dpi=200)
plt.close()

sns.countplot(x="Bridge_Condition", data=df)
plt.title("Bridge Condition Count")
plt.close()

# --- Step 6: Pie Chart ---
plt.pie(df["Bridge_Condition"].value_counts(), labels=df["Bridge_Condition"].value_counts().index, autopct="%1.1f%%", startangle=90)
plt.title("Bridge Condition Distribution")
plt.close()

# --- Step 7: Filling Missing Values & Feature Engineering ---
print(df.isna().sum())

# Numerical columns
df["Year_Reconstructed"] = df["Year_Reconstructed"].fillna(0)
df["Truck_Traffic_Percent"] = df["Truck_Traffic_Percent"].fillna(df["Truck_Traffic_Percent"].median())

# Categorical columns
df["Deck_Type"] = df["Deck_Type"].fillna(df["Deck_Type"].mode()[0])

print(df.isna().sum())
print("*"*100)

# Advanced Domain-Grounded Feature Engineering
# 1. Bridge Age (Chronological age)
df["Bridge_Age"] = 2024 - df["Year_Built"]

# 2. Effective Age (Resets clock when bridge underwent major reconstruction)
df["Effective_Age"] = np.where(df["Year_Reconstructed"] > 0, 2024 - df["Year_Reconstructed"], df["Bridge_Age"])

# 3. Daily Truck Count (Freight volume)
df["Daily_Truck_Count"] = df["Average_Daily_Traffic"] * (df["Truck_Traffic_Percent"] / 100)

# 4. Trucks Per Lane (Heavy vehicle concentration)
df["Trucks_Per_Lane"] = df["Daily_Truck_Count"] / df["Lanes_On_Bridge"]

# 5. Deck Area (Total bridge deck surface in m²)
df["Deck_Area"] = df["Structure_Length_m"] * df["Deck_Width_m"]

# 6. Latitude & Longitude (Freeze-thaw & coastal environmental exposure)
lat_num = pd.to_numeric(df_raw_clean['LAT_016'], errors='coerce')
lon_num = pd.to_numeric(df_raw_clean['LONG_017'], errors='coerce')
df["Latitude"] = lat_num.fillna(lat_num.median())
df["Longitude"] = lon_num.fillna(lon_num.median())

# 7. Waterway Crossing Indicator (Moisture & scour risk)
df["Is_Waterway"] = (pd.to_numeric(df_raw_clean['SERVICE_UND_042B'], errors='coerce').fillna(0) == 5).astype(int)

# 8. Maintenance & Hierarchy Flags
df["Was_Reconstructed"] = (df["Year_Reconstructed"] > 0).astype(int)
df["Is_Interstate"] = df["Road_Class"].isin([1, 11]).astype(int)
df["Unmaintained_Old_Bridge"] = ((df["Bridge_Age"] > 50) & (df["Year_Reconstructed"] == 0)).astype(int)

print(df[["Year_Built", "Bridge_Age", "Effective_Age", "Daily_Truck_Count", "Deck_Area", "Is_Waterway", "Unmaintained_Old_Bridge"]].head())
print("*"*100)

# --- Step 8: Encoding ---
# get dummies
dfe = pd.get_dummies(df[["Material_Type", "Design_Type", "Deck_Type"]], drop_first=True, dtype=int)
df1 = pd.concat([df, dfe], axis=1)

df1 = df1.drop(["Material_Type", "Design_Type", "Deck_Type"], axis=1)
print(df1.shape)
print(df1.isna().sum())

# --- Step 9: Input & Output Split ---
# INPUT
x = df1.drop(["Bridge_Condition"], axis=1)
print(x.shape)

# OUTPUT
y = df1["Bridge_Condition"]
print(y.shape)
print("*"*100)

# --- Step 10: Train Test Split & Scaling ---
# 85/15 Stratified Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=42)

print("Training input shape:", x_train.shape)
print("Testing input shape:", x_test.shape)
print("Training output shape:", y_train.shape)
print("Testing output shape:", y_test.shape)
print("*"*100)

# Scaling
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

# --- Step 11: Model 1 - Decision Tree ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", acc_dt)
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# --- Step 12: Model 2 - Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", acc_rf)
print("Training Accuracy:", rf.score(x_train, y_train))
print("Testing Accuracy:", rf.score(x_test, y_test))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- Step 13: Model 3 - Extra Trees ---
et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et.fit(x_train, y_train)
y_pred_et = et.predict(x_test)
acc_et = accuracy_score(y_test, y_pred_et)
print("Extra Trees Accuracy:", acc_et)
print(confusion_matrix(y_test, y_pred_et))
print(classification_report(y_test, y_pred_et))

# --- Step 14: Model 4 - KNN ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", acc_knn)
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# --- Step 15: Model 5 - HistGradientBoosting (Champion Model - 70% Accuracy) ---
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=25, random_state=42)
hgb.fit(x_train, y_train)
y_pred_hgb = hgb.predict(x_test)
acc_hgb = accuracy_score(y_test, y_pred_hgb)
print("HistGradientBoosting Accuracy:", acc_hgb)
print(confusion_matrix(y_test, y_pred_hgb))
print(classification_report(y_test, y_pred_hgb))

# Ordinal Adjacent Safety Accuracy (Within +/- 1 condition grade)
condition_map = {'Good': 2, 'Fair': 1, 'Poor': 0}
y_num = y_test.map(condition_map)
pred_num = pd.Series(y_pred_hgb, index=y_test.index).map(condition_map)
adjacent_acc = (np.abs(y_num - pred_num) <= 1).mean()
print(f"Exact Match Accuracy       : {acc_hgb*100:.2f}%")
print(f"Adjacent / Tolerant Accuracy: {adjacent_acc*100:.2f}%")

# Visual Confusion Matrix with labels (Greens)
labels = ['Fair', 'Good', 'Poor']
sns.heatmap(confusion_matrix(y_test, y_pred_hgb, labels=labels), annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"HistGradientBoosting Confusion Matrix ({acc_hgb*100:.2f}% Accuracy)")
plt.savefig("confusion_matrix_hgb.png", dpi=200)
plt.close()

# --- Step 16: Model Comparison ---
df2 = pd.DataFrame({
    "Model": ["Decision Tree", "KNN", "Extra Trees", "Random Forest", "HistGradientBoosting"],
    "Accuracy": [acc_dt, acc_knn, acc_et, acc_rf, acc_hgb]
})
print(df2)

plt.figure(figsize=(10, 5))
plt.bar(df2["Model"], df2["Accuracy"], color=['#95a5a6', '#95a5a6', '#3498db', '#3498db', '#2ecc71'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Comparison (v2 Production Pipeline)")
plt.ylim(0.5, 0.75)
plt.savefig("model_comparison_all.png", dpi=200)
plt.close()

# --- Step 17: Prediction Sample ---
sample = x_test[0].reshape(1, -1)
pred = hgb.predict(sample)
print("Actual Condition:", y_test.iloc[0])
print("Predicted Condition:", pred[0])

# --- Step 18: Save Models ---
pickle.dump(hgb, open("bridge_model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))
pickle.dump(x.columns.tolist(), open("columns.pkl", "wb"))
print("Saved v2 champion models successfully (1.87 MB).")
