# ==============================================================================
# HIGHWAY BRIDGE STRUCTURAL HEALTH PREDICTION (v1 - Baseline Pipeline)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("bridge_123_raw_100k.csv")

print(df.head())
print(df.shape)
print("*"*100)

# --- Step 2: Drop Unwanted Columns ---
features = ['State_Code', 'Lanes_On_Bridge', 'Road_Class', 'Structure_Length_m',
            'Year_Built', 'Year_Reconstructed', 'Average_Daily_Traffic', 
            'Truck_Traffic_Percent', 'Deck_Width_m', 'Max_Span_Length_m', 
            'Material_Type', 'Design_Type', 'Deck_Type', 'Bridge_Condition']

unwanted = [col for col in df.columns if col not in features]
df.drop(unwanted, axis=1, inplace=True)

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
plt.show()

sns.countplot(x="Bridge_Condition", data=df)
plt.title("Bridge Condition Count")
plt.show()

# --- Step 6: Pie Chart ---
plt.pie(df["Bridge_Condition"].value_counts(), labels=df["Bridge_Condition"].value_counts().index, autopct="%1.1f%%", startangle=90)
plt.title("Bridge Condition Distribution")
plt.show()

# --- Step 7: Filling Missing Values & Feature Engineering ---
print(df.isna().sum())

# Numerical columns
df["Year_Reconstructed"] = df["Year_Reconstructed"].fillna(0)
df["Truck_Traffic_Percent"] = df["Truck_Traffic_Percent"].fillna(df["Truck_Traffic_Percent"].median())

# Categorical columns - Era-based (Decade) mode imputation for Deck_Type
df["Build_Decade"] = (df["Year_Built"] // 10) * 10
overall_deck_mode = df["Deck_Type"].mode()[0]
df["Deck_Type"] = df.groupby("Build_Decade")["Deck_Type"].transform(
    lambda group: group.fillna(group.mode()[0] if not group.mode().empty else overall_deck_mode)
)
df.drop(columns=["Build_Decade"], inplace=True)

print(df.isna().sum())
print("*"*100)

# Feature Engineering (for higher model accuracy)
# 1. Bridge Age
df["Bridge_Age"] = 2024 - df["Year_Built"]

# 2. Daily Truck Count
df["Daily_Truck_Count"] = df["Average_Daily_Traffic"] * (df["Truck_Traffic_Percent"] / 100)

print(df[["Year_Built", "Bridge_Age", "Average_Daily_Traffic", "Daily_Truck_Count"]].head())
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)

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

# training vs testing accuracy (check overfitting)
print("Training Accuracy:", rf.score(x_train, y_train))
print("Testing Accuracy:", rf.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Visual Confusion Matrix with labels
labels = ['Fair', 'Good', 'Poor']
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# --- Step 13: Feature Importance ---
feature_imp = rf.feature_importances_
feature_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_imp
})
feature_df = feature_df.sort_values(by='Importance', ascending=False)
print(feature_df.head(10))

# --- Step 14: Cross Validation ---
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, x_train, y_train, cv=5, n_jobs=-1)
print("Cross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# --- Step 15: Model 3 - KNN ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", acc_knn)
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# --- Step 16: Model Comparison ---
df2 = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "KNN"],
    "Accuracy": [acc_dt, acc_rf, acc_knn]
})
print(df2)

plt.bar(df2["Model"], df2["Accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Decision Tree vs Random Forest vs KNN")
plt.show()

# --- Step 17: Prediction Sample ---
sample = x_test[0].reshape(1, -1)
pred = rf.predict(sample)
print("Actual Condition:", y_test.iloc[0])
print("Predicted Condition:", pred[0])

# --- Step 18: Save Models ---
pickle.dump(rf, open("bridge_model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))
pickle.dump(x.columns.tolist(), open("columns.pkl", "wb"))
print("Saved v1 models successfully.")
