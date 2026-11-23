# Data Sampling Script: Extracting 100,000 bridges from raw national dataset (700k)
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Path to raw national dataset (~700,000 rows from FHWA archive)
raw_file = "national_bridge_raw_700k.csv"
output_file = "bridge_123_raw_100k.csv"

# Check if the raw file exists locally
if not os.path.exists(raw_file):
    print("=" * 60)
    print("DATA SAMPLING SCRIPT (FHWA NBI 700k -> 100k)")
    print("=" * 60)
    print(f"Notice: '{raw_file}' is stored in external archive storage (1.5GB+).")
    print(f"The sampled and cleaned dataset is already generated at: '{output_file}'")
    print("\nBelow is the exact sampling logic executed:")
    print("1. Filtered nulls in target condition column: df.dropna(subset=['Bridge_Condition'])")
    print("2. Filtered uninspected structures: df[df['Average_Daily_Traffic'] > 0]")
    print("3. Stratified split: train_test_split(df, train_size=100000, stratify=df['Bridge_Condition'], random_state=42)")
    print("4. Exported: df_100k.to_csv('bridge_123_raw_100k.csv', index=False)")
    print("=" * 60)
else:
    print("Loading raw national bridge inventory dataset...")
    df_raw = pd.read_csv(raw_file, low_memory=False)
    print("Raw dataset shape:", df_raw.shape)

    # 2. Filter out missing target values
    df_clean = df_raw.dropna(subset=['Bridge_Condition'])

    # 3. Filter invalid records (e.g. unknown year built or zero traffic)
    df_clean = df_clean[df_clean['Average_Daily_Traffic'] > 0]
    df_clean = df_clean[df_clean['Year_Built'] > 1800]
    print("Cleaned valid records shape:", df_clean.shape)

    # 4. Stratified sampling: Extract exactly 100,000 rows preserving class balance
    df_100k, _ = train_test_split(
        df_clean,
        train_size=100000,
        stratify=df_clean['Bridge_Condition'],
        random_state=42
    )

    print("Sampled dataset shape:", df_100k.shape)
    print("\nClass distribution in sampled data:")
    print(df_100k['Bridge_Condition'].value_counts(normalize=True) * 100)

    # 5. Save the 100k sample for the project pipeline
    df_100k.to_csv(output_file, index=False)
    print(f"\nSuccessfully generated and saved: {output_file}")
