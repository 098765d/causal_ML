
"""
It identifies potential mediators (M) between predictors (X) and the outcome (Y = G3, final grade).
The pipeline has three stages:
1. Feature selection for mediators using Random Forest on all features to predict Y.
2. For each selected mediator, identify drivers (X features) using Random Forest to predict the mediator.
3. Compute Spearman correlations for paths: X->M, M->Y, and direct X->Y.

Assumptions:
- Input DataFrame 'df' must contain 'G3' as the target column.
- All columns are convertible to numeric (ordinal/categorical treated as numeric).
- No one-hot encoding is performed; features are used as-is.

Key parameters:
- FREQ_THRESH: Threshold for feature frequency in top-k across repetitions (default 0.70).
- Repeated K-Fold: 5 splits, 20 repeats for stability.

"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from scipy.stats import spearmanr

# Assuming 'df' is already loaded. For completeness, you might add data loading here.
# Example: df = pd.read_csv('path_to_data.csv', sep=';')  # Uncomment and adjust if needed.

# Step 1: Validate and Prepare Data
# Ensure target column 'G3' exists and is numeric.
assert "G3" in df.columns, "DataFrame must contain 'G3' column as the target."

# Extract target y (final grade) and convert to float, handling errors.
y = pd.to_numeric(df["G3"], errors="coerce").astype(float)

# Extract features X (all columns except 'G3'), convert to numeric.
X = df.drop(columns=["G3"]).apply(pd.to_numeric, errors="coerce")

# Filter features with at least 3 non-NA values to avoid sparse or useless columns.
feats = [c for c in X.columns if X[c].notna().sum() >= 3]
X = X[feats].copy()  # Make a copy to avoid SettingWithCopyWarning.

# Total number of features after filtering.
p = len(feats)
print(f"Number of valid features after filtering: {p}")

# Define default mediator columns (M_cols) and predictor columns (X_cols).
# These are based on domain knowledge for student performance data.
M_cols_default = ["studytime", "failures", "paid", "activities", "higher", "reason", "traveltime", "romantic", "freetime", "goout", "Alc", "absences"]
X_cols_default = ["school", "sex", "address", "famsize", "Pstatus", "guardian", "internet", "nursery", "famrel", "health", "Pedu", "Mjob", "Fjob", "schoolsup", "famsup"]

# Filter to only include columns present in X.
M_cols = [c for c in M_cols_default if c in X.columns]
X_cols = [c for c in X_cols_default if c in X.columns]
print(f"Available mediators (M_cols): {M_cols}")
print(f"Available predictors (X_cols): {X_cols}")

# Step 2: Stage 1 - Feature Selection for Mediators Predicting Y
# Use Repeated K-Fold cross-validation for stability in feature importance.
rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)

# Determine top_k: at least 1, but ceiling of 10% of features.
top_k = max(1, int(np.ceil(0.10 * p)))
print(f"Top_k for feature selection: {top_k}")

# Initialize counters for feature frequency and list for all importances.
counts = pd.Series(0.0, index=feats)  # Frequency of being in top_k.
all_imp = []  # List to store importance Series from each run.

# Counter for runs (used as random_state for reproducibility).
run = 0

# Loop over folds and repeats.
for tr, _ in rkf.split(X):
    run += 1
    # Train Random Forest Regressor on training indices.
    rf = RandomForestRegressor(max_features="sqrt", n_jobs=-1, random_state=run).fit(X.iloc[tr], y.iloc[tr])
    
    # Get feature importances as Series.
    imp = pd.Series(rf.feature_importances_, index=feats)
    
    # Append to all_importances.
    all_imp.append(imp)
    
    # Increment count for the top_k most important features.
    counts.loc[imp.nlargest(top_k).index] += 1

# Compute frequency: count divided by total runs.
freq = counts / run

# Concatenate all importances into a DataFrame (rows=features, cols=runs).
imp_mat = pd.concat(all_imp, axis=1)

# Create Stage 1 DataFrame: features with freq_topk and mean_imp.
stage1 = (pd.DataFrame({
    "feature": feats,
    "freq_topk": freq.reindex(feats).values,
    "mean_imp": imp_mat.reindex(feats).mean(axis=1).values
})
.sort_values(["freq_topk", "mean_imp"], ascending=[False, False])
.reset_index(drop=True))

print("Stage 1 feature selection completed.")
print(stage1.head())  # Print top rows for debugging.

# Threshold for keeping features (frequency >= 0.70).
FREQ_THRESH = 0.70

# Select mediators to keep: those in M_cols that meet the threshold.
M_stage1_keep = [f for f in M_cols if f in stage1.loc[stage1.freq_topk >= FREQ_THRESH, "feature"].tolist()]
print(f"Mediators kept after Stage 1: {M_stage1_keep}")

# Step 3: Stage 2 - Identify Drivers (X features) for Each Kept Mediator
drivers = []  # List to collect Stage 2 DataFrames for each mediator.

for m in M_stage1_keep:
    print(f"Processing mediator: {m}")
    
    # Extract mediator values as numeric.
    my = pd.to_numeric(df[m], errors="coerce")
    
    # Subset X to only X_cols (predictors).
    X2 = X[X_cols]
    
    # Skip if no predictors available.
    if X2.shape[1] == 0:
        print(f"No predictors for {m}, skipping.")
        continue
    
    # Number of predictors.
    p2 = X2.shape[1]
    tk2 = max(1, int(np.ceil(0.10 * p2)))  # Top 10% for this subset.
    
    # Initialize counters and importances for this mediator.
    cnt2 = pd.Series(0.0, index=X2.columns)
    all_imp2 = []
    
    run = 0  # Reset run counter.
    
    for tr, _ in rkf.split(X2):
        run += 1
        rf = RandomForestRegressor(max_features="sqrt", n_jobs=-1, random_state=run).fit(X2.iloc[tr], my.iloc[tr])
        imp2 = pd.Series(rf.feature_importances_, index=X2.columns)
        all_imp2.append(imp2)
        cnt2.loc[imp2.nlargest(tk2).index] += 1
    
    # Create Stage 2 DataFrame for this mediator.
    stage2 = (pd.DataFrame({
        "x_feature": X2.columns,
        "freq_topk": (cnt2 / run).values,
        "mean_imp": pd.concat(all_imp2, axis=1).mean(axis=1).values
    })
    .sort_values(["freq_topk", "mean_imp"], ascending=[False, False])
    .reset_index(drop=True))
    
    stage2["mediator"] = m  # Add mediator column.
    drivers.append(stage2)

# Concatenate all Stage 2 DataFrames.
stage2_all = pd.concat(drivers, ignore_index=True) if drivers else pd.DataFrame(columns=["x_feature", "freq_topk", "mean_imp", "mediator"])

# Filter kept drivers based on threshold.
stage2_keep = stage2_all.query("freq_topk >= @FREQ_THRESH").reset_index(drop=True)
print("Stage 2 driver identification completed.")


# Step 4: Stage 3 - Compute Spearman Correlations for Paths
def spear(a, b):
    """
    Compute Spearman rank correlation between two series.
    
    Parameters:
    a, b: pd.Series or array-like, convertible to numeric.
    
    Returns:
    rho (float): Correlation coefficient.
    pval (float): p-value.
    n (int): Number of non-NA pairs used.
    """
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    r, p = spearmanr(a[mask], b[mask], nan_policy="omit")
    return float(r), float(p), int(mask.sum())

# List to collect correlation rows.
rows = []

# M -> Y paths: For each kept mediator to Y.
for m in M_stage1_keep:
    r, p, n = spear(df[m], y)
    rows.append(("M->Y", m, "G3", r, p, n))

# X -> M paths: For each kept driver-mediator pair.
for _, r2 in stage2_keep.iterrows():
    x = r2["x_feature"]
    m = r2["mediator"]
    r, p, n = spear(df[x], df[m])
    rows.append(("X->M", x, m, r, p, n))

# X -> Y paths: Direct from kept X features to Y (those meeting threshold in Stage 1).
direct_x_keep = [f for f in X_cols if f in stage1.loc[stage1.freq_topk >= FREQ_THRESH, "feature"].tolist()]
for x in direct_x_keep:
    r, p, n = spear(df[x], y)
    rows.append(("X->Y", x, "G3", r, p, n))

# Create Stage 3 DataFrame and sort.
stage3 = pd.DataFrame(rows, columns=["path", "src", "dst", "rho", "pval", "n"])
stage3 = stage3.sort_values(["path", "dst", "rho"], ascending=[True, True, False])
print("Stage 3 correlations computed.")

