
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# 1. CREATE A SAMPLE DATASET
# --------------------------------------------------

np.random.seed(42)

# Normal observations
normal_data = np.random.normal(
    loc=0,
    scale=1,
    size=(300, 2)
)

# Artificial anomalies
anomaly_data = np.random.uniform(
    low=4,
    high=6,
    size=(10, 2)
)

# Combine normal observations and anomalies
X = np.vstack([normal_data, anomaly_data])

df = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"]
)

# --------------------------------------------------
# 2. TRAIN ISOLATION FOREST
# --------------------------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(df)

# Prediction: 1 = normal, -1 = anomaly
df["Prediction"] = model.predict(df)

# Scikit-learn decision_function:
# Lower values indicate more anomalous observations
df["Anomaly_Score"] = model.decision_function(df[["Feature_1", "Feature_2"]])

print("Detected anomalies:")
print(df[df["Prediction"] == -1])

print("\nNumber of anomalies detected:",
      (df["Prediction"] == -1).sum())

# --------------------------------------------------
# 3. VISUALIZE ANOMALIES
# --------------------------------------------------

normal = df[df["Prediction"] == 1]
anomalies = df[df["Prediction"] == -1]

plt.figure(figsize=(8, 6))

plt.scatter(
    normal["Feature_1"],
    normal["Feature_2"],
    label="Normal",
    alpha=0.7
)

plt.scatter(
    anomalies["Feature_1"],
    anomalies["Feature_2"],
    label="Anomaly",
    marker="x",
    s=100
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Isolation Forest Anomaly Detection")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 4. EXPLAIN THE MODEL USING SHAP
# --------------------------------------------------

# Explain the decision_function output.
# Lower output means more anomalous.
background = shap.sample(df[["Feature_1", "Feature_2"]], 20,
                         random_state=42)

explainer = shap.Explainer(
    model.decision_function,
    background,
    algorithm="permutation"
)

# Explain a small sample to reduce computation
X_explain = df[["Feature_1", "Feature_2"]].sample(
    n=20,
    random_state=42
)

shap_values = explainer(
    X_explain,
    max_evals=5
)

# --------------------------------------------------
# 5. GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. SHAP SUMMARY PLOT
# --------------------------------------------------

shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. EXPLAIN ONE ANOMALY
# --------------------------------------------------

anomaly_indices = df.index[df["Prediction"] == -1]

if len(anomaly_indices) > 0:

    # Select the most anomalous observation
    idx = df.loc[anomaly_indices, "Anomaly_Score"].idxmin()

    # Explain that observation
    local_explanation = explainer(
        df.loc[[idx], ["Feature_1", "Feature_2"]],
        max_evals=5
    )

    shap.plots.waterfall(local_explanation[0], show=False)
    plt.title(f"Explanation for Anomaly {idx}")
    plt.tight_layout()
    plt.show()

    print("\nMost anomalous observation:")
    print(df.loc[idx])

print("\nAnomaly detection and SHAP explanation completed.")
