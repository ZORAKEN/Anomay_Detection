import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Create sample data
X = np.array([
    [10, 20],
    [11, 21],
    [12, 19],
    [10, 22],
    [11, 20],
    [12, 21],
    [10, 19],
    [11, 22],
    [100, 200],
    [110, 210]
])

# 2. Create the Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination=0.2,
    random_state=42
)

# 3. Train the model and predict anomalies
predictions = model.fit_predict(X)

# 4. Separate normal points and anomalies
normal = X[predictions == 1]
anomalies = X[predictions == -1]

# 5. Print the results
print("Normal data points:")
print(normal)

print("\nAnomalies detected:")
print(anomalies)

# 6. Visualize the results
plt.scatter(
    normal[:, 0], normal[:, 1],
    label="Normal", marker="o"
)

plt.scatter(
    anomalies[:, 0], anomalies[:, 1],
    label="Anomaly", marker="x"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Anomaly Detection using Isolation Forest")
plt.legend()
plt.grid(True)
plt.show()
