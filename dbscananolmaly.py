"""from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels = model.fit_predict(X)

print(labels)"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Create the DBSCAN model
model = DBSCAN(
    eps=0.8,
    min_samples=3
)

# 4. Fit the model and predict clusters
labels = model.fit_predict(X_scaled)

# 5. Identify normal points and anomalies
anomalies = X[labels == -1]
normal = X[labels != -1]

# 6. Print the results
print("Cluster labels:", labels)

print("\nNormal data points:")
print(normal)

print("\nAnomalies detected:")
print(anomalies)

# 7. Visualize the results
plt.scatter(
    normal[:, 0],
    normal[:, 1],
    label="Normal",
    marker="o"
)

plt.scatter(
    anomalies[:, 0],
    anomalies[:, 1],
    label="Anomaly (Noise)",
    marker="x"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Anomaly Detection using DBSCAN")
plt.legend()
plt.grid(True)
plt.show()
