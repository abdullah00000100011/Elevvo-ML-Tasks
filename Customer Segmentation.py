import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.7)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("K-Means Clustering (k=5)")
plt.grid(True)
plt.show()

df["Cluster_KMeans"] = labels_kmeans
avg_spending = df.groupby("Cluster_KMeans")["Spending Score (1-100)"].mean()
print("Average Spending Score per Cluster (KMeans):")
print(avg_spending)

db = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = db.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_dbscan, cmap='rainbow', alpha=0.7)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("DBSCAN Clustering")
plt.grid(True)
plt.show()

df["Cluster_DBSCAN"] = labels_dbscan
if len(set(labels_dbscan)) > 1 and -1 in labels_dbscan:
    avg_spending_db = df[df["Cluster_DBSCAN"] != -1].groupby("Cluster_DBSCAN")["Spending Score (1-100)"].mean()
    print("Average Spending Score per Cluster (DBSCAN):")
    print(avg_spending_db)
elif len(set(labels_dbscan)) > 1:
    avg_spending_db = df.groupby("Cluster_DBSCAN")["Spending Score (1-100)"].mean()
    print("Average Spending Score per Cluster (DBSCAN):")
    print(avg_spending_db)
else:
    print("DBSCAN failed to form meaningful clusters.")

