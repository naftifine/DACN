
# traffic_feature_selection.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. LOAD DATA
# ======================

csv_path = "data_1107_edited.csv"
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

cols_to_exclude = [
    "segmentId","name_vn","geometry","timeStamp","time",
    "sunrise","sunset"
]

numeric_cols_for_clustering = [c for c in numeric_cols if c not in cols_to_exclude]

# ======================
# 2. CORRELATION ANALYSIS
# ======================

df_num = df[numeric_cols_for_clustering].copy()
df_num_imputed = df_num.fillna(df_num.median(numeric_only=True))
corr_matrix = df_num_imputed.corr(method="pearson")

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

threshold = 0.8
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        col_i = corr_matrix.columns[i]
        col_j = corr_matrix.columns[j]
        corr_ij = corr_matrix.iloc[i, j]
        if abs(corr_ij) >= threshold:
            corr_pairs.append((col_i, col_j, corr_ij))

corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
print("Highly correlated pairs (|corr|>=0.8):")
for a,b,v in corr_pairs_sorted:
    print(f"{a} <--> {b} corr={v:.3f}")

# ======================
# 3. PREPROCESSING PIPELINE
# ======================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols_for_clustering)
    ],
    remainder="drop"
)

X_processed = preprocess.fit_transform(df)

# Lấy đúng list feature SAU khi ColumnTransformer xử lý
feature_names_final = [
    name.split("__", 1)[1] 
    for name in preprocess.get_feature_names_out()
]

print("Processed shape:", X_processed.shape)
print("Features actually used:", feature_names_final)
print("Số lượng feature dùng để gom cụm:", len(feature_names_final))

# ======================
# 4. CLUSTERING EVALUATION
# ======================

def evaluate_clustering(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1]) if -1 in unique_labels else len(unique_labels)

    results = {
        "n_clusters": n_clusters,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None
    }

    if n_clusters <= 1:
        return results

    try: results["silhouette"] = silhouette_score(X, labels)
    except: pass

    try: results["davies_bouldin"] = davies_bouldin_score(X, labels)
    except: pass

    try: results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    except: pass

    return results

# ======================
# 5. DROP-ONE FEATURE IMPORTANCE
# ======================

def run_kmeans(X, n_clusters=4, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return evaluate_clustering(X, labels), labels

def run_dbscan(X, eps=0.5, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return evaluate_clustering(X, labels), labels

def feature_importance_drop_one(X, feature_names, cluster_func, **cluster_kwargs):
    base_scores, base_labels = cluster_func(X, **cluster_kwargs)
    base_sil = base_scores["silhouette"]
    base_db  = base_scores["davies_bouldin"]
    base_ch  = base_scores["calinski_harabasz"]

    print("Baseline:", base_scores)

    rows = []

    for i, f in enumerate(feature_names):
        X_drop = np.delete(X, i, axis=1)
        scores_drop, _ = cluster_func(X_drop, **cluster_kwargs)

        row = {
            "feature": f,
            "silhouette_drop": scores_drop["silhouette"],
            "davies_bouldin_drop": scores_drop["davies_bouldin"],
            "calinski_harabasz_drop": scores_drop["calinski_harabasz"],
            "delta_silhouette": None,
            "delta_davies_bouldin": None,
            "delta_calinski_harabasz": None
        }

        if base_sil and scores_drop["silhouette"]:
            row["delta_silhouette"] = scores_drop["silhouette"] - base_sil
        if base_db and scores_drop["davies_bouldin"]:
            row["delta_davies_bouldin"] = scores_drop["davies_bouldin"] - base_db
        if base_ch and scores_drop["calinski_harabasz"]:
            row["delta_calinski_harabasz"] = scores_drop["calinski_harabasz"] - base_ch

        rows.append(row)

    return base_scores, pd.DataFrame(rows)

# ======================
# 6. RUN FEATURE IMPORTANCE (K-MEANS)
# ======================

k = 4
base_scores_kmeans, df_imp_kmeans = feature_importance_drop_one(
    X_processed, 
    feature_names_final,
    cluster_func=run_kmeans,
    n_clusters=k
)

print(df_imp_kmeans.sort_values(by="delta_silhouette").head(20))

# ======================
# 7. RUN FEATURE IMPORTANCE (DBSCAN)
# ======================

eps = 0.5
min_samples = 20
base_scores_dbscan, df_imp_dbscan = feature_importance_drop_one(
    X_processed,
    feature_names_final,
    cluster_func=run_dbscan,
    eps=eps,
    min_samples=min_samples
)

print(df_imp_dbscan.sort_values(by="delta_silhouette").head(20))

# ======================
# 8. FINAL EXPERIMENTS: KMEANS SWEEP
# ======================
k_values = list(range(2, 9))
kmeans_results = []
for k in k_values:
    scores, _ = run_kmeans(X_processed, n_clusters=k)
    kmeans_results.append({
        "algorithm": "kmeans",
        "k": k,
        "n_clusters": scores["n_clusters"],
        "silhouette": scores["silhouette"],
        "davies_bouldin": scores["davies_bouldin"],
        "calinski_harabasz": scores["calinski_harabasz"],
    })
df_kmeans_exp = pd.DataFrame(kmeans_results)
print("\n=== KMEANS EXPERIMENT RESULTS ===")
print(df_kmeans_exp)

# ======================
# 9. FINAL EXPERIMENTS: DBSCAN SWEEP
# ======================
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
min_samples = 20
dbscan_results = []
for eps in eps_values:
    scores, _ = run_dbscan(X_processed, eps=eps, min_samples=min_samples)
    dbscan_results.append({
        "algorithm": "dbscan",
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": scores["n_clusters"],
        "silhouette": scores["silhouette"],
        "davies_bouldin": scores["davies_bouldin"],
        "calinski_harabasz": scores["calinski_harabasz"],
    })
df_dbscan_exp = pd.DataFrame(dbscan_results)
print("\n=== DBSCAN EXPERIMENT RESULTS ===")
print(df_dbscan_exp)

# ======================
# 10. MODEL OPTIMIZATION – KMEANS (ELBOW & METRIC PLOTS)
# ======================

# Elbow method using inertia
inertias = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_processed)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia (WCSS)")
plt.title("K-Means Elbow Method (Inertia vs k)")
plt.grid(True)
plt.show()

# Silhouette vs K
plt.figure()
plt.plot(df_kmeans_exp["k"], df_kmeans_exp["silhouette"], marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Silhouette score")
plt.title("K-Means – Silhouette vs k")
plt.grid(True)
plt.show()

# Calinski–Harabasz vs K
plt.figure()
plt.plot(df_kmeans_exp["k"], df_kmeans_exp["calinski_harabasz"], marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Calinski–Harabasz index")
plt.title("K-Means – Calinski–Harabasz vs k")
plt.grid(True)
plt.show()

# ======================
# 11. MODEL OPTIMIZATION – DBSCAN (K-DISTANCE & PARAM GRID)
# ======================

# K-distance graph to help choose eps
n_neighbors = 20
neighbors = NearestNeighbors(n_neighbors=n_neighbors)
neighbors_fit = neighbors.fit(X_processed)
distances, indices = neighbors_fit.kneighbors(X_processed)
# Use the farthest neighbor distance (last column)
k_distances = np.sort(distances[:, -1])

plt.figure()
plt.plot(k_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{n_neighbors}-NN distance")
plt.title("DBSCAN k-distance graph (use elbow to choose eps)")
plt.grid(True)
plt.show()

# Simple grid search over eps and min_samples
eps_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
min_samples_grid = [5, 10, 15, 20]

dbscan_grid_results = []
for eps in eps_grid:
    for ms in min_samples_grid:
        scores, _ = run_dbscan(X_processed, eps=eps, min_samples=ms)
        dbscan_grid_results.append({
            "eps": eps,
            "min_samples": ms,
            "n_clusters": scores["n_clusters"],
            "silhouette": scores["silhouette"],
            "davies_bouldin": scores["davies_bouldin"],
            "calinski_harabasz": scores["calinski_harabasz"],
        })

df_dbscan_grid = pd.DataFrame(dbscan_grid_results)
print("\n=== DBSCAN PARAM GRID RESULTS ===")
print(df_dbscan_grid)
