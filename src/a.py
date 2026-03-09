# traffic_feature_selection_only.py
#
# Mục tiêu file này:
# 1) Đọc dữ liệu giao thông + thời tiết HCM.
# 2) Tiền xử lý feature numeric (bỏ ID/time, NaN-only, low-variance, high-corr).
# 3) Tạo nhãn traffic_label A–E từ currentSpeed (dùng làm y tham chiếu).
# 4) Feature Selection với NHIỀU kỹ thuật:
#    - RandomForest, DecisionTree, Mutual Information (MI, đã chuẩn hoá về [0,1])
#    - ExtraTrees, GradientBoosting, L1-LogisticRegression, (tùy chọn: XGBoost nếu có)
# 5) In:
#    - Thứ hạng feature của từng kỹ thuật (full ranking).
#    - TOP_K feature của từng kỹ thuật.
#    - So sánh xem TOP_K giữa các kỹ thuật giống / khác nhau thế nào.
# 6) Chạy pipeline clustering cho từng bộ feature được chọn (tất cả FS methods).

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif

# Optional XGBoost import
try:
    from xgboost import XGBClassifier  # optional

    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

import os
import warnings
import ast

import matplotlib

matplotlib.use("Agg")  # headless save-to-file
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA


# =============================================================
# FIGURE OUTPUT CONFIG
# =============================================================
FIG_DIR = os.path.join("a", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _savefig(path, dpi=220):
    """Save current matplotlib figure safely."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# =============================================================
# 0b. FIGURE HELPERS (for report)
# =============================================================


def plot_traffic_label_distribution(df, out_path):
    """Bar chart of traffic_label counts (A..E)."""
    if "traffic_label" not in df.columns:
        return
    vc = df["traffic_label"].value_counts().sort_index()
    plt.figure()
    ax = vc.plot(kind="bar")
    ax.set_xlabel("traffic_label (A=slow ... E=fast)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of traffic_label")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _savefig(out_path)


def plot_missingness_bar(df, cols, top_n, out_path):
    """Plot top-N columns by missing ratio."""
    miss = df[cols].isna().mean().sort_values(ascending=False)
    miss = miss.head(top_n)
    plt.figure(figsize=(10, 4.5))
    ax = miss.plot(kind="bar")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing ratio")
    ax.set_title(f"Top-{top_n} features by missing ratio")
    _savefig(out_path)


def plot_corr_heatmap(df, cols, out_path, max_features=25):
    """Correlation heatmap for a subset of numeric features (for readability)."""
    use = cols[:]
    if len(use) > max_features:
        # pick features with highest variance for a more informative small heatmap
        tmp = df[use].copy()
        tmp = tmp.fillna(tmp.median(numeric_only=True))
        vari = tmp.var().sort_values(ascending=False)
        use = vari.head(max_features).index.tolist()
    tmp = df[use].copy()
    tmp = tmp.fillna(tmp.median(numeric_only=True))
    corr = tmp.corr()

    plt.figure(figsize=(9, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="corr")
    plt.xticks(range(len(use)), use, rotation=90, fontsize=7)
    plt.yticks(range(len(use)), use, fontsize=7)
    plt.title("Correlation heatmap (subset)")
    _savefig(out_path)


def plot_feature_ranking(df_rank, score_col, title, out_path, top_n=15):
    """Horizontal bar plot of top-N feature ranking."""
    d = df_rank.head(top_n).copy()
    d = d.iloc[::-1]  # reverse for nicer top-at-top
    plt.figure(figsize=(9, 5.5))
    plt.barh(d["feature"].astype(str), d[score_col].astype(float))
    plt.xlabel(score_col)
    plt.ylabel("feature")
    plt.title(title)
    _savefig(out_path)


def plot_kmeans_silhouette_curve(results_rows, out_path, title_prefix="KMeans"):
    """Plot silhouette vs k for KMeans across n_init; expects rows from metrics list."""
    rows = [
        r
        for r in results_rows
        if r.get("algorithm") == "KMeans" and r.get("silhouette") is not None
    ]
    if not rows:
        return
    # parse params string like "{'k': 15, 'n_init': 10}"
    parsed = []
    for r in rows:
        try:
            p = (
                ast.literal_eval(r.get("params"))
                if isinstance(r.get("params"), str)
                else r.get("params")
            )
            parsed.append(
                (int(p.get("k")), int(p.get("n_init", 0)), float(r.get("silhouette")))
            )
        except Exception:
            continue
    if not parsed:
        return

    plt.figure(figsize=(8.5, 4.5))
    for n_init in sorted({t[1] for t in parsed}):
        xs = [t[0] for t in parsed if t[1] == n_init]
        ys = [t[2] for t in parsed if t[1] == n_init]
        if xs:
            plt.plot(xs, ys, marker="o", label=f"n_init={n_init}")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title(f"{title_prefix}: Silhouette vs k")
    plt.legend()
    _savefig(out_path)


def plot_dbscan_silhouette_heatmap(results_rows, out_path, title_prefix="DBSCAN"):
    """Heatmap of silhouette over (eps, min_samples) for DBSCAN."""
    rows = [
        r
        for r in results_rows
        if r.get("algorithm") == "DBSCAN" and r.get("silhouette") is not None
    ]
    if not rows:
        return
    parsed = []
    for r in rows:
        try:
            p = (
                ast.literal_eval(r.get("params"))
                if isinstance(r.get("params"), str)
                else r.get("params")
            )
            parsed.append(
                (
                    float(p.get("eps")),
                    int(p.get("min_samples")),
                    float(r.get("silhouette")),
                )
            )
        except Exception:
            continue
    if not parsed:
        return

    eps_vals = sorted({t[0] for t in parsed})
    ms_vals = sorted({t[1] for t in parsed})
    grid = np.full((len(ms_vals), len(eps_vals)), np.nan)
    for eps, ms, sil in parsed:
        i = ms_vals.index(ms)
        j = eps_vals.index(eps)
        grid[i, j] = sil

    plt.figure(figsize=(9, 4.5))
    plt.imshow(grid, aspect="auto")
    plt.colorbar(label="Silhouette")
    plt.xticks(range(len(eps_vals)), [str(e) for e in eps_vals])
    plt.yticks(range(len(ms_vals)), [str(m) for m in ms_vals])
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.title(f"{title_prefix}: Silhouette heatmap")
    _savefig(out_path)


def plot_cluster_size_bar(labels, out_path, title="Cluster size distribution"):
    """Bar plot of cluster sizes (excluding -1 noise if present)."""
    s = pd.Series(labels)
    s = s[s != -1]
    vc = s.value_counts().sort_index()
    plt.figure(figsize=(9, 4.2))
    ax = vc.plot(kind="bar")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _savefig(out_path)


def plot_pca_scatter(X, labels, out_path, title="PCA 2D projection"):
    """2D PCA scatter for visualization (not for evaluation)."""
    try:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
    except Exception:
        return
    plt.figure(figsize=(7, 5.5))
    plt.scatter(X2[:, 0], X2[:, 1], s=10, c=pd.Series(labels).astype(int), alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    _savefig(out_path)


# === Added: Histogram, feature count, model selection scatter/table helpers ===
def plot_histogram(series, out_path, title, xlabel):
    s = pd.Series(series).dropna()
    if s.empty:
        return
    plt.figure(figsize=(7.5, 4.5))
    plt.hist(s.values, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    _savefig(out_path)


def plot_feature_count_pipeline(
    step_counts, out_path, title="Feature/column counts by pipeline step"
):
    # step_counts: list of (step_name, count)
    if not step_counts:
        return
    names = [s[0] for s in step_counts]
    vals = [int(s[1]) for s in step_counts]
    plt.figure(figsize=(9.5, 4.5))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=20, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    for i, v in enumerate(vals):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    _savefig(out_path)


def plot_model_selection_scatter(
    df_models, out_path, title="Model selection: ARI vs Silhouette"
):
    # Scatter to show tradeoff and highlight best_overall.
    use = df_models.copy()
    for c in ["silhouette", "ARI", "weighted_score", "gate_pass", "pareto_front"]:
        if c not in use.columns:
            return
    use = use[use["silhouette"].notna() & use["ARI"].notna()].copy()
    if use.empty:
        return
    plt.figure(figsize=(7.2, 5.4))
    # Color by weighted_score for readability
    plt.scatter(
        use["silhouette"],
        use["ARI"],
        s=14,
        c=use["weighted_score"].astype(float),
        alpha=0.75,
    )
    plt.colorbar(label="weighted_score")
    # Emphasize pareto front + gate pass
    pf = use[(use["pareto_front"] == True) & (use["gate_pass"] == True)]
    if not pf.empty:
        plt.scatter(
            pf["silhouette"],
            pf["ARI"],
            s=35,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
        )
    plt.xlabel("Silhouette (internal)")
    plt.ylabel("ARI (external)")
    plt.title(title)
    _savefig(out_path)


def plot_model_selection_table(
    best_row, out_path, title="Best model (metrics + balance)"
):
    # Render a compact table as a figure (PNG) for inserting into report.
    keys = [
        ("fs_method", "FS method"),
        ("family", "Family"),
        ("algorithm", "Algorithm"),
        ("params", "Params"),
        ("silhouette", "Silhouette"),
        ("davies_bouldin", "Davies-Bouldin"),
        ("calinski_harabasz", "Calinski-Harabasz"),
        ("ARI", "ARI"),
        ("NMI", "NMI"),
        ("accuracy_majority", "Acc (majority)"),
        ("MAE", "MAE"),
        ("RMSE", "RMSE"),
        ("n_clusters", "#clusters"),
        ("noise_ratio", "noise_ratio"),
        ("largest_cluster_ratio", "largest_cluster_ratio"),
        ("min_cluster_size", "min_cluster_size"),
        ("largest_cluster_size", "largest_cluster_size"),
        ("weighted_score", "weighted_score"),
        ("pareto_front", "pareto_front"),
        ("gate_reason", "gate_reason"),
    ]

    rows = []
    for k, label in keys:
        v = best_row.get(k, "")
        if isinstance(v, float):
            v = f"{v:.6f}"
        rows.append([label, str(v)])

    plt.figure(figsize=(10.5, 7.0))
    plt.axis("off")
    plt.title(title, pad=12)
    tbl = plt.table(
        cellText=rows, colLabels=["Field", "Value"], loc="center", cellLoc="left"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.25)
    _savefig(out_path)


# =============================================================
# 0c. Helpers to refit and plot best-overall model (figures)
# =============================================================


def _build_X_for_fs(X_all, feature_names_all, fs_feature_sets, fs_method):
    """Return (X_selected, used_features) for a given FS method name."""
    if fs_method not in fs_feature_sets:
        raise ValueError(f"Unknown fs_method={fs_method}")
    selected_features = fs_feature_sets[fs_method]
    indices = [
        feature_names_all.index(f) for f in selected_features if f in feature_names_all
    ]
    if not indices:
        raise ValueError(f"FS={fs_method}: no selected features found in X_all")
    X_sel = X_all[:, indices]
    used = [feature_names_all[i] for i in indices]
    return X_sel, used


def _fit_predict_from_row(best_row, X_all, feature_names_all, fs_feature_sets):
    """Refit the clustering model described by best_row and return labels + X used."""
    fs_method = str(best_row.get("fs_method"))
    algo = str(best_row.get("algorithm"))
    params_raw = best_row.get("params")
    params = (
        ast.literal_eval(params_raw)
        if isinstance(params_raw, str)
        else (params_raw or {})
    )

    X_sel, used_features = _build_X_for_fs(
        X_all, feature_names_all, fs_feature_sets, fs_method
    )

    # Build estimator based on algorithm
    if algo == "KMeans":
        est = KMeans(
            n_clusters=int(params.get("k")),
            n_init=int(params.get("n_init", 10)),
            random_state=42,
        )
        labels = est.fit_predict(X_sel)
    elif algo == "MiniBatchKMeans":
        est = MiniBatchKMeans(
            n_clusters=int(params.get("k")), random_state=42, batch_size=256
        )
        labels = est.fit_predict(X_sel)
    elif algo.startswith("Agglomerative_"):
        # algo format: Agglomerative_single/complete/average/ward
        link = algo.split("_", 1)[1]
        k = int(params.get("k"))
        if link == "ward":
            est = AgglomerativeClustering(n_clusters=k, linkage=link)
        else:
            est = AgglomerativeClustering(
                n_clusters=k, linkage=link, metric="euclidean"
            )
        labels = est.fit_predict(X_sel)
    elif algo == "DBSCAN":
        est = DBSCAN(
            eps=float(params.get("eps")), min_samples=int(params.get("min_samples"))
        )
        labels = est.fit_predict(X_sel)
    elif algo == "OPTICS":
        est = OPTICS(
            min_samples=int(params.get("min_samples")), xi=float(params.get("xi"))
        )
        labels = est.fit_predict(X_sel)
    elif algo == "GMM":
        est = GaussianMixture(
            n_components=int(params.get("n_components")),
            covariance_type=str(params.get("covariance_type")),
            random_state=42,
        )
        labels = est.fit_predict(X_sel)
    else:
        raise ValueError(f"Unsupported algorithm for refit: {algo}")

    return labels, X_sel, used_features


def plot_best_overall_figures(
    best_row, X_all, feature_names_all, fs_feature_sets, fig_dir
):
    """Create ONLY the 2 main best-overall figures required for the report."""
    try:
        labels, X_sel, used_features = _fit_predict_from_row(
            best_row, X_all, feature_names_all, fs_feature_sets
        )
    except Exception as e:
        print(f"[WARN] Không thể refit best overall để vẽ figure: {e}")
        return

    fs_method = str(best_row.get("fs_method"))
    algo = str(best_row.get("algorithm"))

    # 1) Cluster size distribution (best overall)
    plot_cluster_size_bar(
        labels,
        out_path=os.path.join(fig_dir, "fig_best_overall_cluster_sizes.png"),
        title=f"Best overall cluster sizes (FS={fs_method}, Algo={algo})",
    )

    # 2) PCA scatter (best overall)
    plot_pca_scatter(
        X_sel,
        labels,
        out_path=os.path.join(fig_dir, "fig_best_overall_pca_scatter.png"),
        title=f"Best overall PCA 2D (FS={fs_method}, Algo={algo})",
    )

    print("[INFO] Đã vẽ 2 figure chính cho BEST OVERALL model:")
    print(" - fig_best_overall_cluster_sizes.png")
    print(" - fig_best_overall_pca_scatter.png")


# =============================================================
# 0. HÀM TIỆN ÍCH
# =============================================================


def create_traffic_label(df, speed_col="currentSpeed"):
    """
    Tạo nhãn A..E từ currentSpeed bằng qcut (5 quantile).
    - A: tốc độ thấp nhất (kẹt/đông)
    - E: tốc độ cao nhất (thoáng)
    Chỉ dùng để làm nhãn tham chiếu cho Feature Selection.
    """
    if speed_col not in df.columns:
        print(f"[INFO] Không tìm thấy cột {speed_col}, bỏ qua tạo traffic_label.")
        return df.copy(), None, None

    df2 = df.copy()
    speed = df2[speed_col]
    mask = speed.notna()

    try:
        labels = pd.qcut(speed[mask], q=5, labels=list("ABCDE"))
    except ValueError as e:
        print("[WARN] Không tạo được traffic_label do không đủ giá trị khác nhau:", e)
        return df2, None, None

    df2["traffic_label"] = None
    df2.loc[mask, "traffic_label"] = labels.astype(str)

    label_series = df2["traffic_label"].astype(str)
    label_order = sorted(label_series.dropna().unique().tolist())
    label_to_int = {lab: i for i, lab in enumerate(label_order)}
    y_int = label_series.map(label_to_int).values

    return df2, y_int, label_to_int


def drop_low_variance_and_high_corr(df, cols, var_threshold=1e-6, corr_threshold=0.9):
    """
    Bỏ cột low-variance và high-correlation (unsupervised FS bước 1).
    In log chi tiết để xem rõ từng cột bị loại.
    """
    print("\n[STEP] Phân tích variance & correlation để chọn feature (unsupervised FS)")
    print("- Danh sách feature numeric ban đầu:", cols)

    df_num = df[cols].copy()
    df_num_imputed = df_num.fillna(df_num.median(numeric_only=True))

    # 1) Low variance
    var = df_num_imputed.var()
    low_var_cols = var[var <= var_threshold]
    if not low_var_cols.empty:
        print("- Các cột bị loại do variance quá thấp (≈hằng số):")
        for c, v in low_var_cols.items():
            print(f"    {c}: var={v:.6e}")
    else:
        print("- Không có cột nào bị loại vì low-variance.")

    keep = var[var > var_threshold].index.tolist()

    # 2) High correlation
    corr = df_num_imputed[keep].corr().abs()
    print(f"- Ma trận tương quan kích thước: {corr.shape}")

    keep_final = []
    dropped_corr = []

    for col in corr.columns:
        if not keep_final:
            keep_final.append(col)
            continue

        high_corr_with = [k for k in keep_final if corr.loc[col, k] >= corr_threshold]
        if high_corr_with:
            dropped_corr.append((col, high_corr_with))
        else:
            keep_final.append(col)

    if dropped_corr:
        print(f"- Các cột bị loại do tương quan cao (>|{corr_threshold}|):")
        for col, lst in dropped_corr:
            partners = ", ".join(lst)
            vals = ", ".join([f"{p}={corr.loc[col, p]:.3f}" for p in lst])
            print(f"    {col} ~ {partners} (corr: {vals})")
    else:
        print("- Không có cột nào bị loại vì high-correlation.")

    print("- Số feature sau khi lọc variance & corr:", len(keep_final))
    print("- Feature giữ lại:", keep_final)

    return keep_final


# =============================================================
# Helper: Model-based feature ranking
# =============================================================
def rank_features_by_model(
    model,
    X,
    y,
    feature_names,
    importance_attr="feature_importances_",
    score_name="importance",
):
    """Train a supervised model and return a sorted DataFrame of feature importances.

    Notes
    -----
    - This is *supervised* feature ranking using the reference label traffic_label (A..E -> 0..4).
    - We DO NOT claim this is the only correct FS for clustering; we use it to generate multiple
      candidate feature sets and then test downstream clustering across many algorithms.
    """
    model.fit(X, y)

    if hasattr(model, importance_attr):
        imp = getattr(model, importance_attr)
        imp = np.asarray(imp).ravel()
    else:
        raise ValueError(f"Model does not have attribute {importance_attr}")

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            score_name: imp,
        }
    ).sort_values(score_name, ascending=False)

    return df_imp


# =============================================================
# INTERNAL/EXTERNAL CLUSTERING EVALUATION HELPERS
# =============================================================
def evaluate_clustering_internal(X, labels):
    """
    Đánh giá NỘI BỘ (internal) cho một kết quả gom cụm:
    - n_clusters (bỏ nhãn -1 nếu có)
    - silhouette
    - davies_bouldin
    - calinski_harabasz
    """
    unique_labels = np.unique(labels)
    n_clusters = (
        len(unique_labels[unique_labels != -1])
        if -1 in unique_labels
        else len(unique_labels)
    )

    results = {
        "n_clusters": n_clusters,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if n_clusters <= 1:
        return results

    try:
        results["silhouette"] = silhouette_score(X, labels)
    except Exception:
        pass

    try:
        results["davies_bouldin"] = davies_bouldin_score(X, labels)
    except Exception:
        pass

    try:
        results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    except Exception:
        pass

    return results


def evaluate_clustering_external(y_true_int, labels):
    """
    Đánh giá NGOẠI (external) so với nhãn tham chiếu y_true_int (0..4 tương ứng A..E):
    - ARI
    - NMI
    - accuracy_majority: accuracy sau khi map mỗi cụm -> lớp chiếm đa số
    - MAE / RMSE: sai số tuyệt đối / bình phương giữa nhãn thật và nhãn dự đoán (sau majority vote).
    """
    results = {
        "ARI": None,
        "NMI": None,
        "accuracy_majority": None,
        "MAE": None,
        "RMSE": None,
    }

    # Nếu tất cả điểm bị gán 1 cụm hoặc noise
    unique_labels = np.unique(labels)
    n_clusters = (
        len(unique_labels[unique_labels != -1])
        if -1 in unique_labels
        else len(unique_labels)
    )
    if n_clusters <= 1:
        return results

    # ARI, NMI
    try:
        results["ARI"] = adjusted_rand_score(y_true_int, labels)
    except Exception:
        pass

    try:
        results["NMI"] = normalized_mutual_info_score(y_true_int, labels)
    except Exception:
        pass

    # Majority-vote mapping cluster -> class
    try:
        labels_arr = np.asarray(labels)
        y_arr = np.asarray(y_true_int)

        mask = labels_arr != -1
        labels_use = labels_arr[mask]
        y_use = y_arr[mask]

        cluster_to_class = {}
        for c in np.unique(labels_use):
            idx = labels_use == c
            if idx.sum() == 0:
                continue
            vals, counts = np.unique(y_use[idx], return_counts=True)
            majority_class = vals[np.argmax(counts)]
            cluster_to_class[c] = majority_class

        y_pred_majority = np.array(
            [cluster_to_class.get(lbl, -1) for lbl in labels_use]
        )

        valid_mask = y_pred_majority != -1
        if valid_mask.sum() > 0:
            acc = accuracy_score(y_use[valid_mask], y_pred_majority[valid_mask])
            results["accuracy_majority"] = acc

            # MAE & RMSE trên nhãn số 0..4
            mae = mean_absolute_error(y_use[valid_mask], y_pred_majority[valid_mask])
            rmse = np.sqrt(
                mean_squared_error(y_use[valid_mask], y_pred_majority[valid_mask])
            )
            results["MAE"] = mae
            results["RMSE"] = rmse
    except Exception:
        pass

    return results


# =============================================================
# MODEL SELECTION HELPERS: degenerate filter + scoring + pareto
# =============================================================
def _cluster_size_stats(labels):
    """Return cluster size stats used for degenerate filtering."""
    labels_arr = np.asarray(labels)
    n_total = labels_arr.shape[0]
    noise_count = int((labels_arr == -1).sum())
    noise_ratio = noise_count / n_total if n_total > 0 else 1.0

    valid = labels_arr[labels_arr != -1]
    if valid.size == 0:
        return {
            "n_points": n_total,
            "noise_ratio": noise_ratio,
            "largest_cluster_ratio": 1.0,
            "min_cluster_size": 0,
            "largest_cluster_size": 0,
        }

    _, counts = np.unique(valid, return_counts=True)
    largest = int(counts.max()) if counts.size else 0
    smallest = int(counts.min()) if counts.size else 0
    largest_ratio = largest / n_total if n_total > 0 else 1.0
    return {
        "n_points": n_total,
        "noise_ratio": noise_ratio,
        "largest_cluster_ratio": largest_ratio,
        "min_cluster_size": smallest,
        "largest_cluster_size": largest,
    }


def _degenerate_reason(row, gate_cfg):
    """Return a human-readable reason if the clustering result is degenerate."""
    reasons = []
    n_clusters = row.get("n_clusters", None)
    if n_clusters is None or (
        isinstance(n_clusters, (int, np.integer)) and n_clusters <= 1
    ):
        reasons.append("n_clusters<=1")

    lcr = row.get("largest_cluster_ratio", None)
    if lcr is not None and lcr > gate_cfg["largest_cluster_ratio_max"]:
        reasons.append(f"largest_cluster_ratio>{gate_cfg['largest_cluster_ratio_max']}")

    mcs = row.get("min_cluster_size", None)
    if mcs is not None and mcs < gate_cfg["min_cluster_size_min"]:
        reasons.append(f"min_cluster_size<{gate_cfg['min_cluster_size_min']}")

    nr = row.get("noise_ratio", None)
    if nr is not None and nr > gate_cfg["noise_ratio_max"]:
        reasons.append(f"noise_ratio>{gate_cfg['noise_ratio_max']}")

    return "; ".join(reasons) if reasons else ""


def _minmax_norm(series, higher_is_better=True):
    """Min-max normalize a pandas Series to [0,1]. NaN stays NaN."""
    s = series.astype(float)
    vmin = np.nanmin(s.values) if np.isfinite(s.values).any() else np.nan
    vmax = np.nanmax(s.values) if np.isfinite(s.values).any() else np.nan
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return pd.Series(np.nan, index=series.index)
    norm = (s - vmin) / (vmax - vmin)
    if higher_is_better:
        return norm
    return 1.0 - norm


def pareto_front(df, objectives):
    """Compute Pareto front for mixed objectives.

    objectives: list of tuples (col, direction)
      - direction: 'max' or 'min'
    Returns boolean Series True if nondominated.
    """
    mask = pd.Series(True, index=df.index)
    for col, _ in objectives:
        mask &= df[col].notna()
    cand = df.loc[mask].copy()
    if cand.empty:
        return pd.Series(False, index=df.index)

    vals = cand[[c for c, _ in objectives]].to_numpy(dtype=float)
    for j, (_, direction) in enumerate(objectives):
        if direction == "min":
            vals[:, j] = -vals[:, j]

    is_nd = np.ones(vals.shape[0], dtype=bool)
    for i in range(vals.shape[0]):
        if not is_nd[i]:
            continue
        diff = vals - vals[i]
        ge_all = (diff >= 0).all(axis=1)
        gt_any = (diff > 0).any(axis=1)
        dominators = ge_all & gt_any
        dominators[i] = False
        if dominators.any():
            is_nd[i] = False

    out = pd.Series(False, index=df.index)
    out.loc[cand.index] = is_nd
    return out


# =============================================================
# 1. LOAD DATA + TẠO NHÃN A–E
# =============================================================

csv_path = "traffic_weather_hcm_1208-1209.csv"  # chỉnh lại nếu cần
print(f"[INFO] Input CSV: {os.path.abspath(csv_path)}")

df_raw = pd.read_csv(csv_path)
print("Shape raw:", df_raw.shape)
# ===== Figures: raw dataset overview =====
# 1) currentSpeed distribution (raw)
if "currentSpeed" in df_raw.columns:
    plot_histogram(
        df_raw["currentSpeed"],
        out_path=os.path.join(FIG_DIR, "fig00_raw_currentSpeed_hist.png"),
        title="Raw currentSpeed distribution",
        xlabel="currentSpeed",
    )

df, y_int, label_to_int = create_traffic_label(df_raw, speed_col="currentSpeed")

if y_int is not None:
    print("Đã tạo traffic_label A–E để dùng cho Feature Selection.")
    print("Ánh xạ nhãn -> số:", label_to_int)
    print("Phân bố traffic_label:")
    print(df["traffic_label"].value_counts().sort_index())
    # 2) traffic_label distribution (A..E)
    plot_traffic_label_distribution(
        df, os.path.join(FIG_DIR, "fig01_traffic_label_distribution.png")
    )
else:
    raise RuntimeError("Không tạo được traffic_label → không làm FS có giám sát được.")


# =============================================================
# 2. CHỌN THUỘC TÍNH NUMERIC + TIỀN XỬ LÍ
# =============================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n[STEP] Numeric columns ban đầu (", len(numeric_cols), "):")
print(numeric_cols)
# 3) Missingness overview for numeric columns (raw, before exclusions)
plot_missingness_bar(
    df,
    numeric_cols,
    top_n=min(20, len(numeric_cols)),
    out_path=os.path.join(FIG_DIR, "fig02_missing_ratio_top20_numeric.png"),
)


# Loại ID, time, geometry,... (nếu có trong numeric_col thì bỏ)
cols_to_exclude = [
    "segmentId",
    "id",
    "ID",  # ID
    "timeStamp",
    "timestamp",
    "time",  # thời gian số
    "sunrise",
    "sunset",  # thiên văn
]
cols_to_exclude = [c for c in cols_to_exclude if c in numeric_cols]

numeric_feature_cols = [c for c in numeric_cols if c not in cols_to_exclude]

# bỏ cột toàn NaN
all_nan_cols = [c for c in numeric_feature_cols if df[c].isna().all()]
if all_nan_cols:
    print("\n[INFO] Các cột toàn NaN bị loại:", all_nan_cols)
numeric_feature_cols = [c for c in numeric_feature_cols if c not in all_nan_cols]

print("Số feature numeric sau khi bỏ ID/time/NaN-only:", len(numeric_feature_cols))

# 4) Correlation heatmap BEFORE low-variance/high-corr filtering (subset)
plot_corr_heatmap(
    df,
    numeric_feature_cols,
    out_path=os.path.join(FIG_DIR, "fig03_corr_heatmap_before_filter.png"),
    max_features=25,
)

# Unsupervised FS bước 1: low-variance + high-correlation
numeric_feature_cols = drop_low_variance_and_high_corr(
    df,
    numeric_feature_cols,
    var_threshold=1e-6,
    corr_threshold=0.9,
)

# 5) Correlation heatmap AFTER filtering (subset)
plot_corr_heatmap(
    df,
    numeric_feature_cols,
    out_path=os.path.join(FIG_DIR, "fig04_corr_heatmap_after_filter.png"),
    max_features=25,
)


# Preprocessing: impute + scale
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_feature_cols),
    ],
    remainder="drop",
)

X_all = preprocess.fit_transform(df)
print("\nShape X_all (sau preprocessing):", X_all.shape)
print("Các feature thực sự dùng trong X_all (theo thứ tự):")
print(numeric_feature_cols)
# 6) Pipeline step-count summary (columns/features)
step_counts = [
    ("raw_cols", int(df_raw.shape[1])),
    ("raw_numeric", int(len(df_raw.select_dtypes(include=[np.number]).columns))),
    (
        "numeric_after_excl",
        int(
            len(
                [
                    c
                    for c in df.select_dtypes(include=[np.number]).columns
                    if c not in cols_to_exclude
                ]
            )
        ),
    ),
    (
        "numeric_no_nan_only",
        int(
            len(
                [
                    c
                    for c in [
                        c
                        for c in df.select_dtypes(include=[np.number]).columns
                        if c not in cols_to_exclude
                    ]
                    if not df[c].isna().all()
                ]
            )
        ),
    ),
    ("after_var_corr", int(len(numeric_feature_cols))),
]
plot_feature_count_pipeline(
    step_counts, out_path=os.path.join(FIG_DIR, "fig05_pipeline_feature_counts.png")
)


# =============================================================
# 3. FEATURE SELECTION: RF, DT, MUTUAL INFORMATION, ExtraTrees, (XGB)
# =============================================================

# NOTE:
# TOP_K=10 is only a *starting point* (not a universal rule).
# In practice, you should sweep TOP_K (e.g., 5, 10, 15, 20) and evaluate downstream clustering.
# For now we keep 10 to match your current experiment plan.
TOP_K = min(10, X_all.shape[1])
print(f"\n[STEP] FEATURE SELECTION (TOP_K = {TOP_K})")

fs_feature_sets = {}

print("\n[FS] Random Forest feature importance")
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_all, y_int)
rf_importance = rf.feature_importances_

rf_df = pd.DataFrame(
    {
        "feature": numeric_feature_cols,
        "importance": rf_importance,
    }
).sort_values("importance", ascending=False)

print("\n>> Bảng xếp hạng feature (RandomForest):")
print(rf_df)

rf_top_features = rf_df.head(TOP_K)["feature"].tolist()
print(f"\n>> TOP {TOP_K} feature (RandomForest):")
print(rf_top_features)
fs_feature_sets["RF"] = rf_top_features

print("\n[FS] Decision Tree feature importance")
dt = DecisionTreeClassifier(random_state=42, max_depth=None)
dt.fit(X_all, y_int)
dt_importance = dt.feature_importances_

dt_df = pd.DataFrame(
    {
        "feature": numeric_feature_cols,
        "importance": dt_importance,
    }
).sort_values("importance", ascending=False)

print("\n>> Bảng xếp hạng feature (DecisionTree):")
print(dt_df)

dt_top_features = dt_df.head(TOP_K)["feature"].tolist()
print(f"\n>> TOP {TOP_K} feature (DecisionTree):")
print(dt_top_features)
fs_feature_sets["DT"] = dt_top_features

print("\n[FS] Mutual Information (feature vs traffic_label)")
mi_scores = mutual_info_classif(X_all, y_int, discrete_features=False, random_state=42)
mi_max = mi_scores.max()
if mi_max > 0:
    mi_scores_norm = mi_scores / mi_max
else:
    mi_scores_norm = mi_scores

mi_df = pd.DataFrame(
    {
        "feature": numeric_feature_cols,
        "mutual_info_norm": mi_scores_norm,
    }
).sort_values("mutual_info_norm", ascending=False)

print("\n>> Bảng xếp hạng feature (Mutual Information – normalized to [0,1]):")
print(mi_df)

mi_top_features = mi_df.head(TOP_K)["feature"].tolist()
print(f"\n>> TOP {TOP_K} feature (MutualInfo):")
print(mi_top_features)
fs_feature_sets["MI"] = mi_top_features

print("\n[FS] ExtraTrees feature importance")
et = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)
et_df = rank_features_by_model(
    et,
    X_all,
    y_int,
    numeric_feature_cols,
    importance_attr="feature_importances_",
    score_name="importance",
)
print("\n>> Bảng xếp hạng feature (ExtraTrees):")
print(et_df)
et_top_features = et_df.head(TOP_K)["feature"].tolist()
print(f"\n>> TOP {TOP_K} feature (ExtraTrees):")
print(et_top_features)
fs_feature_sets["ET"] = et_top_features

if _HAS_XGB:
    print("\n[FS] XGBoost feature importance")
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(np.unique(y_int)),
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    xgb_df = rank_features_by_model(
        xgb,
        X_all,
        y_int,
        numeric_feature_cols,
        importance_attr="feature_importances_",
        score_name="importance",
    )
    print("\n>> Bảng xếp hạng feature (XGBoost):")
    print(xgb_df)
    xgb_top_features = xgb_df.head(TOP_K)["feature"].tolist()
    print(f"\n>> TOP {TOP_K} feature (XGBoost):")
    print(xgb_top_features)
    fs_feature_sets["XGB"] = xgb_top_features
else:
    print("\n[INFO] Chưa cài xgboost → bỏ qua FS=XGB. (pip install xgboost)")


# =============================================================
# 4. SO SÁNH CÁC TẬP FEATURE ĐƯỢC CHỌN
# =============================================================

print("\n[STEP] So sánh TOP_K feature giữa các thuật toán Feature Selection")

# In ra từng bộ feature để bạn nhìn trực tiếp (giống/khác nhau)
for k, feats in fs_feature_sets.items():
    print(f"\n- FS={k} | TOP_{TOP_K}:")
    print(feats)

print("\nHoàn thành bước Feature Selection – xem log ở trên để phân tích kỹ hơn.")


# =============================================================
# 5. CHẠY PIPELINE CLUSTERING VỚI TẤT CẢ BỘ FEATURE TRONG fs_feature_sets
#    - Chạy cho tất cả bộ feature trong fs_feature_sets
# Mục tiêu:
#   * Với MỖI bộ feature, chạy toàn bộ pipeline clustering đa phương pháp
#     (KMeans, MiniBatchKMeans, Agglomerative, DBSCAN, OPTICS, GMM).
#   * Ghi file kết quả vào thư mục con "a" cùng location:
#       - a/clustering_model_comparison_<FS>.csv
#       - a/traffic_clustering_output_<FS>.csv
#   * In ra terminal thông tin best model theo silhouette cho từng bộ FS.
# =============================================================

# Tạo thư mục output "a" nếu chưa tồn tại
output_dir = "a"
os.makedirs(output_dir, exist_ok=True)
# Ensure figure directory exists
os.makedirs(FIG_DIR, exist_ok=True)


def run_clustering_for_feature_set(
    X_all, feature_names_all, selected_features, df_source, fs_name, y_true_int
):
    """Chạy toàn bộ clustering pipeline cho 1 bộ feature.

    Tham số
    -------
    X_all : ndarray
        Ma trận feature sau preprocessing (tương ứng với feature_names_all).
    feature_names_all : list[str]
        Danh sách tên feature (theo thứ tự cột của X_all).
    selected_features : list[str]
        Bộ feature được chọn bởi một kỹ thuật Feature Selection (RF/DT/MI).
    df_source : DataFrame
        DataFrame gốc (đã có cột traffic_label, currentSpeed, freeFlowSpeed,...).
    fs_name : str
        Tên kỹ thuật FS, dùng để đặt tên file output (ví dụ: "RF", "DT", "MI").
    """
    # Chuyển từ tên feature sang index cột trong X_all
    indices = [
        feature_names_all.index(f) for f in selected_features if f in feature_names_all
    ]
    if not indices:
        print(f"[WARN] FS={fs_name}: Không tìm thấy feature nào trong X_all, bỏ qua.")
        return

    X = X_all[:, indices]
    used_features = [feature_names_all[i] for i in indices]

    print("\n==============================================")
    print(f"=== CLUSTERING với bộ feature FS={fs_name} (n={len(used_features)}) ===")
    print("Danh sách feature sử dụng:", used_features)

    metrics = []  # lưu mọi mô hình + metric nội/ngoại để xuất CSV

    best_sil = -1.0
    best_info = None  # lưu thông tin mô hình tốt nhất (theo silhouette)

    # Note: we will export a few summary figures per FS method for the report.

    def update_best(family, algo, params, labels, internal_metrics, external_metrics):
        nonlocal best_sil, best_info

        row = {
            "fs_method": fs_name,
            "family": family,
            "algorithm": algo,
            "params": str(params),
        }
        row.update(internal_metrics)
        row.update(external_metrics)
        stats = _cluster_size_stats(labels)
        row.update(stats)
        metrics.append(row)

        sil = internal_metrics.get("silhouette", None)
        n_clusters = internal_metrics.get("n_clusters", 0)

        if sil is not None and n_clusters > 1 and sil > best_sil:
            best_sil = sil
            best_info = {
                "fs_method": fs_name,
                "family": family,
                "algorithm": algo,
                "params": params,
                "labels": labels,
                "metrics_internal": internal_metrics,
                "metrics_external": external_metrics,
            }

    # ---------------------------------------------------------
    # 5.1 Partitioning: KMeans, MiniBatchKMeans
    # ---------------------------------------------------------
    print("\n=== Partitioning: KMeans ===")
    k_values = list(range(3, 20, 2))  # chỉ dùng k lẻ: 3,5,...,19
    n_init_list = [10, 20]

    for k in k_values:
        for n_init in n_init_list:
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            labels = kmeans.fit_predict(X)
            internal = evaluate_clustering_internal(X, labels)
            external = evaluate_clustering_external(y_true_int, labels)
            print(f"KMeans k={k}, n_init={n_init} -> sil={internal['silhouette']}")
            update_best(
                "partition",
                "KMeans",
                {"k": k, "n_init": n_init},
                labels,
                internal,
                external,
            )

    print("\n=== Partitioning: MiniBatchKMeans ===")
    for k in k_values:
        mbk = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256)
        labels = mbk.fit_predict(X)
        internal = evaluate_clustering_internal(X, labels)
        external = evaluate_clustering_external(y_true_int, labels)
        print(f"MiniBatchKMeans k={k} -> sil={internal['silhouette']}")
        update_best(
            "partition", "MiniBatchKMeans", {"k": k}, labels, internal, external
        )

    # ---------------------------------------------------------
    # 5.2 Hierarchical: AgglomerativeClustering (4 linkage)
    # ---------------------------------------------------------
    print("\n=== Hierarchical: AgglomerativeClustering ===")
    linkages = ["single", "complete", "average", "ward"]

    for link in linkages:
        for k in k_values:
            try:
                if link == "ward":
                    agg = AgglomerativeClustering(n_clusters=k, linkage=link)
                else:
                    agg = AgglomerativeClustering(
                        n_clusters=k, linkage=link, metric="euclidean"
                    )
                labels = agg.fit_predict(X)
                internal = evaluate_clustering_internal(X, labels)
                external = evaluate_clustering_external(y_true_int, labels)
                print(
                    f"Agglomerative link={link}, k={k} -> sil={internal['silhouette']}"
                )
                update_best(
                    "hierarchical",
                    f"Agglomerative_{link}",
                    {"k": k},
                    labels,
                    internal,
                    external,
                )
            except Exception as e:
                print(f"[WARN] Agglomerative link={link}, k={k} lỗi: {e}")

    # ---------------------------------------------------------
    # 5.3 Density-based: DBSCAN, OPTICS
    # ---------------------------------------------------------
    print("\n=== Density-based: DBSCAN ===")
    eps_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    min_samples_list = [5, 10, 20]

    for eps in eps_list:
        for ms in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            internal = evaluate_clustering_internal(X, labels)
            external = evaluate_clustering_external(y_true_int, labels)
            print(f"DBSCAN eps={eps}, ms={ms} -> sil={internal['silhouette']}")
            update_best(
                "density",
                "DBSCAN",
                {"eps": eps, "min_samples": ms},
                labels,
                internal,
                external,
            )

    print("\n=== Density-based: OPTICS ===")
    xi_list = [0.03, 0.05]
    for ms in min_samples_list:
        for xi in xi_list:
            # OPTICS đôi khi sinh RuntimeWarning kiểu "divide by zero" trong reachability ratio.
            # Mình xử lý bằng cách coi warning đó như lỗi để không spam terminal,
            # và bỏ qua cấu hình (ms, xi) nếu warning xảy ra.
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "error",
                        message=".*divide by zero encountered in divide.*",
                        category=RuntimeWarning,
                    )
                    opt = OPTICS(min_samples=ms, xi=xi)
                    labels = opt.fit_predict(X)

                internal = evaluate_clustering_internal(X, labels)
                external = evaluate_clustering_external(y_true_int, labels)
                print(f"OPTICS ms={ms}, xi={xi} -> sil={internal['silhouette']}")
                update_best(
                    "density",
                    "OPTICS",
                    {"min_samples": ms, "xi": xi},
                    labels,
                    internal,
                    external,
                )
            except RuntimeWarning as w:
                print(f"[WARN] OPTICS ms={ms}, xi={xi} bị warning (skip): {w}")
            except Exception as e:
                print(f"[WARN] OPTICS ms={ms}, xi={xi} lỗi: {e}")

    # ---------------------------------------------------------
    # 5.4 Model-based: Gaussian Mixture (EM)
    # ---------------------------------------------------------
    print("\n=== Model-based: GaussianMixture (EM) ===")
    cov_types = ["full", "tied", "diag", "spherical"]

    for n_comp in k_values:  # dùng chung k lẻ làm số thành phần
        for cov in cov_types:
            try:
                gmm = GaussianMixture(
                    n_components=n_comp, covariance_type=cov, random_state=42
                )
                labels = gmm.fit_predict(X)
                internal = evaluate_clustering_internal(X, labels)
                external = evaluate_clustering_external(y_true_int, labels)
                print(f"GMM n={n_comp}, cov={cov} -> sil={internal['silhouette']}")
                update_best(
                    "model",
                    "GMM",
                    {"n_components": n_comp, "covariance_type": cov},
                    labels,
                    internal,
                    external,
                )
            except Exception as e:
                print(f"[WARN] GMM n={n_comp}, cov={cov} lỗi: {e}")

    # ---------------------------------------------------------
    # 5.5 Ghi file kết quả cho bộ feature hiện tại
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(metrics)
    cmp_path = os.path.join(output_dir, f"clustering_model_comparison_{fs_name}.csv")
    metrics_df.to_csv(cmp_path, index=False)
    print(f"\nĐã ghi {cmp_path}")

    if best_info is None:
        print(
            f"[WARN] Không tìm được mô hình tốt cho FS={fs_name} (silhouette <= 0 hoặc chỉ 1 cụm)."
        )
        return metrics_df, None

    print("\nBest model cho FS=", fs_name)
    print("  Family   :", best_info["family"])
    print("  Algorithm:", best_info["algorithm"])
    print("  Params  :", best_info["params"])
    print("  Silhouette:", best_info["metrics_internal"].get("silhouette"))
    labels_best = best_info["labels"]

    # Phân bố kích thước cụm
    cluster_counts = pd.Series(labels_best).value_counts().sort_index()
    print("\nPhân bố kích thước cụm của mô hình tốt nhất:")
    print(cluster_counts)

    # Tạo DataFrame output cho bài toán giao thông
    df_output = df_source.copy(deep=True)
    cluster_col_name = f"cluster_label_{fs_name}"
    df_output[cluster_col_name] = labels_best

    # Chênh lệch giữa freeFlowSpeed và currentSpeed (nếu có)
    if {"currentSpeed", "freeFlowSpeed"}.issubset(df_output.columns):
        df_output["speed_drop"] = df_output["freeFlowSpeed"] - df_output["currentSpeed"]

    # Mô tả định tính mức độ đông đúc dựa trên traffic_label (nếu có)
    if "traffic_label" in df_output.columns:
        df_output["congestion_level_desc"] = df_output["traffic_label"].map(
            {
                "A": "Rất đông / kẹt xe",
                "B": "Đông",
                "C": "Trung bình",
                "D": "Tương đối thoáng",
                "E": "Thông thoáng",
            }
        )

    # Điểm kẹt xe tổng hợp (giữ nguyên ý tưởng cũ, nếu cột tồn tại)
    congestion_components = {
        "congestionIndex": 0.4,
        "speed_drop": 0.3,
        "trafficVolume": 0.2,
        "occupancy": 0.1,
    }
    df_output["congestion_score"] = 0.0
    for col, w in congestion_components.items():
        if col in df_output.columns:
            col_series = df_output[col].astype(float)
            cmin, cmax = col_series.min(), col_series.max()
            if cmax > cmin:
                norm = (col_series - cmin) / (cmax - cmin)
                df_output["congestion_score"] += w * norm

    # Các cột export chính (chỉ lấy cột thực sự tồn tại)
    export_cols = [
        "segmentId",
        "name_vn",
        "lat_start",
        "lon_start",
        "lat_end",
        "lon_end",
        "speedLimit",
        "currentSpeed",
        "freeFlowSpeed",
        "congestionIndex",
        "trafficVolume",
        "occupancy",
        "crossTime",
        "traffic_label",
        cluster_col_name,
        "speed_drop",
        "congestion_level_desc",
        "congestion_score",
    ]
    export_cols = [c for c in export_cols if c in df_output.columns]

    out_path = os.path.join(output_dir, f"traffic_clustering_output_{fs_name}.csv")
    df_output[export_cols].to_csv(out_path, index=False)
    print(f"Đã ghi {out_path}")

    return metrics_df, best_info


print(
    "\n[STEP] Chạy pipeline clustering cho từng bộ feature FS (trong fs_feature_sets)..."
)

# Collect metrics from all models across all FS methods
all_metrics_dfs = []
best_by_fs = {}
for fs_name, feats in fs_feature_sets.items():
    metrics_df, best_info = run_clustering_for_feature_set(
        X_all, numeric_feature_cols, feats, df, fs_name, y_int
    )
    if metrics_df is not None:
        all_metrics_dfs.append(metrics_df.assign(fs_method=fs_name))
    if best_info is not None:
        best_by_fs[fs_name] = best_info

# So sánh mô hình TỐT NHẤT của từng bộ feature trên nhiều tiêu chí đánh giá
best_list = list(best_by_fs.values())
if best_list:
    summary_rows = []
    for b in best_list:
        row = {
            "fs_method": b["fs_method"],
            "family": b["family"],
            "algorithm": b["algorithm"],
            "params": str(b["params"]),
        }
        row.update(b["metrics_internal"])
        row.update(b["metrics_external"])
        summary_rows.append(row)

    best_summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "best_models_summary_all_FS.csv")
    best_summary_df.to_csv(summary_path, index=False)
    print("\n[STEP] So sánh các mô hình tốt nhất giữa CÁC bộ feature (toàn bộ FS):")
    print(best_summary_df)
    print(f"\nĐã ghi bảng tổng hợp mô hình tốt nhất ra: {summary_path}")
else:
    print("\n[WARN] Không có mô hình tốt nào để so sánh giữa các bộ feature.")

#
# ---------------------------------------------------------------------
# FILTER KẾT QUẢ GOM CỤM (MODEL SELECTION / AUDIT)
#
# Vì chạy grid nhiều thuật toán/param sẽ sinh ra rất nhiều kết quả “đẹp giả”:
# - n_clusters <= 1 (tất cả dồn 1 cụm) -> silhouette/DBI không còn ý nghĩa.
# - 1 cụm chiếm gần hết điểm (largest_cluster_ratio rất lớn) -> nhìn sil có thể cao,
#   nhưng thực chất là phân cụm lệch, không hữu ích.
# - cụm quá nhỏ (min_cluster_size rất bé) -> thường là outlier/artefact.
# - với density-based, noise_ratio quá cao -> mô hình đang coi phần lớn điểm là noise.
#
# Bước 1) Gate (lọc suy biến):
#   - largest_cluster_ratio_max: tỉ lệ cụm lớn nhất / tổng điểm không vượt ngưỡng.
#   - min_cluster_size_min: kích thước cụm nhỏ nhất phải >= ngưỡng.
#   - noise_ratio_max: tỉ lệ noise (-1) không vượt ngưỡng.
#   - n_clusters phải > 1.
#   -> Nếu vi phạm, ghi rõ lý do vào gate_reason và gate_pass = False.
#
# Bước 2) Chuẩn hoá metric về [0,1] để cộng điểm:
#   - silhouette, ARI, NMI: càng cao càng tốt.
#   - davies_bouldin, RMSE: càng thấp càng tốt (được đảo chiều khi normalize).
#
# Bước 3) Tính weighted_score (điểm tổng hợp có trọng số).
#
# Bước 4) Pareto front (đa mục tiêu):
#   Chọn các nghiệm không bị “thống trị” đồng thời trên 5 mục tiêu:
#   silhouette (max), davies_bouldin (min), ARI (max), NMI (max), RMSE (min).
#
# Bước 5) Chọn best overall:
#   - Ưu tiên các model gate_pass=True.
#   - Sau đó sort theo: pareto_front desc -> weighted_score desc -> ARI desc -> silhouette desc.
#   - Lấy dòng đầu tiên làm best_model_overall.
# ---------------------------------------------------------------------
if all_metrics_dfs:
    all_models_df = pd.concat(all_metrics_dfs, ignore_index=True)
    # Gate config for degenerate filter
    gate_cfg = {
        "largest_cluster_ratio_max": 0.7,
        "min_cluster_size_min": 10,
        "noise_ratio_max": 0.5,
    }
    # Compute degenerate reason for each row
    all_models_df["gate_reason"] = all_models_df.apply(
        lambda row: _degenerate_reason(row, gate_cfg), axis=1
    )
    all_models_df["gate_pass"] = all_models_df["gate_reason"] == ""

    # Normalized metrics for scoring
    all_models_df["silhouette_norm"] = _minmax_norm(
        all_models_df["silhouette"], higher_is_better=True
    )
    all_models_df["davies_bouldin_norm"] = _minmax_norm(
        all_models_df["davies_bouldin"], higher_is_better=False
    )
    all_models_df["ARI_norm"] = _minmax_norm(
        all_models_df["ARI"], higher_is_better=True
    )
    all_models_df["NMI_norm"] = _minmax_norm(
        all_models_df["NMI"], higher_is_better=True
    )
    all_models_df["RMSE_norm"] = _minmax_norm(
        all_models_df["RMSE"], higher_is_better=False
    )

    # Weighted score (example weights, adjust as needed)
    # silhouette: 0.25, davies_bouldin: 0.15, ARI: 0.25, NMI: 0.15, RMSE: 0.20
    all_models_df["weighted_score"] = (
        0.25 * all_models_df["silhouette_norm"].fillna(0)
        + 0.15 * all_models_df["davies_bouldin_norm"].fillna(0)
        + 0.25 * all_models_df["ARI_norm"].fillna(0)
        + 0.15 * all_models_df["NMI_norm"].fillna(0)
        + 0.20 * all_models_df["RMSE_norm"].fillna(0)
    )

    # Pareto front: objectives: (silhouette max, davies_bouldin min, ARI max, NMI max, RMSE min)
    objectives = [
        ("silhouette", "max"),
        ("davies_bouldin", "min"),
        ("ARI", "max"),
        ("NMI", "max"),
        ("RMSE", "min"),
    ]
    all_models_df["pareto_front"] = pareto_front(all_models_df, objectives)

    # Choose best overall among gate_pass rows by sorting
    gate_pass_df = all_models_df[all_models_df["gate_pass"]].copy()
    if not gate_pass_df.empty:
        gate_pass_df = gate_pass_df.sort_values(
            ["pareto_front", "weighted_score", "ARI", "silhouette"],
            ascending=[False, False, False, False],
        )
        best_row = gate_pass_df.iloc[0]
    else:
        best_row = all_models_df.iloc[0]

    # Export audit and best overall
    audit_path = os.path.join(output_dir, "model_selection_audit.csv")
    all_models_df.to_csv(audit_path, index=False)
    best_path = os.path.join(output_dir, "best_model_overall.csv")
    pd.DataFrame([best_row]).to_csv(best_path, index=False)
    # Figures that correspond EXACTLY to the best overall model
    plot_best_overall_figures(
        best_row, X_all, numeric_feature_cols, fs_feature_sets, FIG_DIR
    )

    # 3rd main report output: model selection table (metrics + balance)
    plot_model_selection_table(
        best_row, out_path=os.path.join(FIG_DIR, "fig_best_model_selection_table.png")
    )

    # Optional (high-value): show tradeoff landscape for all tried models
    try:
        plot_model_selection_scatter(
            all_models_df,
            out_path=os.path.join(
                FIG_DIR, "fig06_model_selection_scatter_ARI_vs_Sil.png"
            ),
        )
    except Exception:
        pass

    print("\n[MODEL SELECTION] Đã ghi toàn bộ audit ra:", audit_path)
    print("[MODEL SELECTION] Đã ghi best_model_overall ra:", best_path)
    print("\n[MODEL SELECTION] Best model overall:")
    print(f"  FS method      : {best_row['fs_method']}")
    print(f"  Algorithm      : {best_row['algorithm']}")
    print(f"  Family         : {best_row['family']}")
    print(f"  Params         : {best_row['params']}")
    print(
        f"  Internal metrics: silhouette={best_row['silhouette']}, davies_bouldin={best_row['davies_bouldin']}, calinski_harabasz={best_row.get('calinski_harabasz', None)}"
    )
    print(
        f"  External metrics: ARI={best_row['ARI']}, NMI={best_row['NMI']}, accuracy_majority={best_row['accuracy_majority']}, MAE={best_row['MAE']}, RMSE={best_row['RMSE']}"
    )
    print(
        f"  Cluster balance: n_clusters={best_row['n_clusters']}, noise_ratio={best_row['noise_ratio']}, largest_cluster_ratio={best_row['largest_cluster_ratio']}, min_cluster_size={best_row['min_cluster_size']}, largest_cluster_size={best_row['largest_cluster_size']}"
    )
    print(f"  Weighted score : {best_row['weighted_score']}")
    print(f"  Pareto front   : {best_row['pareto_front']}")
    print(f"  Gate reason    : {best_row['gate_reason']}")

print(
    "\nHoàn thành toàn bộ pipeline: Feature Selection (nhiều kỹ thuật) + Clustering đa phương pháp cho từng bộ feature + So sánh mô hình tốt nhất."
)
