import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    f_classif, SelectKBest, f_regression
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def encode_labels(catergorical_col):
    le = LabelEncoder()
    encoded_col = le.fit_transform(catergorical_col)
    return encoded_col, le


# Set plot style and font
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 100

# =========================================================
# 1. FEATURE CORRELATION ANALYSIS
# =========================================================
def analyze_feature_correlation(X, y):
    """
    Robust correlation analysis:
    - Feature-feature correlation
    - Feature-label correlation
    - Streamlit-friendly outputs
    """

    # ----------------------------------
    # ✅ VALIDATION
    # ----------------------------------
    if X is None or y is None:
        raise ValueError("X or y is None")

    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"Empty dataset: X={X.shape}, y={y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Mismatch between X and y")

    # ----------------------------------
    # ✅ CLEANING
    # ----------------------------------
    X = X.copy()
    y = y.copy()

    # Remove NaN labels
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) == 0:
        raise ValueError("No samples after removing NaN labels")

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Convert label if categorical (IMPORTANT)
    if y.dtype == "object":
        y = pd.factorize(y)[0]

    # Handle NaNs / inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    X = X.fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No features available")

    feature_names = X.columns

    # ----------------------------------
    # 🔗 FEATURE-FEATURE CORRELATION
    # ----------------------------------
    corr_matrix = X.corr()

    # Identify highly correlated pairs
    high_corr_pairs = []
    threshold = 0.8

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                high_corr_pairs.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": val
                })

    high_corr_df = pd.DataFrame(high_corr_pairs)

    # ----------------------------------
    # 🎯 FEATURE-LABEL CORRELATION
    # ----------------------------------
    feature_label_corr = pd.DataFrame({
        "Feature": feature_names,
        "Correlation_with_Label": [
            abs(X[col].corr(y)) for col in feature_names
        ]
    }).sort_values("Correlation_with_Label", ascending=False)

    # ----------------------------------
    # 📊 PLOT (Top 15)
    # ----------------------------------
    top_corr = feature_label_corr.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(top_corr["Feature"], top_corr["Correlation_with_Label"])
    ax.set_xlabel("Absolute Correlation with Label")
    ax.set_title("Top Feature-Label Correlations")

    ax.invert_yaxis()  # highest on top

    plt.tight_layout()

    # ----------------------------------
    # 📦 RETURN CLEAN STRUCTURE
    # ----------------------------------
    return {
        "corr_matrix": corr_matrix,
        "feature_label_corr": feature_label_corr,
        "high_corr_pairs": high_corr_df,
        "figure": fig
    }

# =========================================================
# 2. FEATURE IMPORTANCE
# =========================================================
def evaluate_feature_importance(X, y):
    """
    Robust + adaptive feature importance:
    - Handles classification & regression
    - Safe preprocessing
    - Streamlit-friendly (no file saving)
    """

    # ----------------------------------
    # ✅ VALIDATION
    # ----------------------------------
    if X is None or y is None:
        raise ValueError("X or y is None")

    if len(X) == 0 or len(y) == 0:
        raise ValueError(f"Empty dataset: X={X.shape}, y={y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Mismatch between X and y")

    # ----------------------------------
    # ✅ CLEANING
    # ----------------------------------
    X = X.copy()
    y = y.copy()

    # Remove NaN labels
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) == 0:
        raise ValueError("No samples after removing NaN labels")

    pd.options.display.float_format = '{:.2e}'.format
    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # Handle NaN / inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    X = X.fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No features available")

    feature_names = X.columns

    # ----------------------------------
    # ✅ STANDARDIZATION
    # ----------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------------
    # 🧠 AUTO TASK DETECTION
    # ----------------------------------
    if y.nunique() < 20 and y.dtype != float:
        task = "classification"
        mi = mutual_info_classif(X_scaled, y, random_state=42)
        f_vals, p_vals = f_classif(X_scaled, y)
    else:
        task = "regression"
        mi = mutual_info_regression(X_scaled, y, random_state=42)
        f_vals, p_vals = f_regression(X_scaled, y)

    # ----------------------------------
    # ✅ BUILD DATAFRAME
    # ----------------------------------
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mutual_Information": mi,
        "F_Score": f_vals,
        "P_Value": p_vals
    })

    importance_df = importance_df.sort_values(
        by="Mutual_Information", ascending=False
    ).reset_index(drop=True)

    # ----------------------------------
    # 📊 NORMALIZATION (for plotting)
    # ----------------------------------
    mi_norm = (importance_df["Mutual_Information"] - importance_df["Mutual_Information"].min())
    mi_norm /= (importance_df["Mutual_Information"].max() - importance_df["Mutual_Information"].min() + 1e-9)

    f_norm = (importance_df["F_Score"] - importance_df["F_Score"].min())
    f_norm /= (importance_df["F_Score"].max() - importance_df["F_Score"].min() + 1e-9)

    importance_df["MI_Normalized"] = mi_norm
    importance_df["F_Normalized"] = f_norm

    # ----------------------------------
    # 📈 PLOT (TOP 15)
    # ----------------------------------
    top_df = importance_df.head(15)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(top_df))
    width = 0.4

    ax.bar(x - width/2, top_df["MI_Normalized"], width, label="Mutual Info")
    ax.bar(x + width/2, top_df["F_Normalized"], width, label="F-score")

    ax.set_xticks(x)
    ax.set_xticklabels(top_df["Feature"], rotation=45, ha="right")
    ax.set_title(f"Feature Importance ({task})")
    ax.legend()

    plt.tight_layout()

    return {
        "data": importance_df,
        "figure": fig,
        "task": task
    }


# =========================================================
# 3. PCA / DIMENSIONALITY REDUCTION
# =========================================================
def perform_dimensionality_reduction(X:pd.DataFrame, y:pd.Series=None, feature_names=None, save_dir=None):
    """
    Robust dimensionality reduction:
    - PCA (unsupervised)
    - Feature selection (supervised if y provided)
    - Streamlit-ready outputs
    """

    # ----------------------------------
    # ✅ VALIDATION
    # ----------------------------------
    if X is None or len(X) == 0:
        raise ValueError("Empty feature matrix")

    X = X.copy()

    # Encode categorical
    X = pd.get_dummies(X, drop_first=True)

    # Handle NaN / inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    X = X.fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No features available")

    feature_names = X.columns

    # ----------------------------------
    # ✅ STANDARDIZATION
    # ----------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # ==================================
    # 📉 1. PCA (UNSUPERVISED)
    # ==================================
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    variance_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    n_components = [np.argmax(cumulative_var >= x) + 1 for x in variance_thresholds]


    # 📊 PCA Plot
    fig_pca, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='o')
    ax.axhline(0.9, linestyle='--')
    ax.axhline(0.95, linestyle='--')
    ax.axvline(n_components[4], linestyle=':')
    ax.axvline(n_components[5], linestyle=':')

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance")
    ax.set_title("PCA Explained Variance")

    plt.tight_layout()

    results["pca"] = {
        "explained_variance_ratio": explained_var,
        "cumulative_variance_ratio": cumulative_var,
        "figure": fig_pca,
        "variance_thresholds": variance_thresholds,
        "n_components": n_components
    }

    # ==================================
    # 🎯 2. FEATURE SELECTION (if y exists)
    # ==================================
    if y is not None:

        if len(y) != len(X):
            raise ValueError("Mismatch between X and y")

        # Clean y
        y = y.loc[X.index]

        if y.dtype == "object":
            y = pd.factorize(y)[0]

        # Auto detect task
        if y.nunique() < 20:
            score_func = f_classif
            task = "classification"
        else:
            score_func = f_regression
            task = "regression"

        def get_adaptive_k_values(n_features):
            k_values = sorted(set([
                max(1, int(0.2 * n_features)),
                max(1, int(0.4 * n_features)),
                max(1, int(0.6 * n_features)),
                max(1, int(0.8 * n_features)),
                n_features
            ]))
            return k_values
        k_values = get_adaptive_k_values(X.shape[1])

        fs_results = []

        scores_plot = []
        ks_plot = []

        for k in k_values:
            if k >= X.shape[1]:
                continue

            selector = SelectKBest(score_func=score_func, k=k)
            X_sel = selector.fit_transform(X_scaled, y)

            mask = selector.get_support()
            selected_features = feature_names[mask]

            mean_score = np.mean(selector.scores_[mask])

            fs_results.append({
                "k": k,
                "mean_score": mean_score,
                "selected_features": list(selected_features)
            })

            ks_plot.append(k)
            scores_plot.append(mean_score)

        # 📊 Feature Selection Plot
        fig_fs, ax = plt.subplots(figsize=(8, 5))

        ax.plot(ks_plot, scores_plot, marker='o')
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Mean Score")
        ax.set_title(f"Feature Selection ({task})")

        plt.tight_layout()

        results["feature_selection"] = {
            "results": fs_results,
            "figure": fig_fs,
            "task": task
        }

    return results

# =========================================================
# 4. MODEL EVALUATION (RAW vs REDUCED)
# =========================================================
def evaluate_reduction_performance(X, y, feature_names, save_dir='/mnt'):
    print("\n=== Dimensionality Reduction Performance Evaluation ===")

    # 🚨 Safety checks (fix your previous errors)
    if X is None or y is None or X.empty or len(y) == 0:
        raise ValueError("X or y is empty.")

    if X.shape[0] < 5:
        raise ValueError("Not enough samples for cross-validation (need ≥5).")

    # Detect task type
    is_classification = y.nunique() < 20

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def evaluate_single_scheme(X_data, y_data, method_name, random_state=42):

        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            scoring = 'r2'

        # Adaptive CV (fix small dataset crash)
        n_splits = min(5, len(y_data))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) \
            if is_classification else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        cv_scores = cross_val_score(model, X_data, y_data, cv=cv, scoring=scoring)

        model.fit(X_data, y_data)
        y_pred = model.predict(X_data)

        if is_classification:
            report = classification_report(y_data, y_pred, output_dict=True, zero_division=0)
            weighted_f1 = report['weighted avg']['f1-score']
        else:
            weighted_f1 = 0  # not applicable

        return {
            'method': method_name,
            'dimension': X_data.shape[1],
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'training_score': model.score(X_data, y_data),
            'weighted_f1': weighted_f1,
        }

    evaluation_results = []

    # --- Scheme 1: Original ---
    eval_original = evaluate_single_scheme(X_scaled, y, "Original Data")
    evaluation_results.append(eval_original)

    # --- Scheme 2: PCA ---
    n_comp = min(5, X.shape[1])  # prevent crash if <5 features
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    eval_pca = evaluate_single_scheme(X_pca, y, "PCA")
    evaluation_results.append(eval_pca)

    # --- Scheme 3 & 4: Feature Selection ---
    score_func = f_classif if is_classification else f_regression

    k10 = min(10, X.shape[1])
    selector_10 = SelectKBest(score_func=score_func, k=k10)
    X_sel_10 = selector_10.fit_transform(X_scaled, y)
    evaluation_results.append(
        evaluate_single_scheme(X_sel_10, y, f"Top {k10} Features")
    )

    k5 = min(5, X.shape[1])
    selector_5 = SelectKBest(score_func=score_func, k=k5)
    X_sel_5 = selector_5.fit_transform(X_scaled, y)
    evaluation_results.append(
        evaluate_single_scheme(X_sel_5, y, f"Top {k5} Features")
    )

    # --- Summary ---
    eval_df = pd.DataFrame(evaluation_results)
    print("\n=== Summary ===")
    print(eval_df.to_string(index=False))

    return evaluation_results