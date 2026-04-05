import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.core.feature_analysis import (
    analyze_feature_correlation,
    evaluate_feature_importance,
    perform_dimensionality_reduction,
    evaluate_reduction_performance
)
import pickle, os



def render():
    # Initiate global variables to store results
    result_corr = None
    result_importance = None
    result_pca = None
    results_eval = None
    BTN_CLICKED = False # if clicked, Don't show Persistent results.

    st.title("Feature Engineering & Analysis")

    df = st.session_state.get("data")

    st.sidebar.subheader("⚙️ Analysis Controls")

    run_corr = st.sidebar.checkbox("Feature Correlation", True)
    run_importance = st.sidebar.checkbox("Feature Importance", True)
    run_pca = st.sidebar.checkbox("Dimensionality Reduction", True)
    run_eval = st.sidebar.checkbox("Performance Evaluation", False)

    save_dir = st.sidebar.text_input("Save directory", "/tmp")

    # Identify label column
    if df is not None: # only for a uploaded dataset, otherwise we will show the previous results (if any)
        label_col = st.selectbox(
            "🎯 Select Label Column",
            options=df.columns,
            index=len(df.columns) - 1  # default = last column (optional)
        )
        if label_col is None:
            st.warning("Please select a label column")
            return

        if df[label_col].isnull().all():
            st.error("❌ Selected label column is empty")
            return
    

    if st.button("🚀 Run Full Analysis"):
        BTN_CLICKED = True
        if df is None:
            st.warning("Upload data first")
            return


        if label_col not in df.columns:
            st.error("Invalid label column")
            return

        X = df.drop(columns=[label_col], errors="ignore")
        y = df[label_col]
        feature_names = X.columns.tolist()

        # Basic dataset checks
        st.write("After split → X:", X.shape, "y:", y.shape)

        X = X.select_dtypes(include=["number"])
        st.write("After numeric filter → X:", X.shape)

        X = X.fillna(X.mean())
        st.write("After fillna → X:", X.shape)

        try:
            results_corr = analyze_feature_correlation(X, y)

            st.subheader("📌 Feature-Label Correlation")
            st.dataframe(results_corr["feature_label_corr"].head(20))

            st.subheader("⚠️ Highly Correlated Feature Pairs")
            st.dataframe(results_corr["high_corr_pairs"])

            st.subheader("📊 Correlation Plot")
            st.pyplot(results_corr["figure"])

        except Exception as e:
            st.error(f"Correlation analysis failed: {e}")

        if run_importance:
            st.subheader("Feature Importance")
            try:
                results_importance = evaluate_feature_importance(X, y)

                st.write(f"Detected task: {results_importance['task']}")
                st.dataframe(results_importance["data"].head(20))
                st.pyplot(results_importance["figure"])
            except Exception as e:
                st.error(f"Feature importance failed: {e}")

        if run_pca:
            st.subheader("Dimensionality Reduction")
            
            try:
                results_pca = perform_dimensionality_reduction(X, y)

                # PCA
                st.pyplot(results_pca["pca"]["figure"])
                
                st.write("PCA Summary")
                variance_thresholds = results_pca['pca']["variance_thresholds"]
                n_components = results_pca['pca']["n_components"]
                for i, threshold in enumerate(variance_thresholds):
                    st.write(f"{int(threshold * 100)}% variance → {n_components[i]} components")

                # Feature Selection
                if "feature_selection" in results_pca:
                    st.subheader("🎯 Feature Selection")
                    st.write(f"Task: {results_pca['feature_selection']['task']}")
                    st.pyplot(results_pca["feature_selection"]["figure"])

                    st.write(results_pca["feature_selection"]["results"])
            except Exception as e:
                st.error(f"PCA failed: {e}")
          
        if run_eval:
            st.markdown("---")
            st.subheader("📊 Model-based Evaluation")

            st.caption("Compare original features vs PCA and feature selection using cross-validation.")

            try:
                results_eval = evaluate_reduction_performance(
                    X, y, feature_names, save_dir
                )

                eval_df = pd.DataFrame(results_eval)

                # --- Format CV score ---
                eval_df["CV Score"] = eval_df.apply(
                    lambda x: f"{x['cv_score_mean']:.3f} ± {x['cv_score_std']:.3f}",
                    axis=1
                )

                # --- Clean table ---
                display_df = eval_df[
                    ["method", "dimension", "CV Score", "training_score", "weighted_f1"]
                ].rename(columns={
                    "method": "Method",
                    "dimension": "Dim",
                    "training_score": "Train Score",
                    "weighted_f1": "F1 Score"
                })

                st.dataframe(display_df, use_container_width=True)

                # --- Best method ---
                best_idx = eval_df["cv_score_mean"].idxmax()
                best = eval_df.loc[best_idx]

                st.success(
                    f"🏆 Best: **{best['method']}** "
                    f"(CV Score: {best['cv_score_mean']:.3f})"
                )

                # --- Simple interpretation ---
                if best["method"] != "Original Data":
                    st.info("ℹ️ Dimensionality reduction improves model performance.")
                else:
                    st.info("ℹ️ Original features already perform best.")

            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")

        st.success("Analysis Completed")
        
        # Save results to session state for potential use in other pages
        analysis_data = {
            "correlation": results_corr,
            "importance": results_importance,
            "pca": results_pca,
            "evaluation": results_eval
        }

        # Store in session (keep this ✅)
        st.session_state["analysis_results"] = analysis_data

        # Convert to bytes
        pickle_bytes = pickle.dumps(analysis_data)

        # Download button (THIS replaces file dialog)
        st.download_button(
            label="💾 Save Analysis (.pkl)",
            data=pickle_bytes,
            file_name="analysis_results.pkl",
            mime="application/octet-stream"
        )
    
    
    # =========================================================
    # ✅ PERSISTENT DISPLAY (fix disappearing UI)
    # =========================================================
    if not BTN_CLICKED and "analysis_results" in st.session_state:

        results = st.session_state["analysis_results"]

        st.markdown("---")
        st.subheader("📂 Loaded Analysis Results")

        # --- Correlation ---
        if results.get("correlation") is not None:
            st.subheader("📌 Feature-Label Correlation")
            st.dataframe(results["correlation"]["feature_label_corr"].head(20))
            st.dataframe(results["correlation"]["high_corr_pairs"])
            st.pyplot(results["correlation"]["figure"])

        # --- Importance ---
        if results.get("importance") is not None:
            st.subheader("📊 Feature Importance")
            st.write(f"Task: {results['importance']['task']}")
            st.dataframe(results["importance"]["data"].head(20))
            st.pyplot(results["importance"]["figure"])

        # --- PCA ---
        if results.get("pca") is not None:
            st.subheader("📉 PCA Analysis")
            st.pyplot(results["pca"]["pca"]["figure"])

            variance_thresholds = results["pca"]["pca"]["variance_thresholds"]
            n_components = results["pca"]["pca"]["n_components"]

            for i, threshold in enumerate(variance_thresholds):
                st.write(f"{int(threshold * 100)}% variance → {n_components[i]} components")

            # Feature Selection
            results_fs = results["pca"]['feature_selection']
            if results_fs is not None:
                st.subheader("🎯 Feature Selection")
                st.write(f"Task: {results_fs['task']}")
                st.pyplot(results_fs["figure"])

                st.write(results_fs["results"])
                    
        # --- Evaluation ---
        if results.get("evaluation") is not None:
            st.subheader("📊 Model-based Evaluation")

            eval_df = pd.DataFrame(results["evaluation"])

            eval_df["CV Score"] = eval_df.apply(
                lambda x: f"{x['cv_score_mean']:.3f} ± {x['cv_score_std']:.3f}",
                axis=1
            )

            display_df = eval_df[
                ["method", "dimension", "CV Score", "training_score", "weighted_f1"]
            ].rename(columns={
                "method": "Method",
                "dimension": "Dim",
                "training_score": "Train Score",
                "weighted_f1": "F1 Score"
            })

            st.dataframe(display_df, use_container_width=True)

            
if __name__ == "__main__":
    render()