# src/pages/feature_eng_analysis.py
"""Use st.session_state['feature_analysis']['key'] to store and retrieve related variables."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import warnings
import numpy as np
from typing import Optional, Dict, Any
from src.core.feature_analysis import (
    analyze_feature_correlation,
    evaluate_feature_importance,
    perform_dimensionality_reduction,
    evaluate_reduction_performance
)
import src.core.utils as utils

# Set matplotlib style for consistent plots
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')

# ------------------------------
# Helper Functions
# ------------------------------
def initialize_session_state() -> None:
    """Initialize feature analysis session state with default values."""
    default_state = {
        "analysis_results": None,
        "current_task_type": None,
        "data_validated": False,
        "last_analysis_params": None
    }
    
    if 'feature_analysis' not in st.session_state:
        st.session_state['feature_analysis'] = default_state
    else:
        # Add new keys if missing (backward compatibility)
        for key, value in default_state.items():
            if key not in st.session_state['feature_analysis']:
                st.session_state['feature_analysis'][key] = value

def validate_data(df: pd.DataFrame, label_col: str) -> tuple[bool, str]:
    """Validate input data before analysis."""
    # Check label column exists
    if label_col not in df.columns:
        return False, f"Label column '{label_col}' not found in dataset"
    
    # Check label column is not empty
    if df[label_col].isnull().all():
        return False, "Selected label column contains only missing values"
    
    # Check numeric features exist
    numeric_df = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=["number"])
    if numeric_df.empty:
        return False, "No numeric features found for analysis (required for all feature engineering tasks)"
    
    # Check dataset size (warn for very large datasets)
    if len(df) > 1_000_000:
        st.warning("⚠️ Large dataset detected (>1M rows) - analysis may take longer and use more memory")
    
    return True, "Data validation passed"

def prepare_analysis_data(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series, list]:
    """Prepare cleaned data for feature analysis."""
    # Split features and target
    X = df.drop(columns=[label_col], errors="ignore")
    y = df[label_col]
    
    # Keep only numeric features
    X = X.select_dtypes(include=["number"])
    
    # Impute missing values (mean for numeric)
    X = X.fillna(X.mean())
    
    # Store task type (classification/regression)
    unique_vals = y.nunique()
    if pd.api.types.is_numeric_dtype(y) and unique_vals > 10:
        st.session_state['feature_analysis']['current_task_type'] = "Regression"
    else:
        st.session_state['feature_analysis']['current_task_type'] = "Classification"
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names

def render_analysis_section(title: str, content_func, expanded: bool = False):
    """Reusable wrapper for analysis sections with error handling."""
    with st.expander(title, expanded=expanded):
        try:
            content_func()
        except Exception as e:
            st.error(f"❌ Error in {title}: {str(e)}")
            st.exception(e)  # Show full traceback in debug mode

def download_result_button(data: Any, filename: str, label: str):
    """Create a download button for analysis results/plots."""
    if isinstance(data, plt.Figure):
        # Save plot to bytes
        import io
        buf = io.BytesIO()
        data.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label=label,
            data=buf,
            file_name=filename,
            mime="image/png"
        )
    else:
        # Save pickle data
        pickle_bytes = pickle.dumps(data)
        st.download_button(
            label=label,
            data=pickle_bytes,
            file_name=filename,
            mime="application/octet-stream"
        )

# ------------------------------
# Main Render Function
# ------------------------------
def render():
    st.set_page_config(
        page_title="Feature Engineering & Analysis",
        page_icon="🔬",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Page Header
    st.title("🔬 Feature Engineering & Analysis")
    st.markdown("### Comprehensive feature analysis, importance, and dimensionality reduction")
    st.divider()

    # ------------------------------
    # Control Panel
    # ------------------------------
    with st.expander("⚙️ Data Analysis Control Panel", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload for precomputed results
            uploaded_file = st.file_uploader(
                "📤 Load Precomputed Analysis (.pkl)",
                type=["pkl"],
                help="Upload a saved analysis results file to view without re-running"
            )
            
            if uploaded_file is not None:
                with st.spinner("Loading analysis results..."):
                    try:
                        loaded_results = pickle.load(uploaded_file)
                        st.session_state['feature_analysis']["analysis_results"] = loaded_results
                        utils.temp_show("✅ Analysis loaded successfully!", 'success', 2.0)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load file: {str(e)}")
        
        with col2:
            # Clear results button
            if st.button(
                "♻️ Clear All Results",
                type="secondary",
                width='stretch',
                disabled=st.session_state['feature_analysis']["analysis_results"] is None
            ):
                st.session_state['feature_analysis'] = {
                    "analysis_results": None,
                    "current_task_type": None,
                    "data_validated": False,
                    "last_analysis_params": None
                }
                utils.temp_show("🔄 Results cleared successfully!", 'success', 1.0)
                st.rerun()
        
        st.divider()
        
        # Check for main dataset
        df = st.session_state.get("data")
        if "data" not in st.session_state or df is None:
            st.warning(
                "⚠️ Please upload your main dataset first (in the Data Upload page) to run new analysis",
                icon="ℹ️"
            )
            run_analysis_disabled = True
        else:
            run_analysis_disabled = False
            
            # Dataset info
            st.info(
                f"📊 Current Dataset: {df.shape[0]} rows × {df.shape[1]} columns",
                icon="ℹ️"
            )
            
            # Analysis Configuration
            st.subheader("📋 Analysis Configuration", help="Select which analyses to run")
            
            # Column layout for configuration
            config_col1, config_col2 = st.columns([1, 2])
            
            with config_col1:
                # Target column selection
                label_col = st.selectbox(
                    "🎯 Target Column",
                    options=df.columns,
                    index=len(df.columns)-1 if len(df.columns) > 0 else 0,
                    help="Select the target variable for analysis"
                )
                
                # Task type display (preview)
                if label_col in df.columns:
                    unique_vals = df[label_col].nunique()
                    if pd.api.types.is_numeric_dtype(df[label_col]) and unique_vals > 10:
                        task_type = "Regression"
                    else:
                        task_type = "Classification"
                    st.success(f"🔍 Detected Task Type: {task_type}")
            
            with config_col2:
                # Analysis checkboxes with better layout
                st.markdown("#### Select Analyses to Run")
                run_corr = st.checkbox("📈 Feature Correlation Analysis", True)
                run_importance = st.checkbox("🎯 Feature Importance", True)
                run_pca = st.checkbox("📉 Dimensionality Reduction (PCA)", True)
                run_eval = st.checkbox("📊 Model-based Evaluation", True)
            
            # Data validation
            validation_success, validation_msg = validate_data(df, label_col)
            if not validation_success:
                st.error(f"❌ {validation_msg}")
                run_analysis_disabled = True
            else:
                st.success(f"✅ {validation_msg}", icon="✔️")
            
            # Run Analysis Button
            st.divider()
            if st.button(
                "🚀 Run Full Analysis",
                type="primary",
                width='stretch',
                disabled=run_analysis_disabled
            ):
                with st.spinner("Running comprehensive feature analysis... This may take a moment ⏳"):
                    # Prepare data
                    X, y, feature_names = prepare_analysis_data(df, label_col)
                    
                    # Store analysis parameters
                    st.session_state['feature_analysis']['last_analysis_params'] = {
                        "label_col": label_col,
                        "run_corr": run_corr,
                        "run_importance": run_importance,
                        "run_pca": run_pca,
                        "run_eval": run_eval
                    }
                    
                    # Initialize results dictionary
                    analysis_data = {}
                    
                    # ------------------------------
                    # Correlation Analysis
                    # ------------------------------
                    if run_corr:
                        def corr_content():
                            results_corr = analyze_feature_correlation(X, y)
                            analysis_data["correlation"] = results_corr
                            
                            st.subheader("📌 Feature-Label Correlation")
                            st.dataframe(results_corr["feature_label_corr"].head(20), width='stretch')
                            
                            st.subheader("⚠️ Highly Correlated Feature Pairs (r > 0.8)")
                            st.dataframe(results_corr["high_corr_pairs"], width='stretch')
                            
                            st.subheader("📊 Correlation Heatmap")
                            st.pyplot(results_corr["figure"])
                            download_result_button(
                                results_corr["figure"],
                                "feature_correlation_heatmap.png",
                                "💾 Download Correlation Plot"
                            )
                        
                        render_analysis_section(
                            "Feature-Feature & Feature-Label Correlation",
                            corr_content,
                            expanded=False
                        )
                    
                    # ------------------------------
                    # Feature Importance
                    # ------------------------------
                    if run_importance:
                        def importance_content():
                            results_importance = evaluate_feature_importance(X, y)
                            analysis_data["importance"] = results_importance
                            
                            st.write(f"🎯 Task Type: {results_importance['task']}")
                            st.subheader("📊 Top Feature Importance")
                            st.dataframe(results_importance["data"].head(20), width='stretch')
                            st.pyplot(results_importance["figure"])
                            download_result_button(
                                results_importance["figure"],
                                "feature_importance_plot.png",
                                "💾 Download Importance Plot"
                            )
                        
                        render_analysis_section(
                            "Feature Importance Analysis",
                            importance_content,
                            expanded=False
                        )
                    
                    # ------------------------------
                    # Dimensionality Reduction (PCA)
                    # ------------------------------
                    if run_pca:
                        def pca_content():
                            results_pca = perform_dimensionality_reduction(X, y)
                            analysis_data["pca"] = results_pca
                            
                            # PCA Results
                            st.subheader("📉 PCA Variance Analysis")
                            st.pyplot(results_pca["pca"]["figure"])
                            download_result_button(
                                results_pca["pca"]["figure"],
                                "pca_variance_plot.png",
                                "💾 Download PCA Plot"
                            )
                            
                            st.subheader("📊 PCA Summary")
                            variance_thresholds = results_pca['pca']["variance_thresholds"]
                            n_components = results_pca['pca']["n_components"]
                            pca_df = pd.DataFrame({
                                "Variance Explained": [f"{int(t * 100)}%" for t in variance_thresholds],
                                "Number of Components": n_components
                            })
                            st.table(pca_df)
                            
                            # Feature Selection Results
                            if "feature_selection" in results_pca:
                                st.subheader("🎯 Feature Selection Results")
                                st.write(f"Task: {results_pca['feature_selection']['task']}")
                                st.pyplot(results_pca["feature_selection"]["figure"])
                                download_result_button(
                                    results_pca["feature_selection"]["figure"],
                                    "feature_selection_plot.png",
                                    "💾 Download Feature Selection Plot"
                                )
                                st.write(results_pca["feature_selection"]["results"])
                        
                        render_analysis_section(
                            "Dimensionality Reduction (PCA)",
                            pca_content,
                            expanded=False
                        )
                    
                    # ------------------------------
                    # Model-based Evaluation
                    # ------------------------------
                    if run_eval:
                        def eval_content():
                            # Get save dir (ensure it exists)
                            save_dir = os.path.join("temp", "feature_analysis")
                            os.makedirs(save_dir, exist_ok=True)
                            
                            results_eval = evaluate_reduction_performance(
                                X, y, feature_names, save_dir
                            )
                            analysis_data["evaluation"] = results_eval
                            
                            st.caption("📊 Cross-validation comparison: Original vs Reduced Features")
                            
                            # Format results
                            eval_df = pd.DataFrame(results_eval)
                            eval_df["CV Score"] = eval_df.apply(
                                lambda x: f"{x['cv_score_mean']:.3f} ± {x['cv_score_std']:.3f}",
                                axis=1
                            )
                            
                            # Clean display dataframe
                            display_df = eval_df[
                                ["method", "dimension", "CV Score", "training_score", "weighted_f1"]
                            ].rename(columns={
                                "method": "Method",
                                "dimension": "Dim",
                                "training_score": "Train Score",
                                "weighted_f1": "F1 Score"
                            })
                            
                            st.dataframe(display_df, width='stretch')
                            
                            # Highlight best method
                            best_idx = eval_df["cv_score_mean"].idxmax()
                            best = eval_df.loc[best_idx]
                            st.success(
                                f"🏆 Best Performing Method: **{best['method']}** "
                                f"(CV Score: {best['cv_score_mean']:.3f})"
                            )
                            
                            # Interpretation
                            if best["method"] != "Original Data":
                                st.info("ℹ️ Dimensionality reduction improves model performance - consider using reduced features")
                            else:
                                st.info("ℹ️ Original features perform best - no need for dimensionality reduction")
                        
                        render_analysis_section(
                            "Model-based Evaluation (Cross-Validation)",
                            eval_content,
                            expanded=False
                        )
                    
                    # ------------------------------
                    # Save Results
                    # ------------------------------
                    # Store in session state
                    st.session_state['feature_analysis']["analysis_results"] = analysis_data
                    
                    # Download full results
                    pickle_bytes = pickle.dumps(analysis_data)
                    st.download_button(
                        label="💾 Save Full Analysis Results (.pkl)",
                        data=pickle_bytes,
                        file_name=f"feature_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream",
                        width='stretch'
                    )
                    
                    utils.temp_show("✅ Full feature analysis completed successfully!", 'success', 3.0)

    # ------------------------------
    # Persistent Results Display
    # ------------------------------
    results = st.session_state['feature_analysis']["analysis_results"]
    if results is not None:
        st.divider()
        with st.expander("📂 Loaded Analysis Results", expanded=True):
            st.subheader("📋 Previous Analysis Results", divider="blue")
            
            # Show analysis parameters
            params = st.session_state['feature_analysis'].get("last_analysis_params")
            if params:
                st.caption(f"🔧 Analysis Parameters: Target='{params['label_col']}' | "
                          f"Corr={params['run_corr']} | "
                          f"Importance={params['run_importance']} | "
                          f"PCA={params['run_pca']} | "
                          f"Eval={params['run_eval']}")
            
            # Task type display
            task_type = st.session_state['feature_analysis'].get("current_task_type")
            if task_type:
                st.success(f"🎯 Detected Task Type: {task_type}")
            
            # Correlation Results
            if results.get("correlation") is not None:
                render_analysis_section(
                    "📈 Feature Correlation Results",
                    lambda: display_saved_results("correlation", results["correlation"]),
                    expanded=False
                )
            
            # Importance Results
            if results.get("importance") is not None:
                render_analysis_section(
                    "🎯 Feature Importance Results",
                    lambda: display_saved_results("importance", results["importance"]),
                    expanded=False
                )
            
            # PCA Results
            if results.get("pca") is not None:
                render_analysis_section(
                    "📉 PCA & Dimensionality Reduction Results",
                    lambda: display_saved_results("pca", results["pca"]),
                    expanded=False
                )
            
            # Evaluation Results
            if results.get("evaluation") is not None:
                render_analysis_section(
                    "📊 Model Evaluation Results",
                    lambda: display_saved_results("evaluation", results["evaluation"]),
                    expanded=False
                )

# ------------------------------
# Saved Results Display Function
# ------------------------------
def display_saved_results(result_type: str, result_data: Dict[str, Any]):
    """Display saved analysis results consistently."""
    if result_type == "correlation":
        st.subheader("📌 Feature-Label Correlation")
        st.dataframe(result_data["feature_label_corr"].head(20), width='stretch')
        
        st.subheader("⚠️ Highly Correlated Feature Pairs")
        st.dataframe(result_data["high_corr_pairs"], width='stretch')
        
        st.subheader("📊 Correlation Plot")
        st.pyplot(result_data["figure"])
    
    elif result_type == "importance":
        st.write(f"🎯 Task Type: {result_data['task']}")
        st.subheader("📊 Feature Importance")
        st.dataframe(result_data["data"].head(20), width='stretch')
        st.pyplot(result_data["figure"])
    
    elif result_type == "pca":
        st.subheader("📉 PCA Variance Analysis")
        st.pyplot(result_data["pca"]["figure"])
        
        st.subheader("📊 PCA Summary")
        variance_thresholds = result_data['pca']["variance_thresholds"]
        n_components = result_data['pca']["n_components"]
        for i, threshold in enumerate(variance_thresholds):
            st.write(f"{int(threshold * 100)}% variance → {n_components[i]} components")
        
        if "feature_selection" in result_data:
            st.subheader("🎯 Feature Selection Results")
            st.write(f"Task: {result_data['feature_selection']['task']}")
            st.pyplot(result_data["feature_selection"]["figure"])
            st.write(result_data["feature_selection"]["results"])
    
    elif result_type == "evaluation":
        eval_df = pd.DataFrame(result_data)
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
        
        st.dataframe(display_df, width='stretch')
        
        # Highlight best method
        best_idx = eval_df["cv_score_mean"].idxmax()
        best = eval_df.loc[best_idx]
        st.success(
            f"🏆 Best Method: **{best['method']}** (CV Score: {best['cv_score_mean']:.3f})"
        )

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    render()