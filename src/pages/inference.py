import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.tree import plot_tree, export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import src.core.ai_models as AIM
def plot_clean_cm(y_test, y_pred, class_names=None, cbar=False):
    if class_names:
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="coolwarm",
            linecolor="white",
            linewidth=2,
            cbar=cbar,
            # annot_kws={"weight": "bold", "size": size},
            xticklabels=class_names,
            yticklabels=class_names
        )
    else:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="coolwarm",
            linecolor="white",
            linewidth=2,
            cbar=cbar,
            # annot_kws={"weight": "bold", "size": size}
        )
    ax.set_title(f"Confusion Matrix | {st.session_state.get('model_name', 'Unknown')}", fontsize=12, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    return fig


def generate_svg(model, feature_names, target_col=None):
    try:
        target_enc = st.session_state.get(f"{target_col}_encoder", None)
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=st.session_state.get("feature_names"),
            class_names= target_enc.classes_.tolist() if target_enc is not None else None,
            filled=True,
            rounded=True,
            special_characters=True
        )

        graph = graphviz.Source(dot_data)
        svg_data = graph.pipe(format="svg")

        st.download_button(
            label="📥 Download Tree as SVG",
            data=svg_data,
            file_name="decision_tree.svg",
            mime="image/svg+xml"
        )

    except Exception as e:
        st.warning(f"Could not export tree: {e}")


def evaluate_model(task:str, target:str, use_encoded:bool=False):
    if not "y_test" in st.session_state or not "y_pred" in st.session_state:
        st.warning("Train the model first to see evaluation metrics.")
        return

    model_name = st.session_state["model_name"]
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    if task == "classification":
        labels_display = None

        if not use_encoded: # back to original labels for better interpretability
            try:
                target_enc = st.session_state.get(f"{target}_encoder")
                y_test_disp = target_enc.inverse_transform(y_test)
                y_pred_disp = target_enc.inverse_transform(y_pred)
                labels_display = target_enc.classes_.tolist()
            except:
                y_test_disp = y_test
                y_pred_disp = y_pred
        else:
            y_test_disp = y_test
            y_pred_disp = y_pred
        
        acc = accuracy_score(y_test_disp, y_pred_disp)
        f1 = f1_score(y_test_disp, y_pred_disp, average='weighted')
        st.subheader(f"📊 {model_name} Evaluation: `Accuracy: {acc*100:.2f}% ({acc})` | `F1 Score: {f1*100:.2f}% ({f1})`")
        
        if labels_display is not None:
            st.pyplot(plot_clean_cm(y_test_disp, y_pred_disp, class_names=labels_display), width=1000)
        else:
            st.pyplot(plot_clean_cm(y_test_disp, y_pred_disp), width=1000)

    else: # Regression
        target_scaler = st.session_state.get("target_scaler")
        target_scaled = st.session_state.get("target_scaled", False)
        if target_scaled and target_scaler is not None:
            y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.subheader(f"📊 {model_name} Evaluation: `MSE: {mse}` | `R²: {r2 * 100:.2f}% ({r2})`")

        # =============================
        # ✅ FORMULA
        # =============================
        if model_name == "Linear Regression":
            model = st.session_state["trained_model"]
            feature_names = st.session_state["feature_names"],
            st.subheader("📝 Regression Formula")
            formula = AIM.get_linear_regression_formula(model, feature_names, target)
            st.markdown(formula)


# =========================================================
# 📊 MAIN PAGE
# =========================================================
def render():
    st.title("🤖 Inference")
    df = st.session_state.get("data")
    if df is None:
        st.warning("⚠️ No dataset loaded")
        return


    left, right = st.columns(2)
    with left:
        st.write(f"Uploaded Dataset: {st.session_state.get('uploaded_file_name', 'Unknown')}, shape: {df.shape}")
        # =====================================================
        # 🎯 TARGET
        # =====================================================
        target = st.selectbox("Select target column", df.columns, index=len(df.columns.tolist()) - 1)
        st.session_state["target"] = target

        feature_cols = [col for col in df.columns if col != target]
        st.session_state["feature_names"] = feature_cols
        X = df[feature_cols]
        y = df[target]

        st.caption("Target properties:")
        st.write("Scaler type:", st.session_state.get("target_scaler"))
        st.write("Mean value:", y.mean())
        st.write("Standard deviation:", y.std())

    with right:
        st.caption("Features properties:")
        st.write("Used for prediction:", feature_cols)
        st.write("Scaler type:", st.session_state.get("scaler"))
        st.write("Scaled columns:", st.session_state.get("scaled_columns"))
        st.write("Encoded features:", [f for f in feature_cols if f in st.session_state.get(f"{f}_enc", [])])

    # =====================================================
    # 🧠 TASK TYPE
    # =====================================================
    task_choice, model_choice, test_size_choice, train_btn = st.columns([2, 3, 3, 2])
    with task_choice:
        task = st.radio("Task type", ["classification", "regression"])

    # =====================================================
    # 🤖 MODEL (FULL ORIGINAL)
    # =====================================================
    with model_choice:
        model_name = st.selectbox(
            "Model", AIM.get_available_models(task)
        )
        st.session_state["model_name"] = model_name

    # =====================================================
    # ⚙️ SPLIT
    # =====================================================
    with test_size_choice:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    # =====================================================
    # 🚀 TRAIN
    # =====================================================
    with train_btn:
        if st.button("🚀 Train Model"):
            try:
                # 🔴 LOADER STARTS HERE
                with st.spinner("🔄 Training model, please wait..."):
                    # Optional progress bar (visual feedback)
                    progress_bar = st.progress(0)

                    model = AIM.get_model(task, st.session_state["model_name"])
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    progress_bar.progress(40)

                    # Training happens here
                    model.fit(X_train, y_train)
                    progress_bar.progress(80)

                    y_pred = model.predict(X_test)
                    progress_bar.progress(100)

                    st.session_state["y_test"] = y_test
                    st.session_state["y_pred"] = y_pred
                # 🔴 LOADER AUTOMATICALLY STOPS HERE

                st.session_state["trained_model"] = model

                train_success = st.empty()
                train_success.success("✅ Model trained successfully!")
                time.sleep(0.5)
                train_success.empty()
                
                # Visualize decision tree if applicable 
                if st.session_state["model_name"] == "Decision Tree":
                    generate_svg(st.session_state["trained_model"], feature_cols, target_col=target) 

            except Exception as e:
                st.error(f"❌ Training failed: {e}")

   
    # =============================
    # 📊 EVALUATION
    # =============================
    if "y_test" in st.session_state and "y_pred" in st.session_state:
        st.markdown("---")

        if task_choice == "classification":
            use_encoded = st.checkbox("Use encoded labels on the confusion matrix", value=False)
        else: use_encoded = False;
        evaluate_model(task, target, use_encoded)

        # =====================================================
        # 🔮 PREDICTION
        # =====================================================
        st.markdown("---")
        st.subheader("🔮 Predict")
        if "trained_model" not in st.session_state:
            st.info("Train model first")
            return

        model = st.session_state["trained_model"]
        cols = st.session_state["feature_names"]
        features_scaler = st.session_state.get("scaler")
        scaled_cols = st.session_state.get("scaled_columns", [])
        target_scaler = st.session_state.get("target_scaler")
        target_scaled = st.session_state.get("target_scaled", False)

        # Split the columns into chunks of 5 for each row
        input_data = {}
        cols_per_row = 10
        for i in range(0, len(cols), cols_per_row):
            # Get the next 5 columns for this row
            row_cols = cols[i:i+cols_per_row]
            
            # Create Streamlit columns for the row
            st_cols = st.columns(len(row_cols))  # Makes 5 equal-width columns
            
            # Add one number input per column in the row
            for idx, col_name in enumerate(row_cols):
                with st_cols[idx]:
                    val = st.number_input(col_name, value=0.0, format="%.4f")
                    input_data[col_name] = val

        btn_col, result_col = st.columns([2, 8])
        with btn_col:
            predict_clicked = st.button("Predict", use_container_width=True)
        with result_col:
            if predict_clicked:
                try:
                    input_df = pd.DataFrame([input_data])

                    # Apply feature scaling (unchanged)
                    if features_scaler is not None and len(scaled_cols) > 0:
                        cols_to_scale = [c for c in scaled_cols if c in input_df.columns]
                        if len(cols_to_scale) > 0:
                            input_df[cols_to_scale] = features_scaler.transform(input_df[cols_to_scale])

                    pred = model.predict(input_df)

                    # Inverse target scaling (unchanged)
                    if task == "regression" and target_scaled and target_scaler is not None:
                        pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

                    # 👇 Success message shows HERE (same line)
                    st.markdown(f"# `{pred[0]:.4f}`")

                except Exception as e:
                    # 👇 Error message shows HERE (same line)
                    st.error(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    render()
