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
        target_scaled = st.session_state.get("is_target_scaled", False)
        if target_scaled and target_scaler is not None:
            target_scaler = st.session_state.get("target_scaler")
            y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.subheader(f"📊 {model_name} Evaluation: `MSE: {mse}` | `R²: {r2 * 100:.2f}% ({r2})`")

        # =============================
        # ✅ FORMULA
        # =============================
        if model_name == "Linear Regression":
            model = st.session_state["model"]
            feature_names = st.session_state["feature_names"],
            st.subheader("📝 Regression Formula")
            formula = AIM.get_linear_regression_formula(model, feature_names, target)
            st.markdown(formula)


def NN_builder(task:str):
    with st.expander("🧠 Build YNeural Network (Hidden Layers)", expanded=True):
        # 1. Number of hidden layers
        num_layers = st.number_input(
            "Number of hidden layers",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        # 2. Dynamic grid: MAX 10 COLUMNS PER ROW → auto wrap
        hidden_layer_sizes = []
        max_cols_per_row = 10

        # Calculate how many full rows + remaining columns
        num_rows = (num_layers + max_cols_per_row - 1) // max_cols_per_row

        for row_idx in range(num_rows):
            # How many columns in this row
            start = row_idx * max_cols_per_row
            end = min(start + max_cols_per_row, num_layers)
            current_cols = end - start

            # Create row
            cols = st.columns(current_cols)

            # Fill inputs
            for i_in_row, global_idx in enumerate(range(start, end)):
                with cols[i_in_row]:
                    neurons = st.number_input(
                        f"Layer {global_idx + 1} neurons",
                        min_value=2,
                        max_value=512,
                        value=4,
                        step=1,
                        key=f"layer_{global_idx}"
                    )
                    hidden_layer_sizes.append(neurons)

        # 3. max iterations
        max_ter_col, early_stop_col = st.columns(2)
        with max_ter_col:
            max_iters = st.number_input(
                "Maximum iterations",
                min_value=100,
                max_value=10000,
                value=100,
                step=5
            )

        # 4. Early stopping
        with early_stop_col:
            early_stopping = st.checkbox("Enable Early Stopping (Early-stopping models do not allow partial training!)", value=True)

        # --------------------------
        # LIVE NEURAL NETWORK DIAGRAM
        # --------------------------
        # with st.button("🎨 Visualize NN Architecture"):
        st.markdown("### 🧠 Live Network Architecture")

        # Get input size from your X_train (SAFE version)
        X_train = st.session_state.get("X_train", None)
        input_size = X_train.shape[1] if X_train is not None else 10

        # Output size based on task
        task = st.session_state.get("task", "regression")
        output_size = 1

        # Draw and display
        fig = AIM.draw_nn_architecture(
            input_size=input_size,
            hidden_layers=hidden_layer_sizes,
            output_size=output_size,
            task=task
        )
        st.pyplot(fig, use_container_width=True)

        st.session_state["model"] = AIM.build_nn_model(task, hidden_layer_sizes, early_stopping, max_iters)


# =========================================================
# 📊 MAIN PAGE
# =========================================================
def render():
    st.title("🤖 AI Training & Inference")
    if "data" not in st.session_state:
        st.warning("⚠️ No dataset loaded")
        return
    elif not "X_train" in st.session_state or not "y_train" in st.session_state:
        st.warning("⚠️ Preprocess the data first")
        return

    data_size = st.session_state.get("data_size")
    test_size = st.session_state.get("test_size")
    test_set_size = test_size * 100
    train_set_size = 100 - test_set_size
    st.write(f"Train set size: {train_set_size:.0f}% = {int(data_size) * float(1-test_size)} samples |"\
        f" Test set size: {test_set_size:.0f}% = {int(data_size) * float(test_size)} samples")

    # =====================================================
    # 🧠 TASK TYPE
    # =====================================================
    model_choice, train_btn = st.columns([7, 3])

    # =====================================================
    # 🤖 MODEL (FULL ORIGINAL)
    # =====================================================
    with model_choice:
        task = st.session_state.get("task", "classification")
        model_name = st.selectbox(
            f"Choose a model for {task}:", AIM.get_available_models(task)
        )
        st.session_state["model_name"] = model_name

    if model_name == "Custom Neural Network":
        NN_builder(task)


    # =====================================================
    # 🚀 TRAIN
    # =====================================================
    st.session_state["plot_placeholder"] = st.empty()
    with train_btn:
        if st.button("🚀 Train Model"):
            st.session_state["train_btn_clicked"] = True

    if st.session_state.get("train_btn_clicked", None) is True:
        try:
            # 🔴 LOADER STARTS HERE
            with st.status("🔄 Initializing...", expanded=True) as status:
                progress_bar = st.progress(0)

                # Step 1: Initializing
                status.update(label="🔧 Initializing training process...", state="running")
                time.sleep(0.5)
                progress_bar.progress(10)

                # Step 2: Loading model
                status.update(label="📥 Loading model architecture...", state="running")
                if model_name == "Custom Neural Network":
                    model = st.session_state["model"]  # Custom NN already sets in session
                else:
                    model = AIM.get_model(task, model_name)
                time.sleep(0.5)
                progress_bar.progress(25)

                # Step 3: Loading data splits
                status.update(label="📂 Loading train/test data...", state="running")
                X_train = st.session_state.get("X_train")
                X_test = st.session_state.get("X_test")
                y_train = st.session_state.get("y_train")
                y_test = st.session_state.get("y_test")
                time.sleep(0.5)
                progress_bar.progress(40)

                # Step 4: Training (NO CHANGES HERE)
                status.update(label="🚀 Training model...", state="running")
                AIM.train_model(model, X_train, y_train, X_test, y_test)
                progress_bar.progress(80)

                # Step 5: Validating / Predicting
                status.update(label="✅ Validating predictions...", state="running")
                y_pred = model.predict(X_test)
                time.sleep(0.5)
                progress_bar.progress(100)

                # Final: Complete
                status.update(label="✅ Training complete!", state="complete")

                st.session_state["y_pred"] = y_pred
                

            st.session_state["model"] = model

            train_success = st.empty()
            train_success.success("✅ Model trained successfully!")
            time.sleep(0.5)
            train_success.empty()

            if st.session_state["model_name"] == "Decision Tree":
                generate_svg(st.session_state["model"], st.session_state["feature_names"], target_col=st.session_state["target"])

            st.session_state["train_btn_clicked"] = False
        except Exception as e:
            st.error(f"❌ Training failed: {e}")

    # =============================
    # 📊 EVALUATION
    # =============================
    if "y_test" in st.session_state and "y_pred" in st.session_state:
        st.markdown("---")

        if task == "classification":
            use_encoded = st.checkbox("Use encoded labels on the confusion matrix", value=False)
        else: use_encoded = False;
        evaluate_model(task, st.session_state["target"], use_encoded)

        # =====================================================
        # 🔮 PREDICTION
        # =====================================================
        st.markdown("---")
        st.subheader("🔮 Predict")
        if "model" not in st.session_state:
            st.info("Train model first")
            return

        model = st.session_state["model"]
        cols = st.session_state["feature_names"]
        features_scaler = st.session_state.get("features_scaler")
        scaled_cols = st.session_state.get("are_features_scaled", [])
        target_scaler = st.session_state.get("target_scaler")
        target_scaled = st.session_state.get("is_target_scaled", False)

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
