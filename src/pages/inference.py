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
from src.core import ai_models as AIM, utils


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


def generate_svg_decision_tree():
    try:
        target_col = st.session_state["preprocessing"]["target"]
        target_enc = st.session_state.get(f"{target_col}_encoder", None)
        dot_data = export_graphviz(
            st.session_state['inference']["model_name"],
            out_file=None,
            feature_names=st.session_state["preprocessing"].get("feature_names"),
            class_names= target_enc.classes_.tolist() if target_enc is not None else None,
            filled=True,
            rounded=True,
            special_characters=True
        )

        graph = graphviz.Source(dot_data)
        svg_data = graph.pipe(format="svg")

        st.download_button(
            label="📥 Download Tree (SVG)",
            data=svg_data,
            file_name="decision_tree.svg",
            mime="image/svg+xml"
        )

    except Exception as e:
        st.warning(f"Could not export tree: {e}")


def evaluate_model():
    if not "y_test" in st.session_state["preprocessing"] or not "y_pred" in st.session_state["inference"]:
        st.warning("Train the model first to see evaluation metrics.")
        return

    model_name = st.session_state['inference']["model_name"]
    y_test = st.session_state["preprocessing"]["y_test"]
    y_pred = st.session_state['inference']["y_pred"]
    task = st.session_state['preprocessing']['task']
    target = st.session_state['preprocessing']['target']

    # Add this: Initialize session state for evaluation data
    if "evaluation_data" not in st.session_state:
        st.session_state["evaluation_data"] = {}

    if task == "classification":
        use_encoded = st.checkbox("Use encoded labels on the confusion matrix", value=False)
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
        eval_header = f"📊 {model_name} Evaluation: `Accuracy: {acc*100:.2f}% ({acc})` | `F1 Score: {f1*100:.2f}% ({f1})`"
        
        # Save metrics to session state
        st.session_state["evaluation_data"]["header"] = eval_header
        st.session_state["evaluation_data"]["task"] = "classification"
        st.session_state["evaluation_data"]["labels_display"] = labels_display
        st.session_state["evaluation_data"]["y_test_disp"] = y_test_disp
        st.session_state["evaluation_data"]["y_pred_disp"] = y_pred_disp

        st.subheader(eval_header)
        
        if labels_display is not None:
            fig = plot_clean_cm(y_test_disp, y_pred_disp, class_names=labels_display)
            st.session_state["evaluation_data"]["fig"] = fig  # Save plot to session state
            st.pyplot(fig, width=1000)
        else:
            fig = plot_clean_cm(y_test_disp, y_pred_disp)
            st.session_state["evaluation_data"]["fig"] = fig  # Save plot to session state
            st.pyplot(fig, width=1000)

    else: # Regression
        target_scaled = st.session_state.get("is_target_scaled", False)
        target_scaler = st.session_state['preprocessing'].get("target_scaler")  # Fix: Define target_scaler before use
        if target_scaled and target_scaler is not None:
            y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        eval_header = f"📊 {model_name} Evaluation: `MSE: {mse}` | `R²: {r2 * 100:.2f}% ({r2})`"
        
        # Save regression metrics to session state
        st.session_state["evaluation_data"]["header"] = eval_header
        st.session_state["evaluation_data"]["task"] = "regression"
        st.session_state["evaluation_data"]["y_test"] = y_test
        st.session_state["evaluation_data"]["y_pred"] = y_pred
        st.session_state["evaluation_data"]["mse"] = mse
        st.session_state["evaluation_data"]["r2"] = r2

        st.subheader(eval_header)
        # =============================
        # ✅ FORMULA
        # =============================
        if model_name == "Linear Regression":
            model = st.session_state['inference']["model"]
            feature_names = st.session_state['preprocessing']["feature_names"],
            st.subheader("📝 Regression Formula")
            formula = AIM.get_linear_regression_formula(model, feature_names, target)
            st.markdown(formula)
            st.session_state["evaluation_data"]["formula"] = formula  # Save formula


def NN_builder():
    with st.expander("🧠 Build YNeural Network (Hidden Layers)", expanded=True):
        # 1. Number of hidden layers
        num_layers = st.number_input(
            "Number of hidden layers",
            min_value=1,
            max_value=20,
            value=st.session_state['inference'].get("nn_num_hidden_layers", 2),
            step=1
        )
        st.info("Note on Activations: `[1] Output:` Classification `uses` logistic `for binary and` softmax `for multi-class (supports single class & probability distribution).` Regression `uses` linear (no activation). [2] All hidden layers share one activation; [3] The diagram shows your unmodified design.")

        # 2. Dynamic HORIZONTAL layout: LAYER -> ACTIVATION -> LAYER -> ACTIVATION ...
        hidden_layer_sizes = []
        activation_functions = []
        ACTIVATIONS = ["relu", "tanh", "logistic", "identity"]

        # --------------------------
        # HORIZONTAL LAYER + ACTIVATION ROW (your exact request)
        # --------------------------
        # We build ONE ROW with: [Layer] [Activation] [Layer] [Activation] ...
        # Total widgets: layers + (layers - 1 activations)
        total_widgets = num_layers + (num_layers - 1)
        cols = st.columns(total_widgets)

        col_idx = 0
        for layer_i in range(num_layers):
            # ----------------------
            # HIDDEN LAYER NEURONS
            # ----------------------
            with cols[col_idx]:
                neurons = st.number_input(
                    f"Layer {layer_i + 1}",
                    min_value=2,
                    max_value=512,
                    value=4,
                    step=1,
                    key=f"layer_{layer_i}"
                )
                hidden_layer_sizes.append(neurons)
            col_idx += 1

            # ----------------------
            # ACTIVATION BETWEEN LAYERS (HORIZONTAL)
            # ----------------------
            if layer_i < num_layers - 1:
                with cols[col_idx]:
                    st.markdown("### <div style='text-align:center'>→</div>", unsafe_allow_html=True)
                    act = st.selectbox(
                        f"Act {layer_i + 1}",
                        options=ACTIVATIONS,
                        index=0,
                        key=f"act_{layer_i}"
                    )
                    activation_functions.append(act)
                col_idx += 1

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
            st.session_state['inference']["nn_max_iters"] = max_iters

        # 4. Early stopping
        with early_stop_col:
            early_stopping = st.checkbox("Enable Early Stopping (Early-stopping models do not allow partial training!)", value=True)
            st.session_state['inference']["nn_early_stopping"] = early_stopping

        # --------------------------
        # LIVE NEURAL NETWORK DIAGRAM
        # --------------------------
        st.markdown("### 🧠 Live Network Architecture")

        X_train = st.session_state['preprocessing'].get("X_train", None)
        input_size = X_train.shape[1] if X_train is not None else 10
        task = st.session_state['preprocessing'].get("task", "regression")
        num_of_classes = len(st.session_state["preprocessing"].get("target_labels", [0, 1]))
        output_size = 1 if task == "regression" else num_of_classes

        # Draw and display
        fig = AIM.draw_nn_architecture(
            input_size=input_size,
            hidden_layers=hidden_layer_sizes,
            activation_fncs=activation_functions,
            output_size=output_size
        )

        update_btn = st.session_state['inference']['nn_update_btn_placeholder']
        with update_btn:
            if st.button("💾 Update the model with this architecture --->(⚠️ this will reset the trained model to initial settings)"):
                # Use first activation (sklearn MLP only supports ONE global activation)
                final_act = activation_functions[0] if activation_functions else "relu"
                
                st.session_state['inference']["model"] = AIM.build_nn_model(
                    task, 
                    hidden_layer_sizes, 
                    final_act,
                    early_stopping,
                    max_iters
                )
                st.session_state['inference']["nn_num_hidden_layers"] = num_layers
                utils.temp_show("✅ New model compiled and loaded...", 'success', .5)
        # Show the plot below the buttons bar
        st.pyplot(fig, use_container_width=True)

            
# =========================================================
#                     📊 MAIN PAGE
# =========================================================
def render():
    st.title("🤖 AI Training & Inference")
    if "data" not in st.session_state:
        st.warning("⚠️ No dataset loaded")
        return
    elif not "preprocessing" in st.session_state:
        st.warning("⚠️ Preprocess the data first")
        return

    if "inference" not in st.session_state:
        st.session_state['inference'] = {};
    try:
        data_size = st.session_state['preprocessing'].get("data_size")
        test_size = st.session_state['preprocessing'].get("test_size")
        test_size_percent = test_size * 100
        train_size_percent = 100 - test_size_percent
        st.write(f"Train set size: {train_size_percent:.0f}% = {int(data_size * (1-test_size))} samples |"\
            f" Test set size: {test_size_percent:.0f}% = {int(data_size * test_size)} samples")
    except Exception as e:
        pass

    # =====================================================
    # 🧠 TASK TYPE
    # =====================================================
    model_choice, partial_train_col, train_btn_col = st.columns([4, 3, 3])

    # =====================================================
    # 🤖 MODEL (FULL ORIGINAL)
    # =====================================================
    with model_choice:
        task = st.session_state['preprocessing'].get("task")
        available_models = AIM.get_available_models(task)
        model_name = st.selectbox(
            f"Choose a model for {task}:", available_models,
            index=available_models.index(st.session_state['inference'].get("model_name", available_models[0]))
        )
        st.session_state['inference']["model_name"] = model_name

    if model_name == "Custom Neural Network": NN_builder();

    
    # =====================================================
    # Choose or not partial training
    # =====================================================
    with partial_train_col:
        partial_train = st.checkbox("Partial Training (train button will be used for incremental training)", value=False)
        st.session_state['inference']["partial_training"] = partial_train
    
    # =====================================================
    # 🚀 TRAIN MODEL SECTION
    # =====================================================
    st.session_state["plot_placeholder"] = st.empty()

    # Initialize persistent containers FIRST
    if "training_status_container" not in st.session_state:
        st.session_state["training_status_container"] = st.empty()

    # Track if training was completed (persists after page switch)
    if "training_completed" not in st.session_state['inference']:
        st.session_state['inference']['training_completed'] = False

    # Train Button
    with train_btn_col:
        btn_label = "🚀 Train Model"
        if partial_train:
            if st.session_state['inference']['training_completed']:
                btn_label = "🚀 Continue Training: 30 more epochs"  # Partial: 2nd+ pass
            else:
                btn_label = "🚀 Start Training: 30 epochs"     # Partial: first run

        if st.button(btn_label):
            st.session_state['inference']['train_btn_clicked'] = True

    # --------------------------
    # Training Execution Flow
    # --------------------------
    if st.session_state['inference'].get("train_btn_clicked", False):
        try:
            with st.session_state["training_status_container"].container():
                with st.status("🔄 Initializing...", expanded=True) as status:
                    progress_bar = st.progress(0)

                    # Step 1: Initialize
                    status.update(label="🔧 Initializing training process...", state="running")
                    time.sleep(0.5)
                    progress_bar.progress(10)

                    # Step 2: Load Model
                    status.update(label="📥 Loading model architecture...", state="running")
                    if model_name == "Custom Neural Network":
                        model = st.session_state['inference']["model"]
                    else:
                        model = AIM.get_model(task, model_name)
                    time.sleep(0.5)
                    progress_bar.progress(25)

                    # Step 3: Load Data
                    status.update(label="📂 Loading train/test data...", state="running")
                    X_train = st.session_state['preprocessing'].get("X_train")
                    X_test = st.session_state['preprocessing'].get("X_test")
                    y_train = st.session_state['preprocessing'].get("y_train")
                    y_test = st.session_state['preprocessing'].get("y_test")
                    time.sleep(0.5)
                    progress_bar.progress(40)

                    # Step 4: Train
                    status.update(label="🚀 Training model...", state="running")
                    AIM.train_model(model, X_train, y_train, X_test, y_test, task, partial_train, status)
                    progress_bar.progress(80)

                    # Step 5: Validate Predictions
                    status.update(label="✅ Validating predictions...", state="running")
                    y_pred = model.predict(X_test)
                    time.sleep(0.5)
                    progress_bar.progress(100)

                    # Finalize
                    status.update(label="✅ Training complete!", state="complete")
                    st.session_state['inference']["y_pred"] = y_pred

                    # Download button (✅ WITH UNIQUE KEY)
                    import io
                    buf = io.BytesIO()
                    fig = st.session_state['inference']['conv_curves']
                    fig.savefig(buf, format="svg", bbox_inches="tight")
                    buf.seek(0)

                    st.download_button(
                        label="📥 Download HQ image of Convergence Curve (SVG)",
                        data=buf,
                        file_name="convergence_curve.svg",
                        mime="image/svg+xml",
                        key="download_during_training"  # ✅ Unique key
                    )

            # Save trained model
            st.session_state['inference']["model"] = model
            st.session_state['inference']['training_completed'] = True

            # Decision tree visualization if needed
            if st.session_state['inference']["model_name"] == "Decision Tree":
                generate_svg_decision_tree()

            # Reset train flag
            st.session_state['inference']["train_btn_clicked"] = False

        except Exception as e:
            st.session_state['inference']['training_completed'] = False
            if isinstance(e, KeyError) and "model" in str(e) or "object has no attribute 'fit'" in str(e):
                st.markdown("### ⚠️ Your new model is not yet loaded into memory. Please press `Update model...`")
            elif "early_stopping=True" in str(e):
                st.markdown("### ⚠️ Partial model fit does not support Early Stopping. Please disable it and update the model.")
            else:
                st.error(f"❌ Training failed: {str(e)}")

    # --------------------------
    # Restore After Page Switch
    # --------------------------
    elif st.session_state['inference'].get("training_completed", False):
        with st.session_state["training_status_container"].container():
            with st.status("✅ Training complete!", expanded=True, state="complete"):
                # Restore plot
                fig = st.session_state['inference']['conv_curves']

                # Download button (✅ WITH UNIQUE KEY)
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format="svg", bbox_inches="tight")
                buf.seek(0)

                st.download_button(
                    label="📥 Download HQ image of Convergence Curve (SVG)",
                    data=buf,
                    file_name="convergence_curve.svg",
                    mime="image/svg+xml",
                    key="download_after_restore"  # ✅ Unique key
                )
                st.pyplot(fig)
        
    # =============================
    # 📊 EVALUATION
    # =============================
    if "y_test" in st.session_state['preprocessing'] and "y_pred" in st.session_state['inference']:
        with st.status("🔄 Initializing Model Evaluation...", expanded=True) as status:

            evaluate_model();

            status.update(label="✅ Evaluation ready!", state="complete")

            # =====================================================
            # 🔮 PREDICTION
            # =====================================================
            st.markdown("---")
            st.subheader("🔮 Predict")
            if "model" not in st.session_state['inference']:
                st.info("Train model first")
                return

            model = st.session_state['inference']["model"]
            cols = st.session_state['preprocessing']["feature_names"]
            features_scaler = st.session_state['preprocessing'].get("features_scaler")
            scaled_cols = st.session_state['preprocessing'].get("are_features_scaled", [])
            target_scaler = st.session_state['preprocessing'].get("target_scaler")
            target_scaled = st.session_state['preprocessing'].get("is_target_scaled", False)

            # Split the columns into chunks of 5 for each row
            input_data = {}
            cols_per_row = 10
            for i in range(0, len(cols), cols_per_row):
                # Get the next 5 columns for this row
                row_cols = cols[i:i+cols_per_row]
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
                        if task == "regression":
                            if target_scaled and target_scaler is not None:
                                pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                            st.markdown(f"### `{pred[0]:.4f}`")

                        elif task == "classification":
                            st.markdown(f"### Most likely class: `{pred[i]}`")
                            st.markdown(f"## Classes Probabilities Distibution:")

                            class_name_mapping = st.session_state.get("class_names_mapping", None)
                            pred_prob = model.predict_proba(input_df)[0]
                            pred_cols = st.columns(len(pred_prob))
                            if class_name_mapping is not None:
                                class_names = list(class_name_mapping.keys())
                                for i in range(len(pred_prob)):
                                    with pred_cols[i]:
                                        st.markdown(f"### {class_names[i]}: `{pred_prob[i]}`")
                            else:
                                for i in range(len(pred_prob)):
                                    with pred_cols[i]:
                                        st.markdown(f"### class {i}: `{pred_prob[i]}`")
                                
                    except Exception as e:
                        if "'NoneType' object has no attribute 'predict'" in str(e):
                            st.markdown("### ⚠️ Your new model is not yet loaded into memory. Please, Press `Update model...` to do so.")

                        st.error(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    render()
