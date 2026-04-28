import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from src.core import ai_models as AIM, utils

def init_session_state():
    default = {
        "partial_training": False, "train_btn_clicked": False,
        "eval_btn_clicked": False, "training_completed": False,
        "nn_num_hidden_layers": 2, "nn_max_iters": 100,
        "nn_early_stopping": True, "model_name": "",
        "model": None, "y_pred": None, "conv_curves": None,
        "nn_update_btn_placeholder": None
    }
    if "inference" not in st.session_state:
        st.session_state["inference"] = default
    else:
        for k,v in default.items():
            if k not in st.session_state["inference"]:
                st.session_state["inference"][k] = v

def plot_confusion_matrix(y_test, y_pred, class_names=None):
    cm = confusion_matrix(y_test, y_pred)
    
    if class_names is None or len(class_names) == 0:
        xticklabels = False
        yticklabels = False
    else:
        xticklabels = class_names
        yticklabels = class_names

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    sns.heatmap(
        cm, annot=True, fmt="g", cmap="coolwarm",
        linewidth=1, cbar=False,
        xticklabels=xticklabels, yticklabels=yticklabels
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    # ✅ EXPANDER + PNG + SVG DOWNLOADS
    with st.expander("📊 Confusion Matrix", expanded=False):
        st.pyplot(fig, width=1000)

        # Save to buffers
        import io
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=200)
        buf_png.seek(0)

        buf_svg = io.BytesIO()
        fig.savefig(buf_svg, format="svg", bbox_inches="tight")
        buf_svg.seek(0)

        # Buttons side by side
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download as PNG", data=buf_png, file_name="confusion_matrix.png", mime="image/png")
        with col2:
            st.download_button("📥 Download as SVG", data=buf_svg, file_name="confusion_matrix.svg", mime="image/svg+xml")

    plt.close(fig)
    return fig

def run_model_evaluation():
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
            fig = plot_confusion_matrix(y_test_disp, y_pred_disp, class_names=labels_display)
            st.session_state["evaluation_data"]["fig"] = fig  # Save plot to session state
        else:
            fig = plot_confusion_matrix(y_test_disp, y_pred_disp)
            st.session_state["evaluation_data"]["fig"] = fig  # Save plot to session state

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

    # --------------------------
    # ✅ DECISION TREE - SVG FORMAT (Zoomable, Sharp)
    # --------------------------
    if "Decision Tree" in model_name:
        with st.expander("🌳 Decision Tree (SVG - Zoomable)", expanded=False):
            try:
                import io
                from sklearn.tree import export_graphviz
                import pydotplus

                # Export to DOT
                dot_data = export_graphviz(
                    model,
                    out_file=None,
                    feature_names=feature_names,
                    filled=True,
                    rounded=True,
                    special_characters=True
                )

                # Create SVG
                graph = pydotplus.graph_from_dot_data(dot_data)
                svg_data = graph.create_svg()

                # Display SVG in Streamlit
                st.markdown(f"<div style='overflow:auto'>{svg_data.decode('utf-8')}</div>", unsafe_allow_html=True)
                st.download_button("📥 Download as SVG Tree", data=svg_data, file_name="decision_tree.svg", mime="image/svg+xml")

            except Exception as e:
                st.info(e)

def build_neural_network_ui():
    """
    Fully compatible Neural Network builder UI with proper session state handling
    and integration with existing codebase
    """
    # Initialize session state defaults if missing
    if "inference" not in st.session_state:
        st.session_state['inference'] = {}
    
    if "nn_num_hidden_layers" not in st.session_state['inference']:
        st.session_state['inference']["nn_num_hidden_layers"] = 2
    
    if "nn_max_iters" not in st.session_state['inference']:
        st.session_state['inference']["nn_max_iters"] = 100
    
    if "nn_early_stopping" not in st.session_state['inference']:
        st.session_state['inference']["nn_early_stopping"] = True

    with st.expander("🧠 Build Your Neural Network (Hidden Layers)", expanded=True):
        # 1. Number of hidden layers
        num_layers = st.number_input(
            "Number of hidden layers",
            min_value=1,
            max_value=20,
            value=st.session_state['inference']["nn_num_hidden_layers"],
            step=1,
            key="nn_num_layers_input"
        )
        st.info(
            "📝 Notes on Activations:\n"
            "1. Output layer: Classification uses logistic (binary)/softmax (multi-class), Regression uses linear\n"
            "2. All hidden layers share ONE activation function (sklearn MLP limitation)\n"
            "3. Diagram shows your network architecture in real-time"
        )

        # 2. Dynamic HORIZONTAL layout: LAYER -> ACTIVATION -> LAYER -> ACTIVATION ...
        hidden_layer_sizes = []
        activation_functions = []
        ACTIVATIONS = ["relu", "tanh", "logistic", "identity"]

        # Calculate total widgets and create columns (handle edge case for 1 layer)
        total_widgets = num_layers + (num_layers - 1) if num_layers > 1 else num_layers
        cols = st.columns(total_widgets)

        col_idx = 0
        for layer_i in range(num_layers):
            # ----------------------
            # HIDDEN LAYER NEURONS
            # ----------------------
            with cols[col_idx]:
                # Use unique key with num_layers to prevent widget state issues
                neurons = st.number_input(
                    f"Layer {layer_i + 1} Neurons",
                    min_value=2,
                    max_value=512,
                    value=4,
                    step=1,
                    key=f"nn_layer_{num_layers}_{layer_i}_neurons"
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
                        f"Activation {layer_i + 1}",
                        options=ACTIVATIONS,
                        index=0,
                        key=f"nn_activation_{num_layers}_{layer_i}",
                        label_visibility="collapsed"
                    )
                    activation_functions.append(act)
                col_idx += 1

        # 3. Training parameters (max iterations + early stopping)
        max_ter_col, early_stop_col = st.columns(2)
        with max_ter_col:
            max_iters = st.number_input(
                "Maximum Iterations",
                min_value=100,
                max_value=10000,
                value=st.session_state['inference']["nn_max_iters"],
                step=5,
                key="nn_max_iters_input"
            )
            st.session_state['inference']["nn_max_iters"] = max_iters

        with early_stop_col:
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=st.session_state['inference']["nn_early_stopping"],
                key="nn_early_stopping_checkbox",
                help="⚠️ Early-stopping models do not support partial training!"
            )
            st.session_state['inference']["nn_early_stopping"] = early_stopping

        # --------------------------
        # LIVE NEURAL NETWORK DIAGRAM
        # --------------------------
        st.markdown("### 🧠 Live Network Architecture")

        # Safely get input size from session state (fallback to 10 if not available)
        X_train = st.session_state['preprocessing'].get("X_train", None)
        input_size = X_train.shape[1] if (X_train is not None and hasattr(X_train, 'shape')) else 10
        
        # Safely get task and output size
        task = st.session_state['preprocessing'].get("task", "regression")
        target_labels = st.session_state["preprocessing"].get("target_labels", [0, 1])
        num_of_classes = len(target_labels) if isinstance(target_labels, (list, np.ndarray)) else 2
        output_size = 1 if task == "regression" else num_of_classes

        # Draw and display network architecture
        try:
            fig = AIM.draw_nn_architecture(
                input_size=input_size,
                hidden_layers=hidden_layer_sizes,
                activation_fncs=activation_functions,
                output_size=output_size
            )
        except Exception as e:
            st.warning(f"⚠️ Could not render network diagram: {e}")
            # Create fallback figure
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.text(0.5, 0.5, f"Network Architecture\nInput: {input_size} → Hidden: {hidden_layer_sizes} → Output: {output_size}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')

        # Create update button row (fix placeholder issue)
        if st.button(
            "💾 Update Model Architecture",
            key="nn_update_model_btn",
            help="⚠️ This will reset the trained model to initial settings!"
        ):
            # Use first activation (sklearn MLP only supports ONE global activation)
            final_act = activation_functions[0] if activation_functions else "relu"
            
            # Build the model using AIM module
            try:
                new_model = AIM.build_nn_model(
                    task=task,
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation_fnc=final_act,
                    early_stopping=early_stopping,
                    max_iter=max_iters
                )
                
                # Update session state
                st.session_state['inference']["model"] = new_model
                st.session_state['inference']["nn_num_hidden_layers"] = num_layers
                st.session_state['inference']["model_name"] = "Custom Neural Network"  # Match dropdown value
                
                # Show success message
                utils.temp_show("✅ New model compiled and loaded!", 'success', 0.5)
                st.rerun()  # Refresh to reflect changes
                
            except Exception as e:
                st.error(f"❌ Failed to build model: {str(e)}")

        # Show the network diagram
        st.pyplot(fig, width='stretch')

def render_prediction_ui():
    st.markdown("---")
    st.subheader(f"🔮 Predict {st.session_state['preprocessing']['target']}")
    model = st.session_state["inference"].get("model")
    if not model:
        st.info("Train first")
        return

    feats = st.session_state["preprocessing"]["feature_names"]
    data = {}

    # ✅ USER-CHOOSABLE COLUMNS (1 to 10)
    cols_per_row = st.slider(
        "Number of input columns per row",
        min_value=1,
        max_value=10,
        value=4,  # default
        step=1
    )

    # Dynamic grid layout
    for i in range(0, len(feats), cols_per_row):
        row_features = feats[i:i+cols_per_row]
        columns = st.columns(len(row_features))

        for col, feat_name in zip(columns, row_features):
            with col:
                data[feat_name] = st.number_input(
                    feat_name, value=0.0, format="%.4f"
                )

    if st.button("Predict", width='stretch'):
        df = pd.DataFrame([data])
        pred = model.predict(df)
        st.success(f"Result: {pred[0]}")

def run_model_training():
    if not st.session_state["inference"]["train_btn_clicked"]:
        return
    if "training_status_container" not in st.session_state:
        st.session_state["training_status_container"] = st.empty()
    try:
        with st.session_state["training_status_container"].container():
            with st.status("Training...", expanded=True) as status:
                mn = st.session_state["inference"]["model_name"]
                task = st.session_state["preprocessing"]["task"]

                if mn == "Custom Neural Network":
                    model = st.session_state["inference"].get("model")
                else:
                    model = AIM.get_model(task, mn)

                Xt = st.session_state["preprocessing"]["X_train"]
                Xte = st.session_state["preprocessing"]["X_test"]
                yt = st.session_state["preprocessing"]["y_train"]
                yte = st.session_state["preprocessing"]["y_test"]

                AIM.train_model(model, Xt, yt, Xte, yte, task, st.session_state["inference"]["partial_training"])

                # Save model & predictions
                st.session_state["inference"]["model"] = model
                st.session_state["inference"]["y_pred"] = model.predict(Xte)
                st.session_state["inference"]["training_completed"] = True

                status.update(label="✅ Training complete!", state="complete")

    except Exception as e:
        if "'NoneType' object has no attribute 'fit'" in str(e):
            st.warning("⚠️ Update the model before you train!")
        else:
            st.error(f"Error: {e}")
        st.rerun()
    finally:
        st.session_state["inference"]["train_btn_clicked"] = False


def render():
    st.title("🤖 Inference")
    if "data" not in st.session_state or "preprocessing" not in st.session_state:
        st.warning("Upload & preprocess data first")
        return

    init_session_state()
    task = st.session_state["preprocessing"]["task"]
    available_models = AIM.get_available_models(task)

    if not available_models:
        st.error("No models available")
        return

    # ✅ FIXED: No more ValueError
    current_model = st.session_state["inference"].get("model_name")
    if current_model not in available_models:
        current_model = available_models[0]

    mcol, pcol, tcol, rcol = st.columns([4,2,2,2])
    with mcol:
        model_name = st.selectbox("Model", available_models, index=available_models.index(current_model))
        st.session_state["inference"]["model_name"] = model_name

    if model_name == "Custom Neural Network":
        build_neural_network_ui()

    with pcol:
        pt = st.checkbox("Partial Training", value=st.session_state["inference"]["partial_training"])
        st.session_state["inference"]["partial_training"] = pt

    with rcol:
        if st.button("Reset"):
            st.session_state.pop("inference")
            st.rerun()

    with tcol:
        if st.button("Train"):
            st.session_state["inference"]["train_btn_clicked"] = True

    run_model_training()

    if st.session_state["inference"]["training_completed"]:
        with st.expander("Model Conversion Curves"):
            buf, fig = AIM.conv_curves("conv_curves_in_training")
            st.pyplot(fig, width="stretch")

        run_model_evaluation()
        render_prediction_ui()

if __name__ == "__main__":
    render()