import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def plot_clean_cm(y_test, y_pred, class_names=None, cbar=False):
    size = 100 // len(set(y_test))  # Adjust size based on number of classes
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
    ax.set_title("Confusion Matrix", fontsize=12, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    return fig


# =========================================================
# 🧠 MODEL FACTORY (FULL ORIGINAL)
# =========================================================
def get_model(task, model_name):
    if task == "classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier()
        elif model_name == "Neural Network":
            return MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=1000, early_stopping=True)
    elif task == "regression":
        if model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "Decision Tree":
            return DecisionTreeRegressor()
        elif model_name == "Neural Network":
            return MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), max_iter=1000, early_stopping=True)
    return None

# =========================================================
# 📝 TEACHING FORMULA (SAFE: NO TARGET IN FEATURES)
# =========================================================
def get_linear_regression_formula(model, feature_names, target_name):
    intercept = model.intercept_
    coefs = model.coef_

    parts = [f"{intercept:.4f}"]
    for coef, feat in zip(coefs, feature_names):
        if coef >= 0:
            parts.append(f"+ {coef:.4f} × {feat}")
        else:
            parts.append(f"- {abs(coef):.4f} × {feat}")

    formula = f"**{target_name} = {' '.join(parts)}**"
    return formula


def evaluate_model(task:str, target:str, use_encoded:bool=False):
    if not "y_test" in st.session_state or not "y_pred" in st.session_state:
        st.warning("Train the model first to see evaluation metrics.")
        return
     
    if task == "classification":
        y_test = st.session_state[f"{target}_test"]
        y_pred = st.session_state[f"{target}_pred"]
        labels_display = None

        if not use_encoded: # back to original labels for better interpretability
            try:
                target_enc = st.session_state.get(f"{target}_enc")
                y_test_disp = target_enc.inverse_transform(y_test)
                y_pred_disp = target_enc.inverse_transform(y_pred)
                labels_display = target_enc.classes_.tolist()
            except:
                y_test_disp = y_test
                y_pred_disp = y_pred
        else:
            y_test_disp = y_test
            y_pred_disp = y_pred
        
        st.write("Accuracy:", accuracy_score(y_test_disp, y_pred_disp))
        st.write("F1:", f1_score(y_test_disp, y_pred_disp, average="weighted"))
        
        if labels_display is not None:
            st.pyplot(plot_clean_cm(y_test_disp, y_pred_disp, class_names=labels_display))
        else:
            st.pyplot(plot_clean_cm(y_test_disp, y_pred_disp))

    else: # Regression
        scaler_y = st.session_state.get("scaler_y")
        target_scaled = st.session_state.get("target_scaled", False)
        if target_scaled and scaler_y is not None:
            y_test = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R2:", r2_score(y_test, y_pred))

        # =============================
        # ✅ FORMULA
        # =============================
        st.subheader("📝 Regression Formula (Teaching)")
        if model_name == "Linear Regression":
            formula = get_linear_regression_formula(model, feature_cols, target)
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
    st.write(f"Dataset shape: {df.shape}")

    # =====================================================
    # 🎯 TARGET
    # =====================================================
    target = st.selectbox("Select target column", df.columns, index=len(df.columns.tolist()) - 1)
    st.session_state["target"] = target

    feature_cols = [col for col in df.columns if col != target]
    X = df[feature_cols]
    y = df[target]

    st.caption("Features properties:")
    st.write("Used for prediction:", feature_cols)
    st.write("Scaler type:", st.session_state.get("scaler"))
    st.write("Scaled columns:", st.session_state.get("scaled_columns"))
    st.write("Encoded features:", [f for f in feature_cols if f in st.session_state.get(f"{f}_enc", [])])
    st.caption("Target properties:")
    st.write("Scaler type:", st.session_state.get("scaler_y"))
    st.write("Mean value:", y.mean())
    st.write("Standard deviation:", y.std())

    # =====================================================
    # 🧠 TASK TYPE
    # =====================================================
    task = st.radio("Task type", ["classification", "regression"])

    # =====================================================
    # 🤖 MODEL (FULL ORIGINAL)
    # =====================================================
    if task == "classification":
        model_name = st.selectbox(
            "Model",
            ["Logistic Regression", "Decision Tree", "Neural Network"]
        )
    else:
        model_name = st.selectbox(
            "Model",
            ["Linear Regression", "Decision Tree", "Neural Network"]
        )

    # =====================================================
    # ⚙️ SPLIT
    # =====================================================
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    # =====================================================
    # 🚀 TRAIN
    # =====================================================
    if st.button("🚀 Train Model"):
        try:
            # 🔴 LOADER STARTS HERE
            with st.spinner("🔄 Training model, please wait..."):
                # Optional progress bar (visual feedback)
                progress_bar = st.progress(0)

                model = get_model(task, model_name)
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
            st.session_state["X_columns"] = feature_cols

            st.success("✅ Model trained successfully!")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")

    
    # =============================
    # 📊 EVALUATION
    # =============================
    if "y_test" in st.session_state and "y_pred" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Evaluation")
        use_encoded = st.checkbox("Use encoded labels on the confusion matrix", value=False)
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
        cols = st.session_state["X_columns"]
        scaler_X = st.session_state.get("scaler")
        scaled_cols = st.session_state.get("scaled_columns", [])
        scaler_y = st.session_state.get("scaler_y")
        target_scaled = st.session_state.get("target_scaled", False)

        input_data = {}
        for col in cols:
            val = st.number_input(col, value=0.0)
            input_data[col] = val

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])

                # Apply feature scaling
                if scaler_X is not None and len(scaled_cols) > 0:
                    cols_to_scale = [c for c in scaled_cols if c in input_df.columns]
                    if len(cols_to_scale) > 0:
                        input_df[cols_to_scale] = scaler_X.transform(input_df[cols_to_scale])

                pred = model.predict(input_df)

                # Inverse target scaling
                if task == "regression" and target_scaled and scaler_y is not None:
                    pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()

                st.success(f"Prediction: {pred[0]:.4f}")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    render()
