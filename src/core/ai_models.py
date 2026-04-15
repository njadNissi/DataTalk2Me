from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import log_loss, mean_squared_error
import time
import io


def get_available_models(task):
    if task == "classification":
        return [
            "Logistic Regression",
            "Logistic Regression (L2/Ridge)",
            "SGD Classifier",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "AdaBoost",
            "K-Nearest Neighbors (KNN)",
            "Support Vector Machine (SVM)",
            "Linear SVM",
            "Gaussian Naive Bayes",
            "Neural Network (4 LAYERS)",
            "Neural Network (3 LAYERS)",
            "Custom Neural Network"
        ]
    elif task == "regression":
        return [
            "Linear Regression",
            "Ridge Regression (L2)",
            "Lasso Regression (L1)",
            "ElasticNet (L1+L2)",
            "SGD Regressor",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "AdaBoost",
            "K-Nearest Neighbors (KNN)",
            "Support Vector Machine (SVM)",
            "Linear SVM",
            "Neural Network (4 LAYERS)",
            "Neural Network (3 LAYERS)",
            "Custom Neural Network"
        ]
    else:
        return []


def get_model(task, model_name):
    if task == "classification":
        # --- Linear Models ---
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, warm_start=True)
        elif model_name == "Logistic Regression (L2/Ridge)":
            return LogisticRegression(penalty="l2", C=0.5, max_iter=1000)
        elif model_name == "SGD Classifier":
            return SGDClassifier(max_iter=1000, warm_start=True)
        
        # --- Tree & Ensemble ---
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier()
        elif model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)
        elif model_name == "Extra Trees":
            return ExtraTreesClassifier(n_estimators=100, random_state=42)
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_name == "AdaBoost":
            return AdaBoostClassifier(n_estimators=100, random_state=42)
        
        # --- Neighbors & SVM ---
        elif model_name == "K-Nearest Neighbors (KNN)":
            return KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Support Vector Machine (SVM)":
            return SVC(kernel="rbf", probability=True, random_state=42)
        elif model_name == "Linear SVM":
            return LinearSVC(max_iter=10000, random_state=42)
        
        # --- Naive Bayes ---
        elif model_name == "Gaussian Naive Bayes":
            return GaussianNB()
        
        # --- Neural Network ---
        elif model_name == "Neural Network (4 LAYERS)":
            return MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
        elif model_name == "Neural Network (3 LAYERS)": # ONLY FOR PARTIAL TRAINING (AUTO TASK)
            return MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=1000,
                early_stopping=False, # this if for partial training
                random_state=42
            )


    elif task == "regression":
        # --- Linear Models ---
        if model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "Ridge Regression (L2)":
            return Ridge(alpha=1.0, max_iter=1000, random_state=42)
        elif model_name == "Lasso Regression (L1)":
            return Lasso(alpha=1.0, max_iter=1000, random_state=42)
        elif model_name == "ElasticNet (L1+L2)":
            return ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, random_state=42)
        elif model_name == "SGD Regressor":
            return SGDRegressor(max_iter=1000, warm_start=True)
        
        # --- Tree & Ensemble ---
        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(random_state=42)
        elif model_name == "Random Forest":
            return RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)
        elif model_name == "Extra Trees":
            return ExtraTreesRegressor(n_estimators=100, random_state=42)
        elif model_name == "Gradient Boosting":
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_name == "AdaBoost":
            return AdaBoostRegressor(n_estimators=100, random_state=42)
        
        # --- Neighbors & SVM ---
        elif model_name == "K-Nearest Neighbors (KNN)":
            return KNeighborsRegressor(n_neighbors=5)
        elif model_name == "Support Vector Machine (SVM)":
            return SVR(kernel="rbf")
        elif model_name == "Linear SVM":
            return LinearSVR(max_iter=10000, random_state=42)
        
        # --- Neural Network ---
        elif model_name == "Neural Network (4 LAYERS)":
            return MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
        elif model_name == "Neural Network (3 LAYERS)": # ONLY FOR PARTIAL TRAINING (AUTO TASK) 
            return MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=1000,
                early_stopping=False, # this if for partial training
                random_state=42
            )

    return None


def build_nn_model(task, hidden_layer_sizes:list[int], activation_fnc:str, early_stopping=True, max_iter=1000):
    if task == "classification":
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation_fnc,  # ONLY ONE ACTIVATION ALLOWED
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=42
        )
    elif task == "regression":
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation_fnc,  # ONLY ONE ACTIVATION ALLOWED
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=42
        )
    else:
        return None
    
        
def get_linear_regression_formula(model, feature_names, target_name):
    intercept = model.intercept_
    coefs = model.coef_

    if isinstance(feature_names, tuple):
        feature_names = feature_names[0]

    parts = [f"{intercept:.4f}"]
    for coef, feat in zip(coefs, feature_names):
        if coef >= 0:
            parts.append(f"+ {coef:.4f} × `{feat}`")
        else:
            parts.append(f"- {abs(coef):.4f} × `{feat}`")

    return f"{target_name} = {' '.join(parts)}"

# --------------------------
# AUTO TASK TRAINING + LIVE CONVERGENCE CURVES
# --------------------------
def train_model(model, X_train, y_train, X_test, y_test, task: str, partial_train:bool, status):
    is_classification = (task == "classification")

    # Plot placeholder inside status container
    if status is not None:
        plot_placeholder = status.empty()
    else:
        plot_placeholder = st.session_state.get("plot_placeholder", st.empty())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # --------------------------
    # CASE 1: Model supports partial_fit (MLP, SGD...)
    # --------------------------
    if hasattr(model, "partial_fit") and partial_train:
        train_losses = []
        test_losses = []
        train_scores = []
        test_scores = []

        if hasattr(model, "warm_start"):
            model.warm_start = True

        epochs = 30
        for epoch in range(epochs):
            # Training step
            if is_classification:
                classes = sorted(set(y_train))
                model.partial_fit(X_train, y_train, classes=classes)
            else:
                model.partial_fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Scores
            train_scores.append(model.score(X_train, y_train))
            test_scores.append(model.score(X_test, y_test))

            # Losses
            if is_classification and hasattr(model, "predict_proba"):
                train_losses.append(log_loss(y_train, model.predict_proba(X_train)))
                test_losses.append(log_loss(y_test, model.predict_proba(X_test)))
            else:
                train_losses.append(mean_squared_error(y_train, y_pred_train))
                test_losses.append(mean_squared_error(y_test, y_pred_test))

            # Update plot
            ax[0].clear()
            ax[0].plot(train_scores, label="Train Score")
            ax[0].plot(test_scores, label="Test Score")
            ax[0].set_title("Score Convergence")
            ax[0].legend()
            ax[0].grid(True)

            ax[1].clear()
            ax[1].plot(train_losses, label="Train Loss")
            ax[1].plot(test_losses, label="Test Loss")
            ax[1].set_title("Loss Convergence")
            ax[1].legend()
            ax[1].grid(True)

            plot_placeholder.pyplot(fig)

    # --------------------------
    # CASE 2: Standard models (no partial_fit)
    # --------------------------
    else:
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        score_train = model.score(X_train, y_train)
        score_test = model.score(X_test, y_test)

        if is_classification and hasattr(model, "predict_proba"):
            loss_train = log_loss(y_train, model.predict_proba(X_train))
            loss_test = log_loss(y_test, model.predict_proba(X_test))
        else:
            loss_train = mean_squared_error(y_train, y_pred_train)
            loss_test = mean_squared_error(y_test, y_pred_test)

        # Simple static plot
        ax[0].bar(["Train", "Test"], [score_train, score_test], color=["#2E8B57", "#FF6347"])
        ax[0].set_title(f"Score: {score_test:.3f} (Test)")
        ax[0].grid(True)

        ax[1].bar(["Train", "Test"], [loss_train, loss_test], color=["#2E8B57", "#FF6347"])
        ax[1].set_title(f"Loss: {loss_test:.3f} (Test)")
        ax[1].grid(True)

        plot_placeholder.pyplot(fig)

    # Save final figure to session state
    st.session_state['inference']['conv_curves'] = fig

    return model


def draw_nn_architecture(input_size, hidden_layers, activation_fncs, output_size=1):
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis('off')

    # Get feature names from session state
    task = st.session_state['preprocessing']["task"]
    feature_names = st.session_state['preprocessing'].get("feature_names", [f"Feature {i+1}" for i in range(input_size)])
    feature_names = feature_names[:input_size]

    # Colors
    color_input = "#4285F4"
    color_hidden = "#34A853"
    color_output = "#FBBC05" if task == "classification" else "#EA4335"

    # 👇 Use visual output neurons for drawing ONLY
    layers = [input_size] + hidden_layers + [output_size]
    max_neurons = max(layers)
    v_spacing = 1.1
    h_spacing = 3.8

    for layer_idx, neuron_count in enumerate(layers):
        x = layer_idx * h_spacing
        start_y = -(neuron_count - 1) * v_spacing / 2

        for n in range(neuron_count):
            y = start_y + n * v_spacing
            circle = plt.Circle(
                (x, y), 0.35,
                color=color_input if layer_idx == 0
                else color_output if layer_idx == len(layers)-1
                else color_hidden,
                alpha=0.85
            )
            ax.add_patch(circle)

            if n == 0:
                ax.text(x, y, str(neuron_count), ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')

        # Show feature names
        if layer_idx == 0:
            for n, fname in enumerate(feature_names):
                y = start_y + n * v_spacing
                fname_short = fname[:10] + "..." if len(fname) > 10 else fname
                ax.text(x - 1.0, y, fname_short, ha='right', va='center', fontsize=8, color='#222')

        # Draw connections
        if layer_idx < len(layers) - 1:
            next_x = (layer_idx + 1) * h_spacing
            next_count = layers[layer_idx + 1]
            next_start_y = -(next_count - 1) * v_spacing / 2

            for n in range(neuron_count):
                y1 = start_y + n * v_spacing
                for m in range(next_count):
                    y2 = next_start_y + m * v_spacing
                    ax.plot([x + 0.35, next_x - 0.35], [y1, y2],
                            'gray', linewidth=0.2, alpha=0.3)

    # ==========================================================================
    # 🌟 ACTIVATION TEXT: FULLY AUTO-SCALED (width + height)
    # ==========================================================================
    if activation_fncs and len(hidden_layers) > 1:
        # Smart scaling based on plot dimensions
        total_layers = 2 + len(hidden_layers)
        dynamic_font = max(5, 13 - (total_layers * 0.6))

        mid_input_y = -(input_size - 1) * v_spacing / 2 + ((input_size - 1) / 2) * v_spacing

        for act_idx in range(len(activation_fncs)):
            x_pos = (1.5 + act_idx) * h_spacing
            act = activation_fncs[act_idx]
            pretty_act = {"logistic": "sigmoid", "relu": "ReLU", "tanh": "Tanh", "identity": "linear"}.get(act, act)

            ax.text(
                x_pos, mid_input_y,
                pretty_act,
                ha='center', va='center',
                fontsize=dynamic_font,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#444', alpha=0.8)
            )

    ax.set_xlim(-2.5, (len(layers)) * h_spacing + 0.5)
    ax.set_ylim(-max_neurons * v_spacing * 0.55, max_neurons * v_spacing * 0.55)
    ax.set_aspect('equal')

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(top=0.95, bottom=0.05)

    download_btn, update_btn = st.columns(2)
    # 📥 DOWNLOAD BUTTON (SVG format)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    with download_btn:
        st.download_button(
            label="📥 Download HQ image of your NeuralNet (SVG)",
            data=buf,
            file_name="convergence_curve.svg",
            mime="image/svg+xml"
        )
    st.session_state['inference']['nn_update_btn_placeholder'] = update_btn

    return fig