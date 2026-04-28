import typing as t
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
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
from sklearn.metrics import log_loss, mean_squared_error

# --------------------------
# Constants
# --------------------------
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_ITER = 1000
NN_DEFAULT_LAYERS_4 = (256, 128, 64, 32)
NN_DEFAULT_LAYERS_3 = (64, 32, 16)
PLOT_FIGSIZE = (12, 4)
NN_PLOT_FIGSIZE = (7, 4.5)
NN_V_SPACING = 1.1
NN_H_SPACING = 3.8
NN_COLORS = {
    "input": "#4285F4",
    "hidden": "#34A853",
    "classification_output": "#FBBC05",
    "regression_output": "#EA4335"
}

# --------------------------
# Type Hints
# --------------------------
ModelType = t.Union[
    LogisticRegression, LinearRegression, DecisionTreeClassifier,
    DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor,
    MLPClassifier, MLPRegressor, SVC, SVR
]

# --------------------------
# Model Configuration
# --------------------------
def get_available_models(task: str) -> t.List[str]:
    """
    Get list of available models for a given task type.
    
    Args:
        task: Task type ("classification" or "regression")
    
    Returns:
        List of model names
    """
    classification_models = [
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
    
    regression_models = [
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
    
    if task == "classification":
        return classification_models
    elif task == "regression":
        return regression_models
    else:
        st.warning(f"⚠️ Unknown task type: {task}")
        return []

def get_model(task: str, model_name: str) -> t.Optional[ModelType]:
    """
    Get initialized model instance for the given task and model name.
    
    Args:
        task: Task type ("classification" or "regression")
        model_name: Name of the model to initialize
    
    Returns:
        Initialized model instance or None if not found
    """
    # Classification models
    if task == "classification":
        model_configs = {
            "Logistic Regression": LogisticRegression(
                max_iter=DEFAULT_MAX_ITER,
                warm_start=True,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Logistic Regression (L2/Ridge)": LogisticRegression(
                penalty="l2",
                C=0.5,
                max_iter=DEFAULT_MAX_ITER,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "SGD Classifier": SGDClassifier(
                max_iter=DEFAULT_MAX_ITER,
                warm_start=True,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=DEFAULT_N_ESTIMATORS,
                warm_start=True,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=DEFAULT_N_ESTIMATORS,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=DEFAULT_N_ESTIMATORS,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=DEFAULT_N_ESTIMATORS,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine (SVM)": SVC(
                kernel="rbf",
                probability=True,
                max_iter=DEFAULT_MAX_ITER,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Linear SVM": LinearSVC(
                max_iter=DEFAULT_MAX_ITER * 10,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Gaussian Naive Bayes": GaussianNB(),
            "Neural Network (4 LAYERS)": MLPClassifier(
                hidden_layer_sizes=NN_DEFAULT_LAYERS_4,
                max_iter=DEFAULT_MAX_ITER,
                early_stopping=True,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Neural Network (3 LAYERS)": MLPClassifier(
                hidden_layer_sizes=NN_DEFAULT_LAYERS_3,
                max_iter=DEFAULT_MAX_ITER,
                early_stopping=False,  # For partial training
                random_state=DEFAULT_RANDOM_STATE
            )
        }
    
    # Regression models
    elif task == "regression":
        model_configs = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression (L2)": Ridge(
                alpha=1.0,
                max_iter=DEFAULT_MAX_ITER,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Lasso Regression (L1)": Lasso(
                alpha=1.0,
                max_iter=DEFAULT_MAX_ITER,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "ElasticNet (L1+L2)": ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                max_iter=DEFAULT_MAX_ITER,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "SGD Regressor": SGDRegressor(
                max_iter=DEFAULT_MAX_ITER,
                warm_start=True,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Decision Tree": DecisionTreeRegressor(
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=DEFAULT_N_ESTIMATORS,
                warm_start=True,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=DEFAULT_N_ESTIMATORS,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=DEFAULT_N_ESTIMATORS,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "AdaBoost": AdaBoostRegressor(
                n_estimators=DEFAULT_N_ESTIMATORS,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Machine (SVM)": SVR(
                kernel="rbf",
                max_iter=DEFAULT_MAX_ITER
            ),
            "Linear SVM": LinearSVR(
                max_iter=DEFAULT_MAX_ITER * 10,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Neural Network (4 LAYERS)": MLPRegressor(
                hidden_layer_sizes=NN_DEFAULT_LAYERS_4,
                max_iter=DEFAULT_MAX_ITER,
                early_stopping=True,
                random_state=DEFAULT_RANDOM_STATE
            ),
            "Neural Network (3 LAYERS)": MLPRegressor(
                hidden_layer_sizes=NN_DEFAULT_LAYERS_3,
                max_iter=DEFAULT_MAX_ITER,
                early_stopping=False,  # For partial training
                random_state=DEFAULT_RANDOM_STATE
            )
        }
    
    else:
        st.error(f"❌ Unsupported task type: {task}")
        return None

    # Return model or show error if not found
    if model_name not in model_configs:
        st.error(f"❌ Model '{model_name}' not available for task: {task}")
        return None
    
    return model_configs[model_name]

def build_nn_model(
    task: str,
    hidden_layer_sizes: t.List[int],
    activation_fnc: str = "relu",
    early_stopping: bool = True,
    max_iter: int = DEFAULT_MAX_ITER
) -> t.Optional[ModelType]:
    """
    Build a custom neural network model.
    
    Args:
        task: Task type ("classification" or "regression")
        hidden_layer_sizes: List of neurons per hidden layer
        activation_fnc: Activation function for hidden layers
        early_stopping: Whether to enable early stopping
        max_iter: Maximum number of iterations
    
    Returns:
        Initialized MLP model or None
    """
    try:
        if not hidden_layer_sizes:
            st.warning("⚠️ At least one hidden layer is required")
            return None
        
        common_kwargs = {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": activation_fnc,
            "max_iter": max_iter,
            "early_stopping": early_stopping,
            "random_state": DEFAULT_RANDOM_STATE
        }

        if task == "classification":
            return MLPClassifier(**common_kwargs)
        elif task == "regression":
            return MLPRegressor(**common_kwargs)
        else:
            st.error(f"❌ Unsupported task type for NN: {task}")
            return None
            
    except Exception as e:
        st.error(f"❌ Failed to build neural network: {str(e)}")
        return None

# --------------------------
# Training Functions
# --------------------------
def train_model(
    model: ModelType,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    partial_train: bool,
    status: st.status = None
) -> ModelType:
    """
    Train model with support for partial training and live convergence plots.
    
    Args:
        model: Model instance to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        task: Task type ("classification" or "regression")
        partial_train: Whether to use partial training
        status: Streamlit status container for progress updates
    
    Returns:
        Trained model instance
    """
    is_classification = (task == "classification")
        # Plot placeholder inside status container
    if status is not None:
        plot_placeholder = status.empty()
    else:
        plot_placeholder = st.session_state['inference'].get("plot_placeholder", st.empty())
    
    # Create figure for convergence plots
    fig, ax = plt.subplots(1, 2, figsize=PLOT_FIGSIZE)
    
    # Partial training (for models that support it)
    if hasattr(model, "partial_fit") and partial_train:
        model.early_stopping = False
        if hasattr(model, "warm_start"):
            model.warm_start = True
        
        train_losses, test_losses = [], []
        train_scores, test_scores = [], []
        epochs = 30

        for epoch in range(epochs):
            # Partial fit step
            if is_classification:
                classes = sorted(set(y_train))
                model.partial_fit(X_train, y_train, classes=classes)
            else:
                model.partial_fit(X_train, y_train)

            # Calculate metrics
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Calculate loss
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            if is_classification and hasattr(model, "predict_proba"):
                train_loss = log_loss(y_train, model.predict_proba(X_train))
                test_loss = log_loss(y_test, model.predict_proba(X_test))
            else:
                train_loss = mean_squared_error(y_train, y_pred_train)
                test_loss = mean_squared_error(y_test, y_pred_test)

            # Store metrics
            train_scores.append(train_score)
            test_scores.append(test_score)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # Update plots
            ax[0].clear()
            ax[0].plot(train_scores, label="Train Score", linewidth=2)
            ax[0].plot(test_scores, label="Test Score", linewidth=2)
            ax[0].set_title("Score Convergence", fontsize=10)
            ax[0].legend(fontsize=8)
            ax[0].grid(True, alpha=0.3)
            
            ax[1].clear()
            ax[1].plot(train_losses, label="Train Loss", linewidth=2)
            ax[1].plot(test_losses, label="Test Loss", linewidth=2)
            ax[1].set_title("Loss Convergence", fontsize=10)
            ax[1].legend(fontsize=8)
            ax[1].grid(True, alpha=0.3)
            
            plot_placeholder.pyplot(fig)
            time.sleep(0.05)  # Small delay for visualization

    # Standard training (one-shot fit)
    else:
        model.fit(X_train, y_train)
        
        # Calculate final metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        if is_classification and hasattr(model, "predict_proba"):
            train_loss = log_loss(y_train, model.predict_proba(X_train))
            test_loss = log_loss(y_test, model.predict_proba(X_test))
        else:
            train_loss = mean_squared_error(y_train, y_pred_train)
            test_loss = mean_squared_error(y_test, y_pred_test)

        # Static bar plot for one-shot training
        ax[0].bar(
            ["Train", "Test"],
            [train_score, test_score],
            color=["#2E8B57", "#FF6347"],
            alpha=0.8
        )
        ax[0].set_title(f"Score: {test_score:.3f} (Test)", fontsize=10)
        ax[0].grid(True, alpha=0.3, axis="y")
        
        ax[1].bar(
            ["Train", "Test"],
            [train_loss, test_loss],
            color=["#2E8B57", "#FF6347"],
            alpha=0.8
        )
        ax[1].set_title(f"Loss: {test_loss:.3f} (Test)", fontsize=10)
        ax[1].grid(True, alpha=0.3, axis="y")
        
        plot_placeholder.pyplot(fig)


    # Save figure to session state
    st.session_state["inference"]["conv_curves"] = fig
    
    return model

# --------------------------
# Visualization & Utilities
# --------------------------
def get_linear_regression_formula(
    model: LinearRegression,
    feature_names: t.Union[t.List[str], t.Tuple[t.List[str]]],
    target_name: str
) -> str:
    """
    Generate human-readable linear regression formula.
    
    Args:
        model: Trained LinearRegression model
        feature_names: List of feature names
        target_name: Target variable name
    
    Returns:
        Formatted regression formula string
    """
    try:
        intercept = model.intercept_
        coefs = model.coef_

        # Unpack feature names if in tuple
        if isinstance(feature_names, tuple):
            feature_names = feature_names[0]

        # Build formula parts
        parts = [f"{intercept:.4f}"]
        for coef, feat in zip(coefs, feature_names):
            if coef >= 0:
                parts.append(f"+ {coef:.4f} × `{feat}`")
            else:
                parts.append(f"- {abs(coef):.4f} × `{feat}`")

        return f"`{target_name}` = {' '.join(parts)}"
    
    except Exception as e:
        st.warning(f"⚠️ Could not generate formula: {str(e)}")
        return f"❌ Error generating formula: {str(e)}"

def draw_nn_architecture(
    input_size: int,
    hidden_layers: t.List[int],
    activation_fncs: t.List[str],
    output_size: int = 1
) -> plt.Figure:
    """
    Draw neural network architecture diagram.
    
    Args:
        input_size: Number of input features
        hidden_layers: List of neurons per hidden layer
        activation_fncs: List of activation functions between layers
        output_size: Number of output neurons
    
    Returns:
        Matplotlib figure with NN architecture
    """
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=NN_PLOT_FIGSIZE)
    ax.axis("off")

    # Get feature names and task type
    task = st.session_state["preprocessing"]["task"]
    feature_names = st.session_state["preprocessing"].get(
        "feature_names",
        [f"Feature {i+1}" for i in range(input_size)]
    )
    feature_names = feature_names[:input_size]

    # Define layer structure
    layers = [input_size] + hidden_layers + [output_size]
    max_neurons = max(layers)
    
    # Draw neurons and connections
    for layer_idx, neuron_count in enumerate(layers):
        x = layer_idx * NN_H_SPACING
        start_y = -(neuron_count - 1) * NN_V_SPACING / 2

        # Draw neurons
        for n in range(neuron_count):
            y = start_y + n * NN_V_SPACING
            
            # Determine neuron color
            if layer_idx == 0:
                color = NN_COLORS["input"]
            elif layer_idx == len(layers) - 1:
                color = NN_COLORS["classification_output"] if task == "classification" else NN_COLORS["regression_output"]
            else:
                color = NN_COLORS["hidden"]

            # Draw neuron circle
            circle = plt.Circle(
                (x, y), 0.35,
                color=color,
                alpha=0.85
            )
            ax.add_patch(circle)

            # Add neuron count label (first neuron only)
            if n == 0:
                ax.text(
                    x, y, str(neuron_count),
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold"
                )

        # Draw feature names (input layer)
        if layer_idx == 0:
            for n, fname in enumerate(feature_names):
                y = start_y + n * NN_V_SPACING
                fname_short = fname[:10] + "..." if len(fname) > 10 else fname
                ax.text(
                    x - 1.0, y, fname_short,
                    ha="right", va="center", fontsize=8, color="#222"
                )

        # Draw connections between layers
        if layer_idx < len(layers) - 1:
            next_x = (layer_idx + 1) * NN_H_SPACING
            next_count = layers[layer_idx + 1]
            next_start_y = -(next_count - 1) * NN_V_SPACING / 2

            for n in range(neuron_count):
                y1 = start_y + n * NN_V_SPACING
                for m in range(next_count):
                    y2 = next_start_y + m * NN_V_SPACING
                    ax.plot(
                        [x + 0.35, next_x - 0.35],
                        [y1, y2],
                        "gray", linewidth=0.2, alpha=0.3
                    )

    # Add activation function labels
    if activation_fncs and len(hidden_layers) > 1:
        total_layers = 2 + len(hidden_layers)
        dynamic_font = max(5, 13 - (total_layers * 0.6))
        mid_input_y = -(input_size - 1) * NN_V_SPACING / 2 + ((input_size - 1) / 2) * NN_V_SPACING

        activation_mapping = {
            "logistic": "sigmoid",
            "relu": "ReLU",
            "tanh": "Tanh",
            "identity": "linear"
        }

        for act_idx, act in enumerate(activation_fncs):
            x_pos = (1.5 + act_idx) * NN_H_SPACING
            pretty_act = activation_mapping.get(act, act)
            
            ax.text(
                x_pos, mid_input_y,
                pretty_act,
                ha="center", va="center",
                fontsize=dynamic_font,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#444", alpha=0.8)
            )

    # Set plot limits and aspect ratio
    ax.set_xlim(-2.5, (len(layers)) * NN_H_SPACING + 0.5)
    ax.set_ylim(-max_neurons * NN_V_SPACING * 0.55, max_neurons * NN_V_SPACING * 0.55)
    ax.set_aspect("equal")

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(top=0.95, bottom=0.05)

    # Add download button
    download_btn, update_btn = st.columns(2)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    
    with download_btn:
        st.download_button(
            label="📥 Download Neural Network Diagram (SVG)",
            data=buf,
            file_name="neural_network_architecture.svg",
            mime="image/svg+xml",
            key="nn_diagram_download"
        )
    
    # Store update button placeholder in session state
    st.session_state["inference"]["nn_update_btn_placeholder"] = update_btn

    return fig

def conv_curves(download_btn_key: str) -> t.Tuple[io.BytesIO, plt.Figure]:
    """
    Create download button for convergence curve plot.
    
    Args:
        download_btn_key: Unique key for the download button
    
    Returns:
        Tuple of (BytesIO buffer, figure)
    """
    try:
        fig = st.session_state["inference"]["conv_curves"]
        buf = io.BytesIO()
        
        fig.savefig(buf, format="svg", bbox_inches="tight")
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Convergence Curve (SVG)",
            data=buf,
            file_name="convergence_curve.svg",
            mime="image/svg+xml",
            key=download_btn_key
        )
        
        return buf, fig
    
    except Exception as e:
        st.warning(f"⚠️ Could not prepare download: {str(e)}")
        return io.BytesIO(), plt.Figure()