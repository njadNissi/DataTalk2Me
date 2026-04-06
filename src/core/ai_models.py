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
        elif model_name == "Neural Network (MLP)":
            return MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=1000,
                early_stopping=True,
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
        elif model_name == "Neural Network (MLP)":
            return MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )

    return None


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
            "Neural Network (MLP)"
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
            "Neural Network (MLP)"
        ]
    else:
        return []

        
