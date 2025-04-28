# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
    RepeatedStratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    matthews_corrcoef,
    average_precision_score,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.pipeline import Pipeline
import warnings
import shap
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import lime
import lime.lime_tabular
import optuna
import plotly.express as px
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from scipy import stats
import time

warnings.filterwarnings("ignore")


# Load and Prepare Data
def load_data(filepath):
    data = pd.read_csv(filepath)
    if data.empty:
        raise ValueError("Data not loaded correctly. Please check the file path.")
    return data


def preprocess_data(data):
    print("Starting data preprocessing...")

    # Check for outliers using z-score
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data_no_outliers = data[filtered_entries]
    print(f"Removed {data.shape[0] - data_no_outliers.shape[0]} outliers using z-score")

    # Check for zero values that should be NaN in medical context
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zeros:
        zero_count = (data[col] == 0).sum()
        print(f"Zeros in {col}: {zero_count}")
        # Convert zeros to NaN for these columns
        data.loc[data[col] == 0, col] = np.nan

    # Print missing value statistics
    print("\nMissing values after zero conversion:")
    print(data.isnull().sum())

    return data


# Enhanced Feature Engineering
def enhanced_feature_engineering(data):
    """
    Apply advanced feature engineering techniques
    """
    print("Applying feature engineering...")

    # BMI categories
    data["BMI_Category"] = pd.cut(
        data["BMI"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    # Age groups
    data["Age_Group"] = pd.cut(
        data["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle-aged", "Senior"]
    )

    # Feature interactions
    data["Glucose_BMI_Interaction"] = data["Glucose"] * data["BMI"]
    data["Age_Insulin_Interaction"] = data["Age"] * data["Insulin"]
    data["Glucose_to_Insulin_Ratio"] = data["Glucose"] / (
        data["Insulin"] + 1
    )  # Avoid division by zero
    data["HOMA-IR"] = (
        data["Glucose"] * data["Insulin"]
    ) / 405  # Homeostatic Model Assessment
    data["Is_Pregnant"] = (data["Pregnancies"] > 0).astype(int)
    data["Age_BP_Product"] = data["Age"] * data["BloodPressure"]
    data["BP_BMI_Product"] = data["BloodPressure"] * data["BMI"]
    data["DiabetesPedigreeFunction_Age"] = (
        data["DiabetesPedigreeFunction"] * data["Age"]
    )

    # Polynomial features for key predictors
    data["Glucose_Squared"] = data["Glucose"] ** 2
    data["BMI_Squared"] = data["BMI"] ** 2
    data["Age_Squared"] = data["Age"] ** 2

    # Logarithmic transformations
    data["Log_Glucose"] = np.log1p(data["Glucose"])
    data["Log_BMI"] = np.log1p(data["BMI"])
    data["Log_Insulin"] = np.log1p(data["Insulin"] + 1)
    data["Log_SkinThickness"] = np.log1p(data["SkinThickness"] + 1)

    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=["BMI_Category", "Age_Group"], drop_first=True)

    return data


# Advanced Feature Selection
def select_optimal_features(X_train, y_train, X_test):
    print("Performing feature selection...")
    results = {}

    # 1. Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_features = pd.Series(mi_scores, index=X_train.columns)
    mi_selected = mi_features.nlargest(15).index.tolist()

    # 2. RFECV with Random Forest
    rfecv = RFECV(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        step=1,
        cv=5,
        scoring="accuracy",
        min_features_to_select=5,
    )
    rfecv.fit(X_train, y_train)
    rfecv_selected = X_train.columns[rfecv.support_].tolist()

    # 3. Feature importance from Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    importance_threshold = np.mean(importances)  # Features above mean importance
    rf_selected = X_train.columns[importances >= importance_threshold].tolist()

    # Store results
    results["mutual_info"] = {
        "features": mi_selected,
        "X_train": X_train[mi_selected],
        "X_test": X_test[mi_selected],
    }

    results["rfecv"] = {
        "features": rfecv_selected,
        "X_train": X_train[rfecv_selected],
        "X_test": X_test[rfecv_selected],
    }

    results["rf_importance"] = {
        "features": rf_selected,
        "X_train": X_train[rf_selected],
        "X_test": X_test[rf_selected],
    }

    # Print selected features
    for method, data in results.items():
        print(
            f"\n{method.upper()} selected features ({len(data['features'])}): {data['features']}"
        )

    return results


# Address class imbalance with multiple techniques
def handle_class_imbalance(X_train, y_train):
    """
    Apply multiple resampling techniques to handle class imbalance
    """
    print("Handling class imbalance...")
    resampling_results = {}

    # 1. SMOTE with different settings
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)

    # 2. BorderlineSMOTE (often better than regular SMOTE)
    b_smote = BorderlineSMOTE(random_state=42)
    X_resampled_b_smote, y_resampled_b_smote = b_smote.fit_resample(X_train, y_train)

    # 3. ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_train, y_train)

    # 4. SMOTE + Tomek Links
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled_smote_tomek, y_resampled_smote_tomek = smote_tomek.fit_resample(
        X_train, y_train
    )

    # 5. SMOTE + ENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled_smote_enn, y_resampled_smote_enn = smote_enn.fit_resample(
        X_train, y_train
    )

    # Store results
    resampling_results["original"] = (X_train, y_train)
    resampling_results["smote"] = (X_resampled_smote, y_resampled_smote)
    resampling_results["borderline_smote"] = (X_resampled_b_smote, y_resampled_b_smote)
    resampling_results["adasyn"] = (X_resampled_adasyn, y_resampled_adasyn)
    resampling_results["smote_tomek"] = (
        X_resampled_smote_tomek,
        y_resampled_smote_tomek,
    )
    resampling_results["smote_enn"] = (X_resampled_smote_enn, y_resampled_smote_enn)

    # Print class distributions
    for method, (X_res, y_res) in resampling_results.items():
        print(f"\n{method.upper()} class distribution:")
        print(pd.Series(y_res).value_counts())
        print(f"Shape: {X_res.shape}")

    return resampling_results


def optimize_random_forest(X_train, y_train):
    # Define the objective function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

        # Create model with chosen hyperparameters
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        # Perform cross-validation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        return scores.mean()

    # Create and run study
    print("Optimizing Random Forest hyperparameters with Optuna...")
    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(objective, n_trials=50)

        # Print results
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best ROC AUC: {study.best_trial.value:.4f}")
        print(f"Best hyperparameters: {study.best_trial.params}")

        # Create best model
        best_params = study.best_trial.params
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Using default parameters instead")
        best_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": "balanced",
            "criterion": "gini",
        }
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

    return best_model, best_params


# Threshold optimization
def optimize_threshold(model, X_val, y_val):
    print("Optimizing classification threshold...")
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics for different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # Calculate a balanced score (balancing precision and recall)
        score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        scores.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_score": score,
            }
        )

    # Convert to DataFrame for analysis
    scores_df = pd.DataFrame(scores)
    best_threshold = scores_df.loc[scores_df["balanced_score"].idxmax()]["threshold"]

    print(f"Optimal threshold: {best_threshold:.2f}")

    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(scores_df["threshold"], scores_df["precision"], label="Precision")
    plt.plot(scores_df["threshold"], scores_df["recall"], label="Recall")
    plt.plot(scores_df["threshold"], scores_df["f1"], label="F1 Score")
    plt.axvline(
        x=best_threshold,
        color="r",
        linestyle="--",
        label=f"Best Threshold: {best_threshold:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Performance Metrics vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_threshold


# Comprehensive model evaluation
def comprehensive_evaluation(
    model, X_test, y_test, model_name, threshold=0.5, y_prob=None
):
    # Get probabilities if not provided
    if y_prob is None and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif y_prob is None:
        # If the model doesn't provide probabilities, use predictions as proxy
        y_prob = model.predict(X_test)

    # Apply threshold for predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n--- {model_name} Evaluation (threshold={threshold:.2f}) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name} (AP = {avg_prec:.2f})")
    plt.show()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "avg_precision": avg_prec,
        "mcc": mcc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def interpret_model_with_shap(model, X_train, X_test, feature_names):
    print("\nInterpreting model with SHAP...")

    # Create TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values - returns array of shape (n_samples, n_features, n_classes)
    shap_values = explainer.shap_values(X_test)

    # Debug print to understand the SHAP values structure
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape: {np.array(shap_values).shape}")

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Old API returns list of arrays for each class
        shap_values_pos = np.array(shap_values[1])  # Positive class SHAP values
        expected_value = explainer.expected_value[1]
    elif len(np.array(shap_values).shape) == 3:
        # New API returns 3D array (n_classes, n_samples, n_features)
        shap_values_pos = np.array(shap_values)[1]  # Positive class SHAP values
        expected_value = explainer.expected_value[1]
    else:
        # Single output case
        shap_values_pos = np.array(shap_values)
        expected_value = explainer.expected_value

    # Transpose if needed to get (n_samples, n_features)
    if shap_values_pos.shape[0] == len(feature_names):
        shap_values_pos = shap_values_pos.T

    print(f"Final SHAP values shape for plotting: {shap_values_pos.shape}")
    print(f"X_test shape: {X_test.shape}")

    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    try:
        shap.summary_plot(
            shap_values_pos, X_test, feature_names=feature_names, show=False
        )
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not create summary plot: {e}")

    # 2. Beeswarm Plot
    plt.figure(figsize=(12, 8))
    try:
        shap.summary_plot(
            shap_values_pos,
            X_test,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
        )
        plt.title("SHAP Values (Beeswarm-style)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not create beeswarm plot: {e}")

    # 3. Waterfall Plot for first instance
    plt.figure(figsize=(12, 8))
    try:
        # Create proper Explanation object
        explanation = shap.Explanation(
            values=shap_values_pos[0],  # First instance's SHAP values
            base_values=expected_value,
            data=X_test.iloc[0].values,
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, max_display=15)
    except Exception as e:
        print(f"Could not create waterfall plot: {e}")
        # Fallback to simple bar plot
        plt.figure(figsize=(12, 8))
        if len(shap_values_pos.shape) == 2:
            # For multiple features, show mean absolute impact
            mean_shap = np.abs(shap_values_pos).mean(axis=0)
            plt.barh(feature_names, mean_shap)
            plt.title("Mean Absolute SHAP Values")
            plt.xlabel("Mean |SHAP Value| (impact on model output)")
        else:
            # For single feature case
            plt.barh(feature_names, np.abs(shap_values_pos))
            plt.title("Absolute SHAP Values")
            plt.xlabel("|SHAP Value| (impact on model output)")
        plt.ylabel("Feature")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    return shap_values_pos


# LIME for local interpretation
def interpret_with_lime(model, X_train, X_test, y_test, feature_names, num_instances=3):
    print("\nInterpreting predictions with LIME...")
    # Create explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=["No Diabetes", "Diabetes"],
        mode="classification",
    )

    # Explain some test instances
    for i in range(num_instances):
        # Pick a random instance or interesting cases
        if i == 0:
            # Find a false positive
            y_pred = model.predict(X_test)
            false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
            if len(false_positives) > 0:
                idx = false_positives[0]
            else:
                idx = np.random.randint(0, len(X_test))
        elif i == 1:
            # Find a false negative
            y_pred = model.predict(X_test)
            false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]
            if len(false_negatives) > 0:
                idx = false_negatives[0]
            else:
                idx = np.random.randint(0, len(X_test))
        else:
            # Random instance
            idx = np.random.randint(0, len(X_test))

        # Get explanation
        exp = explainer.explain_instance(
            X_test.iloc[idx].values, model.predict_proba, num_features=10
        )

        print(f"\nLIME Explanation for Test Instance {i + 1}:")
        print(f"True label: {'Diabetes' if y_test.iloc[idx] == 1 else 'No Diabetes'}")
        y_pred = model.predict(X_test.iloc[idx].values.reshape(1, -1))[0]
        print(f"Predicted label: {'Diabetes' if y_pred == 1 else 'No Diabetes'}")
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test.iloc[idx].values.reshape(1, -1))[0, 1]
            print(f"Prediction probability: {prob:.4f}")

        # Plot explanation
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title(f"LIME Explanation for Test Instance {i + 1}")
        plt.tight_layout()
        plt.show()


# Learning curves
def plot_learning_curves(model, X_train, y_train):
    print("\nGenerating learning curves...")
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
        n_jobs=-1,
    )

    # Calculate mean and standard deviation for training scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")

    # Add fills to represent one standard deviation
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g"
    )

    # Add labels and title
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy Score")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


# Calibration curve
def plot_calibration_curve(model, X_test, y_test, model_name):
    from sklearn.calibration import calibration_curve

    print("\nGenerating calibration curve...")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

        plt.figure(figsize=(10, 6))
        plt.plot(prob_pred, prob_true, "s-", label=model_name)
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend(loc="best")
        plt.grid()
        plt.show()

        # Check if calibration is needed
        from sklearn.metrics import brier_score_loss

        brier_score = brier_score_loss(y_test, y_prob)
        print(f"Brier score: {brier_score:.4f} (Lower is better, 0 is perfect)")


# Feature importance visualization
def plot_feature_importance(model, feature_names):
    print("\nPlotting feature importance...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(
        range(len(feature_names)), [feature_names[i] for i in indices], rotation=90
    )
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()


# Main execution
def main(filepath):
    # Load data
    print("Loading data...")
    data = load_data(filepath)

    # Preprocess data
    data = preprocess_data(data)

    # Apply feature engineering
    data = enhanced_feature_engineering(data)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Split into train, validation, and test sets
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=0.25, random_state=42
    )
    print(
        f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}"
    )

    # Handle missing values with KNN imputation
    print("\nHandling missing values...")
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imputed), columns=X_val_imputed.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed), columns=X_test_imputed.columns
    )

    # Feature selection
    print("\nPerforming feature selection...")
    feature_selection_results = select_optimal_features(
        X_train_scaled, y_train, X_test_scaled
    )

    # Use feature set from RFECV for further analysis
    X_train_selected = feature_selection_results["rfecv"]["X_train"]
    X_val_selected = X_val_scaled[feature_selection_results["rfecv"]["features"]]
    X_test_selected = feature_selection_results["rfecv"]["X_test"]
    selected_features = feature_selection_results["rfecv"]["features"]
    print(f"\nSelected features for modeling: {selected_features}")

    # Handle class imbalance
    print("\nHandling class imbalance...")
    resampling_results = handle_class_imbalance(X_train_selected, y_train)

    # Test different resampling techniques with base random forest
    print("\nComparing resampling techniques...")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    resampling_scores = {}

    for method_name, (X_resampled, y_resampled) in resampling_results.items():
        # Train model with resampled data
        base_model.fit(X_resampled, y_resampled)

        # Evaluate on validation set
        y_val_pred = base_model.predict(X_val_selected)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)

        # Store results
        resampling_scores[method_name] = {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision,
        }

        print(
            f"{method_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}"
        )

    # Identify best resampling method based on F1 score
    best_resampling = max(resampling_scores.items(), key=lambda x: x[1]["f1"])[0]
    print(f"\nBest resampling method based on F1 score: {best_resampling}")

    # Use best resampling method
    X_train_resampled, y_train_resampled = resampling_results[best_resampling]

    # Optimize random forest hyperparameters
    print("\nOptimizing random forest model...")
    best_rf, rf_params = optimize_random_forest(X_train_resampled, y_train_resampled)

    # Train model with optimal hyperparameters
    print("\nTraining model with optimal hyperparameters...")
    best_rf.fit(X_train_resampled, y_train_resampled)

    # Optimize classification threshold on validation set
    best_threshold = optimize_threshold(best_rf, X_val_selected, y_val)

    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_results = comprehensive_evaluation(
        best_rf,
        X_test_selected,
        y_test,
        "Optimized Random Forest",
        threshold=best_threshold,
    )

    # Compare with default threshold
    default_results = comprehensive_evaluation(
        best_rf,
        X_test_selected,
        y_test,
        "Random Forest (Default Threshold)",
        threshold=0.5,
    )

    # Model calibration analysis
    plot_calibration_curve(best_rf, X_test_selected, y_test, "Optimized Random Forest")

    # Try model calibration
    print("\nCalibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(best_rf, cv="prefit")
    calibrated_model.fit(X_val_selected, y_val)

    # Evaluate calibrated model
    calibrated_results = comprehensive_evaluation(
        calibrated_model,
        X_test_selected,
        y_test,
        "Calibrated Random Forest",
        threshold=0.5,
    )

    # Compare all three variants
    print("\nComparison of model variants:")
    comparison = pd.DataFrame(
        {
            "Optimized (Custom Threshold)": test_results,
            "Default Threshold": default_results,
            "Calibrated": calibrated_results,
        }
    )

    print(comparison.loc[["accuracy", "precision", "recall", "f1", "auc"]])

    # Model interpretation
    print("\nInterpreting model...")
    shap_values = interpret_model_with_shap(
        best_rf, X_train_selected, X_test_selected, selected_features
    )
    interpret_with_lime(
        best_rf, X_train_selected, X_test_selected, y_test, selected_features
    )

    # Learning curves
    plot_learning_curves(best_rf, X_train_selected, y_train)

    # Feature importance visualization
    plot_feature_importance(best_rf, selected_features)

    # Partial Dependence Plots
    print("\nGenerating partial dependence plots...")
    top_features = [
        selected_features[i] for i in np.argsort(best_rf.feature_importances_)[-3:]
    ]  # Top 3 features
    PartialDependenceDisplay.from_estimator(
        best_rf, X_test_selected, features=top_features, kind="both"
    )
    plt.tight_layout()
    plt.show()

    # Save model and preprocessing components
    print("\nSaving model and preprocessing components...")
    model_artifacts = {
        "model": best_rf,
        "imputer": imputer,
        "scaler": scaler,
        "selected_features": selected_features,
        "best_threshold": best_threshold,
        "calibrated_model": calibrated_model,
        "feature_engineering_func": enhanced_feature_engineering,
    }

    joblib.dump(model_artifacts, f"enhanced_random_forest_model.pkl")
    print(f"Model saved as 'enhanced_random_forest_model.pkl'")

    # Create a unified prediction function
    def predict_diabetes(data_dict, threshold=best_threshold, use_calibration=False):
        # Convert input to DataFrame
        data = pd.DataFrame([data_dict])

        # Apply feature engineering
        data = model_artifacts["feature_engineering_func"](data)

        # Handle missing values
        data_imputed = pd.DataFrame(
            model_artifacts["imputer"].transform(data), columns=data.columns
        )

        # Scale features
        data_scaled = pd.DataFrame(
            model_artifacts["scaler"].transform(data_imputed),
            columns=data_imputed.columns,
        )

        # Select features
        data_selected = data_scaled[model_artifacts["selected_features"]]

        # Choose model for prediction
        if use_calibration:
            model = model_artifacts["calibrated_model"]
        else:
            model = model_artifacts["model"]

        # Get probability
        probability = model.predict_proba(data_selected)[0, 1]

        # Apply threshold
        prediction = 1 if probability >= threshold else 0

        return prediction, probability

    # Example prediction
    example_patient = {
        "Pregnancies": 6,
        "Glucose": 140,
        "BloodPressure": 80,
        "SkinThickness": 30,
        "Insulin": 200,
        "BMI": 32.1,
        "DiabetesPedigreeFunction": 0.63,
        "Age": 50,
    }

    pred, prob = predict_diabetes(example_patient)
    print("\nExample prediction:")
    print(f"Patient data: {example_patient}")
    print(f"Prediction: {'Diabetes' if pred == 1 else 'No Diabetes'}")
    print(f"Probability: {prob:.4f}")

    # PCA visualization of data
    print("\nPCA visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_selected)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Diabetes Status")
    plt.title("PCA Visualization of Test Set")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.grid()
    plt.show()

    # Return the best model and results
    return best_rf, test_results, selected_features


# Run the main function with your data file path
if __name__ == "__main__":
    filepath = os.path.join(os.path.expanduser("~"), "Desktop", "diabetes.csv")
    best_model, results, selected_features = main(filepath)
    print("\nEnhanced Random Forest Model Complete!")
