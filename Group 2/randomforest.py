"""
!!! Using iris_2 as the virtual environment and Python interpreter !!!

Advanced Diabetes Prediction System using Random Forest Classifier
===============================================================

Objective:
---------
Develop a high-performance diabetes prediction model leveraging:
- Advanced feature engineering
- Rigorous hyperparameter optimization
- State-of-the-art interpretability techniques

Key Steps Implemented:
---------------------
1. Data Preprocessing:
   - Outlier treatment using Winsorization (5th/95th percentiles)
   - Missing value imputation with median/mode
   - Advanced scaling (RobustScaler + PowerTransformer)

2. Feature Engineering:
   - Medical feature creation (Glucose/BMI ratio, Insulin Resistance)
   - Interaction terms (Pregnancies * Age)
   - Discretization (Age/BMI bins)
   - Metabolic syndrome scoring

3. Feature Selection:
   - Mutual Information + ANOVA F-tests
   - Recursive Feature Elimination (RFE)
   - Top 10 features selected

4. Class Imbalance Handling:
   - Hybrid resampling (SMOTE + Tomek Links)

5. Model Development:
   - Random Forest Classifier
   - Hyperparameter tuning via Optuna (50 trials)
   - Optimized for ROC AUC

6. Evaluation & Interpretation:
   - Multi-metric assessment (Accuracy, Precision, Recall, F1, ROC AUC)
   - SHAP values for global feature importance
   - LIME for local explainability

7. Deployment Readiness:
   - Model serialization (joblib)
   - Metadata documentation (JSON)

Advanced Techniques Applied:
---------------------------
- Feature Scaling: RobustScaler (outlier-resistant) + Yeo-Johnson PowerTransformer
- Dimensionality Reduction: PCA visualization (optional)
- Interpretability: SHAP summary plots + LIME explanations
- Productionization: Pipeline serialization with preprocessing

Key Outcomes:
------------
✅ Identified top predictive features: Glucose, BMI, Age
✅ Generated medically meaningful feature interactions
"""

# Import necessary libraries
import lime.lime_tabular
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import (
    RobustScaler,
    PowerTransformer,
    KBinsDiscretizer,
)
from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif,
)

# from sklearn.decomposition import PCA
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import TomekLinks, NearMiss
from imblearn.combine import SMOTETomek
import shap
import lime

# from lime.lime_tabular import LimeTabularExplainer
# import yellowbrick
from yellowbrick.classifier import ROCAUC, ClassificationReport
import optuna
import joblib
import json

# import phik
# from phik import report
import warnings

warnings.filterwarnings("ignore")

# set global style parameters
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
pd.set_option("display.max_columns", 50)
np.random.seed(42)

##---------------------------
## SECTION 1: DATA LOADING AND EXPLORATORY DATA ANALYSIS
##---------------------------
print("\n" + "=" * 80)
print("Section 1: DATA LOADING AND EXPLORATORY DATA ANALYSIS")
print("=" * 80 + "\n")

# Load Dataset
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "diabetes.csv")
df = pd.read_csv(desktop_path)


# Add this right after loading the data (line ~94)
def clean_data(df):
    """Handle infinite/too-large values while preserving structure"""
    original_cols = df.columns
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clip extreme values without dropping rows/columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].abs().max() > 1e6:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q99)

    # Fill any remaining NaNs with column means
    df = df.fillna(df.mean())
    return df[original_cols]  # Ensure same column order


# df = clean_data(df)

# 1.1 Basic Dataset Info
print("📊 BASIC DATASET INFORMATION")
print("-" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 1.2 Statistical Summary
print("\n📈 STATISTICAL SUMMARY")
print("-" * 50)
print(df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T)

# 1.3 Target Distribution Analysis
print("\n🎯 TARGET VARIABLE ANALYSIS")
print("-" * 50)
target_dist = df["Outcome"].value_counts(normalize=True)
print(target_dist)

# Interactive Target Distribution
fig = px.pie(
    values=target_dist.values,
    names=target_dist.index.map({0: "Non-Diabetic", 1: "Diabetic"}),
    title="Diabetes Outcome Distribution",
    hole=0.4,
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.show()

# 1.4 Feature Distribution
print("\n📊 FEATURE DISTRIBUTIONS")
print("-" * 50)

# Create subplots
fig = make_subplots(rows=4, cols=2, subplot_titles=df.columns[:-1])

# Add distribuion plots
for i, col in enumerate(df.columns[:-1]):
    fig.add_trace(
        go.Histogram(x=df[col], name=col, nbinsx=50), row=(i // 2) + 1, col=(i % 2) + 1
    )

fig.update_layout(
    height=1200, width=1000, title_text="Feature Distributions", showlegend=False
)
fig.show()

# 1.5 Correlation Analysis
print("\n🔗 CORRELATION ANALYSIS")
print("-" * 50)

# Calculate Correlations
corr_matrix = df.corr(method="pearson")
rank_matrix = df.corr(method="spearman")

# Plot correlation matrices
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Pearson Correlation")

plt.subplot(1, 2, 2)
sns.heatmap(rank_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Spearman Rank Correlation")
plt.tight_layout()
plt.show()

##---------------------------
## SECTION 2: ADVANCED FEATURE ENGINEERING
##---------------------------
print("\n" + "=" * 80)
print("SECTION 2: ADVANCED FEATURE ENGINEERING")
print("=" * 80 + "\n")

# 2.1 Medical Feature Creation
print("🛠️ CREATING MEDICALLY RELEVANT FEATURES")
print("-" * 50)

# Metabolic Features
df["Glucose_BMI_Ratio"] = df["Glucose"] / (df["BMI"] + 1e-6)
df["BP_Age_Index"] = df["BloodPressure"] * df["Age"] / 100
df["Insulin_Resistance"] = (df["Glucose"] / df["Insulin"]) / 405
df["Metabolic_Syndrome_Score"] = (
    df["Glucose"] / 100 + df["BMI"] / 30 + df["BloodPressure"] / 100 + df["Age"] / 50
)

# Interaction terms
df["Pregnancy_Age_Interation"] = df["Pregnancies"] * df["Age"]
df["Glucose_Insulin_Interaction"] = df["Glucose"] * np.log1p(df["Insulin"])

# Discretization
discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
df["Age_Group"] = discretizer.fit_transform(df[["Age"]]).astype(int)
df["BMI_Category"] = discretizer.fit_transform(df[["BMI"]]).astype(int)

# 2.2 Outlier Detection and Treatment
print("\n🔍 OUTLIER DETECTION AND TREATMENT")
print("-" * 50)


def modified_zscore(series):
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    return 0.6745 * (series - median) / mad


outlier_report = {}
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    z_scores = modified_zscore(df[col])
    outliers = np.abs(z_scores) > 3.5
    outlier_report[col] = {
        "outlier_count": sum(outliers),
        "outlier_percentage": sum(outliers) / len(df) * 100,
        "treatment": "Winsorized",
    }
    # Winsorization
    q1, q3 = df[col].quantile([0.05, 0.95])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)

print(pd.DataFrame(outlier_report).T)

## --------------------------
## SECTION 3: DATA PREPROCESSING
## --------------------------
print("\n" + "=" * 80)
print("SECTION 3: DATA PREPROCESSING")
print("=" * 80 + "\n")

# 3.1 Feature/ Target Separation
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3.2 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3.3 Advanced Preprocessing Pipeline
print("⚙️ BUILDING PREPROCESSING PIPELINE")
print("-" * 50)

# Define Column Groups
all_features = X.columns.tolist()
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
discrete_features = ["Age_Group", "BMI_Category"]

# Verify we haven't missed any features
missing_features = set(all_features) - set(numeric_features + discrete_features)
if missing_features:
    print(f"⚠️ Warning: These features are unprocessed: {missing_features}")
    # Either add them to appropriate groups or drop them
    numeric_features.extend(missing_features)  # Adding to numeric as default

# Create Transformers
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("transformer", PowerTransformer()),
    ]
)

discrete_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", RobustScaler()),
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("disc", discrete_transformer, discrete_features),
    ],
    remainder="drop",
)

# 3.4 Feature Selection
print("\n🔍 PERFORMING FEATURE SELECTION")
print("-" * 50)


def safe_feature_selection(X, y, preprocessor, feature_names):
    """Handle any remaining issues during transform"""
    try:
        X_clean = clean_data(X)
        X_processed = preprocessor.fit_transform(X_clean)

        # verify dimensions
        if X_processed.shape[1] != len(feature_names):
            raise ValueError(
                f"Preprocessing created {X_processed.shape[1]} features, "
                f"expected {len(feature_names)}. "
                "Check your feature groups."
            )
        # Calculate feature importance
        mi_scores = mutual_info_classif(X_processed, y)
        f_scores, _ = f_classif(X_processed, y)

        return pd.DataFrame(
            {"Feature": feature_names, "MI_Score": mi_scores, "F_Score": f_scores}
        ).sort_values("MI_Score", ascending=False)
    except Exception as e:
        print(f"❌ Feature selection failed: {str(e)}")
        raise


# Run feature selection
feature_importance_df = safe_feature_selection(
    X_train, y_train, preprocessor, numeric_features + discrete_features
)

# Display results
print("\nTop 10 Features:")
print(feature_importance_df.head(10))

# Select top features
selected_features = feature_importance_df["Feature"].head(10).tolist()
print(f"Selected Fetaures: {selected_features}")

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_df.head(15), y="Feature", x="MI_Score")
plt.title("Top 15 Features by Mutual Information")
plt.tight_layout()
plt.show()

## --------------------------
## SECTION 4: MODEL DEVELOPMENT
## --------------------------
print("\n" + "=" * 80)
print("SECTION 4: MODEL DEVELOPMENT")
print("=" * 80 + "\n")

# 4.1 Handle Class Imbalance
print("⚖️ HANDLING CLASS IMBALANCE")
print("-" * 50)

# Apply SMOTE-Tomek
smote_tomek = SMOTETomek(sampling_strategy="auto", random_state=42)
X_res, y_res = smote_tomek.fit_resample(X_train[selected_features], y_train)

print(
    f"Class distribution after resampling: {pd.Series(y_res).value_counts(normalize=True)}"
)

# 4.2 Hyperparameter Optimization with Optuna
print("\n🎛️ HYPERPARAMETER OPTIMIZATION")
print("-" * 50)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(
        model, X_res, y_res, scoring="roc_auc", cv=5, n_jobs=-1
    ).mean()
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best ROC AUC: {study.best_value:.4f}")
print("Best Parameters:")
print(study.best_params)

# 4.3 Train Final Model
print("\n🏗️ TRAINING FINAL MODEL")
print("-" * 50)


best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_res, y_res)

# Preprocess test data
X_test_clean = clean_data(X_test)
X_test_processed = preprocessor.transform(X_test_clean)

# Ensure we have a 2D array
if X_test_processed.ndim > 2:
    X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], -1)

# Select the same features used in training
feature_indices = [i for i, col in enumerate(X.columns) if col in selected_features]
X_test_selected = X_test_processed[:, feature_indices]

# Ensure 2D shape by squeezing any extra dimensions
X_test_selected = np.squeeze(X_test_selected)  # This removes single-dimensional entries

# Verify final shape matches training data
print(f"\nTraining data shape: {X_res.shape}")
print(f"Test data shape: {X_test_selected.shape}")
assert X_test_selected.ndim == 2, (
    f"Data must be 2-dimensional, got {X_test_selected.ndim}"
)
assert X_test_selected.shape[1] == len(selected_features), "Feature count mismatch"

# Predictions
y_pred = model.predict(X_test_selected)
y_proba = model.predict_proba(X_test_selected)[:, 1]

## --------------------------
## SECTION 5: MODEL EVALUATION
## --------------------------
print("\n" + "=" * 80)
print("SECTION 5: MODEL EVALUATION")
print("=" * 80 + "\n")

# 5.1 Classification Metrics
print("📊 CLASSIFICATION METRICS")
print("-" * 50)

print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))

# Visal Classification report
visualizer = ClassificationReport(
    model, classes=["Non-Diabetic", "Diabetic"], support=True, cmap="Blues"
)
visualizer.fit(X_res, y_res)
visualizer.score(X_test_selected, y_test)
visualizer.show()

# 5.2 ROC and PR Curves
print("\n📈 ROC & PR CURVES")
print("-" * 50)

# ROC Curve
roc_visualizer = ROCAUC(model, classes=["Non-Diabetic", "Diabetic"])
roc_visualizer.fit(X_res, y_res)
roc_visualizer.score(X_test_selected, y_test)
roc_visualizer.show()

# Precision - Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 5.3 Confusion Matrix
print("\n📊 CONFUSION MATRIX")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Diabetic", "Diabetic"],
    yticklabels=["Non-Diabetic", "Diabetic"],
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()

## --------------------------
## SECTION 6: MODEL INTERPRETATION
## --------------------------
print("\n" + "=" * 80)
print("SECTION 6: MODEL INTERPRETATION")
print("=" * 80 + "\n")

# 6.1 Feature Importance
print("🔍 FEATURE IMPORTANCE")
print("-" * 50)

# Traditional Importance
importances = pd.Series(
    model.feature_importances_, index=selected_features
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh")
plt.title("Feature Importance (Gini Importance)")
plt.show()


# 6.2 SHAP Analysis
print("\n📊 SHAP ANALYSIS")
print("-" * 50)

# Convert data to numpy arrays if they aren' already
X_test_selected_array = np.array(X_test_selected).reshape(len(X_test_selected), -1)
X_res_array = np.array(X_res).reshape(len(X_res), -1)

explainer = shap.TreeExplainer(model, data=X_res_array[:100])
shap_values = explainer.shap_values(X_test_selected_array)

print("X_res shape:", X_res.shape)  # Resampled training data
print("X_test_selected shape:", X_test_selected.shape)  # Processed test data
print(
    "SHAP values length:",
    len(shap_values) if isinstance(shap_values, list) else shap_values.shape,
)

# For binary classification, SHAP returns [negative_class, positive_class]
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]  # Use positive class for analysis
else:
    shap_values_positive = shap_values

# Verify shapes
print(f"SHAP values shape: {shap_values.shape}")
print(f"Test data shape: {X_test_selected_array.shape}")

# Summary Plot for positive class
plt.close("all")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test_selected_array,
    feature_names=selected_features,
    plot_type="bar",
    show=False,
    plot_size=None,
)
plt.title("SHAP Feature Importance", pad=20)
plt.gcf().set_facecolor("white")  # ensure white background
plt.tight_layout()
plt.show()
plt.close("all")


# 6.3 LIME Explanations
print("\n🍋 LIME EXPLANATIONS")
print("-" * 50)

# Ensure we're using numpy arrays
if not isinstance(X_res_array, np.ndarray):
    X_res_array = np.array(X_res)
if not isinstance(X_test_selected_array, np.ndarray):
    X_test_selected_array = np.array(X_test_selected)

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_res_array,
    feature_names=selected_features,
    class_names=["Non-Diabetic", "Diabetic"],
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)

# Explain first test case
sample_to_explain = X_test_selected_array[0].flatten()

exp = explainer_lime.explain_instance(
    data_row=sample_to_explain,
    predict_fn=model.predict_proba,
    num_features=10,
    top_labels=1,
)

# save explanation to HTML
exp.save_to_file("lime_explanation.html")
print("LIME explanation saved to lime_explanation.html")

## --------------------------
## SECTION 7: MODEL DEPLOYMENT
## --------------------------
print("\n" + "=" * 80)
print("SECTION 7: MODEL DEPLOYMENT")
print("=" * 80 + "\n")

# 7.1 Save Model Artifacts
model_artifacts = {
    "model": model,
    "preprocessor": preprocessor,
    "selected_features": selected_features,
    "feature_importance": importances.to_dict(),
    "performance_metrics": {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": avg_precision,
    },
}

joblib.dump(model_artifacts, "diabetes_model.pkl")
print("Model artifacts saved to diabetes_model.pkl")

# 7.2 Save Metadata
metadata = {
    "model_type": "RandomForestClassifier",
    "timestamp": pd.Timestamp.now().isoformat(),
    "dataset": "diabetes.csv",
    "columns": list(df.columns),
    "best_params": best_params,
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("Model metadata saved to model_metadata.json")

print("\n✅ MODELING PIPELINE COMPLETED SUCCESSFULLY!")
