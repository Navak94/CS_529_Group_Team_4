"""
!!! Using iris_2 as the virtual env and also the python interpreter !!!
Diabetes Prediction using Random Forest with Advanced Feature Engineering

Objective:
Develop a robust diabetes prediction model using Random Forest Classifier with comprehensive
feature engineering and optimization techniques to maximize predictive performance.

Key Steps:
1. Perform advanced data cleaning including outlier treatment using Winsorization
2. Create new feartures through interactions, ratios and binned categories
3. Conduct thorough feature analysis using multiple methods(Phi-K, SHAP, Permutation Importance)
4. Implement hybrid resampling (SMOTE + Tomek Links) to handle class imbalance
5. Optimize model through extensive hyperparameter tuning with GridSearchCV
6. Evaluate using multiple metrics (ROC AUC, precision-recall, etc.)
7. Compare performance with other tree-based models(XGBoost, LightGBM)
8. Provide model interpretability using SHAP values and partial dependence plots

Advanced Techniques:
- Multiple feeatur scaling approaches (RobustScaler, PowerTransformer)
- Reecursive Featture Elimination with Cross-Validation
- Dimensionality reduction visualization with PCA
- Comprehensive model interpretation tools
- Production-ready model serialization
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import (
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    KBinsDiscretizer,
)
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    mutual_info_classif,
    f_classif,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss
from imblearn.combine import SMOTETomek
import shap
from lime.lime_tabular import LimeTabularExplainer
import yellowbrick
from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix
import optuna
import joblib
import json
import phik
from phik import report
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
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
discrete_features = ["Age_Group", "BMI_Category"]

# Create Transformers
numeric_transformer = Pipeline(
    steps=[
        ("scaler", RobustScaler()),
        ("power", PowerTransformer(method="yeo-johnson")),
    ]
)

discrete_transformer = Pipeline(steps=[("scaler", RobustScaler())])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("disc", discrete_transformer, discrete_features),
    ]
)

# 3.4 Feature Selection
print("\n🔍 PERFORMING FEATURE SELECTION")
print("-" * 50)

# Fit Preprocessor
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Mutual Information
mi_scores = mutual_info_classif(X_train_preprocessed, y_train)
mi_features = pd.Series(mi_scores, index=X.columns, name="MI Scores")
