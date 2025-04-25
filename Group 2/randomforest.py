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
