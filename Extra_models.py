# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
import joblib
from sklearn.pipeline import Pipeline
import warnings
import shap
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import lime
import lime.lime_tabular
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, mutual_info_classif
from scipy import stats
import time
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# Enhanced Data Loading and Preprocessing
def load_and_preprocess_data(filepath):
    """
    Load and preprocess diabetes dataset with advanced techniques
    """
    # Load data
    data = pd.read_csv(filepath)
    print("Original data shape:", data.shape)
    
    # Check for outliers using z-score
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data_no_outliers = data[filtered_entries]
    print(f"Removed {data.shape[0] - data_no_outliers.shape[0]} outliers using z-score")
    
    # Check for zero values that should be NaN in medical context
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
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
    # BMI categories
    data['BMI_Category'] = pd.cut(data['BMI'], 
                                 bins=[0, 18.5, 24.9, 29.9, 100], 
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Age groups
    data['Age_Group'] = pd.cut(data['Age'], 
                              bins=[0, 30, 50, 100], 
                              labels=['Young', 'Middle-aged', 'Senior'])
    
    # Feature interactions
    data['Glucose_BMI_Interaction'] = data['Glucose'] * data['BMI']
    data['Age_Insulin_Interaction'] = data['Age'] * data['Insulin']
    data['Glucose_to_Insulin_Ratio'] = data['Glucose'] / (data['Insulin'] + 1)  # Avoid division by zero
    data['HOMA-IR'] = (data['Glucose'] * data['Insulin']) / 405  # Homeostatic Model Assessment
    data['Is_Pregnant'] = (data['Pregnancies'] > 0).astype(int)
    data['Age_BP_Product'] = data['Age'] * data['BloodPressure']
    data['BP_BMI_Product'] = data['BloodPressure'] * data['BMI']
    data['DiabetesPedigreeFunction_Age'] = data['DiabetesPedigreeFunction'] * data['Age']
    
    # Polynomial features for key predictors
    data['Glucose_Squared'] = data['Glucose'] ** 2
    data['BMI_Squared'] = data['BMI'] ** 2
    data['Age_Squared'] = data['Age'] ** 2
    
    # Logarithmic transformations
    data['Log_Glucose'] = np.log1p(data['Glucose'])
    data['Log_BMI'] = np.log1p(data['BMI'])
    data['Log_Insulin'] = np.log1p(data['Insulin'] + 1)
    data['Log_SkinThickness'] = np.log1p(data['SkinThickness'] + 1)
    
    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=['BMI_Category', 'Age_Group'], drop_first=True)
    
    return data

# Perform comprehensive EDA
def perform_comprehensive_eda(data):
    """
    Perform comprehensive exploratory data analysis
    """
    print("\n--- Data Summary ---")
    print(data.describe().T)
    
    print("\n--- Missing Values ---")
    print(data.isnull().sum())
    
    print("\n--- Correlation Analysis ---")
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Outcome', data=data)
    plt.title('Distribution of Outcome')
    plt.show()
    
    # Feature distributions by outcome
    numeric_cols = numeric_data.columns[:8]  # Original features
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if col != 'Outcome':
            sns.boxplot(x='Outcome', y=col, data=data, ax=axes[i])
            axes[i].set_title(f'Distribution of {col} by Outcome')
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of key features
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if col != 'Outcome':
            sns.histplot(data=data, x=col, hue='Outcome', kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Pairplot of key features
    sns.pairplot(data[list(numeric_cols) + ['Outcome']], hue='Outcome')
    plt.show()
    
    # Interactive visualization with Plotly
    fig = px.scatter_matrix(data, 
                          dimensions=numeric_cols,
                          color='Outcome',
                          title='Scatter Matrix of Diabetes Features')
    fig.show()

# Advanced Feature Selection
def select_optimal_features(X_train, y_train, X_test):
    """
    Apply multiple feature selection techniques and compare
    """
    results = {}
    
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_features = pd.Series(mi_scores, index=X_train.columns)
    mi_selected = mi_features.nlargest(15).index.tolist()
    
    # 2. RFECV
    rfecv = RFECV(
        estimator=RandomForestClassifier(random_state=42),
        step=1,
        cv=5,
        scoring='accuracy',
        min_features_to_select=5
    )
    rfecv.fit(X_train, y_train)
    rfecv_selected = X_train.columns[rfecv.support_].tolist()
    
    # 3. SelectFromModel with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    selector = SelectFromModel(rf, threshold='median')
    selector.fit(X_train, y_train)
    rf_selected = X_train.columns[selector.get_support()].tolist()
    
    # 4. L1-based feature selection
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    lasso.fit(X_train, y_train)
    lasso_selector = SelectFromModel(lasso, threshold=0.01)
    lasso_selector.fit(X_train, y_train)
    lasso_selected = X_train.columns[lasso_selector.get_support()].tolist()
    
    # Store results
    results['mutual_info'] = {
        'features': mi_selected,
        'X_train': X_train[mi_selected],
        'X_test': X_test[mi_selected]
    }
    
    results['rfecv'] = {
        'features': rfecv_selected,
        'X_train': X_train[rfecv_selected],
        'X_test': X_test[rfecv_selected]
    }
    
    results['random_forest'] = {
        'features': rf_selected,
        'X_train': X_train[rf_selected],
        'X_test': X_test[rf_selected]
    }
    
    results['lasso'] = {
        'features': lasso_selected,
        'X_train': X_train[lasso_selected],
        'X_test': X_test[lasso_selected]
    }
    
    # Print selected features
    for method, data in results.items():
        print(f"\n{method.upper()} selected features ({len(data['features'])}): {data['features']}")
    
    return results

# Address class imbalance with multiple techniques
def handle_class_imbalance(X_train, y_train):
    """
    Apply multiple resampling techniques to handle class imbalance
    """
    resampling_results = {}
    
    # 1. SMOTE with different settings
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)
    
    # 2. ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_train, y_train)
    
    # 3. SMOTE + Tomek Links
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled_smote_tomek, y_resampled_smote_tomek = smote_tomek.fit_resample(X_train, y_train)
    
    # 4. SMOTE + ENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled_smote_enn, y_resampled_smote_enn = smote_enn.fit_resample(X_train, y_train)
    
    # Store results
    resampling_results['original'] = (X_train, y_train)
    resampling_results['smote'] = (X_resampled_smote, y_resampled_smote)
    resampling_results['adasyn'] = (X_resampled_adasyn, y_resampled_adasyn)
    resampling_results['smote_tomek'] = (X_resampled_smote_tomek, y_resampled_smote_tomek)
    resampling_results['smote_enn'] = (X_resampled_smote_enn, y_resampled_smote_enn)
    
    # Print class distributions
    for method, (X_res, y_res) in resampling_results.items():
        print(f"\n{method.upper()} class distribution:")
        print(pd.Series(y_res).value_counts())
        print(f"Shape: {X_res.shape}")
    
    return resampling_results

# Comprehensive model evaluation
def comprehensive_evaluation(model, X_test, y_test, model_name, y_prob=None):
    """
    Perform comprehensive model evaluation
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if not provided
    if y_prob is None and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif y_prob is None:
        y_prob = y_pred
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name} (AP = {avg_prec:.2f})')
    plt.show()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'avg_precision': avg_prec,
        'confusion_matrix': cm
    }

# Compare multiple models
def compare_models(models, X_train, X_test, y_train, y_test, model_names):
    """
    Train and evaluate multiple models
    """
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"\nTraining {name}...")
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'train_time': train_time,
            'inference_time': inference_time,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"Training time: {train_time:.4f}s, Inference time: {inference_time:.4f}s")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1 Score': [results[model]['f1'] for model in results],
        'AUC': [results[model]['auc'] for model in results],
        'Training Time (s)': [results[model]['train_time'] for model in results],
        'Inference Time (s)': [results[model]['inference_time'] for model in results]
    })
    
    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    # Plot comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    comparison_df_melted = pd.melt(comparison_df, 
                                  id_vars=['Model'], 
                                  value_vars=metrics,
                                  var_name='Metric', 
                                  value_name='Score')
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=comparison_df_melted, x='Model', y='Score', hue='Metric')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot ROC curves for all models
    plt.figure(figsize=(10, 8))
    for name in results:
        if hasattr(results[name]['model'], "predict_proba"):
            y_prob = results[name]['y_prob']
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = results[name]['auc']
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.show()
    
    return results, comparison_df

# Optimize logistic regression model
def optimize_logistic_regression(X_train, y_train):
    """
    Perform hyperparameter tuning for logistic regression
    """
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'penalty': ['l2', 'l1', 'elasticnet', None],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000, 2000, 3000]
    }
    
    # Create logistic regression model
    lr = LogisticRegression()
    
    # Setup grid search with cross-validation
    grid_search = GridSearchCV(
        lr, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )
    
    # Train model with grid search
    print("Performing grid search for logistic regression...")
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_lr = grid_search.best_estimator_
    
    return best_lr, grid_search.best_params_, grid_search.cv_results_

# Optimize gradient boosting model
def optimize_gradient_boosting(X_train, y_train):
    """
    Perform hyperparameter tuning for gradient boosting
    """
    # Define parameter distribution
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create gradient boosting model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Setup randomized search with cross-validation
    random_search = RandomizedSearchCV(
        gb, 
        param_distributions=param_dist, 
        n_iter=20, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, 
        random_state=42,
        verbose=1
    )
    
    # Train model with random search
    print("Performing random search for gradient boosting...")
    random_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Get best model
    best_gb = random_search.best_estimator_
    
    return best_gb, random_search.best_params_, random_search.cv_results_

# Train advanced stacking ensemble
def train_stacking_ensemble(X_train, y_train):
    """
    Train an advanced stacking ensemble model
    """
    # Base estimators
    estimators = [
        ('lr', LogisticRegression(max_iter=2000)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42))
    ]
    
    # Create stacking classifier
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        cv=5,
        stack_method='predict_proba'
    )
    
    # Train model
    print("Training stacking ensemble...")
    stack_model.fit(X_train, y_train)
    
    return stack_model

# Model interpretation with SHAP
def interpret_model_with_shap(model, X_train, X_test, feature_names):
    """
    Interpret model using SHAP values
    """
    # Create explainer
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        if hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test[:100])
    
    # If shap_values is a list (for multi-class), use the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Plot summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Plot detailed SHAP values for top features
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, plot_type='bar')
    plt.title('SHAP Feature Importance (Bar)')
    plt.tight_layout()
    plt.show()
    
    # SHAP dependence plots for top features
    top_features = np.argsort(np.abs(shap_values).mean(0))[-5:]
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_test[:100], feature_names=feature_names)
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data('/Users/omniaabouhassan/Desktop/ML project/diabetes.csv')
    
    # Apply feature engineering
    print("\nApplying feature engineering...")
    data = enhanced_feature_engineering(data)
    
    # Perform EDA
    print("\nPerforming exploratory data analysis...")
    perform_comprehensive_eda(data)
    
    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values with KNN imputation
    print("\nHandling missing values...")
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)
    
    # Feature selection
    print("\nPerforming feature selection...")
    feature_selection_results = select_optimal_features(X_train_scaled, y_train, X_test_scaled)
    
    # Use feature set from RFECV for further analysis
    X_train_selected = feature_selection_results['rfecv']['X_train']
    X_test_selected = feature_selection_results['rfecv']['X_test']
    selected_features = feature_selection_results['rfecv']['features']
    
    # Handle class imbalance
    print("\nHandling class imbalance...")
    resampling_results = handle_class_imbalance(X_train_selected, y_train)
    
    # Use SMOTE_ENN for further analysis
    X_train_resampled, y_train_resampled = resampling_results['smote_enn']
    
    # Optimize logistic regression
    print("\nOptimizing logistic regression...")
    best_lr, lr_params, lr_cv_results = optimize_logistic_regression(X_train_resampled, y_train_resampled)
    
    # Optimize gradient boosting
    print("\nOptimizing gradient boosting...")
    best_gb, gb_params, gb_cv_results = optimize_gradient_boosting(X_train_resampled, y_train_resampled)
    
    # Train other models
    print("\nTraining various models...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=42)
    
    # Train stacking ensemble
    stack_model = train_stacking_ensemble(X_train_resampled, y_train_resampled)
    
    # Compare all models
models = [best_lr, rf, svm, best_gb, stack_model]
model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting', 'Stacking Ensemble']

model_results, comparison_df = compare_models(
    models, 
    X_train_resampled,
    X_test_selected,
    y_train_resampled,
    y_test, 
    model_names
)

# Comprehensive evaluation of best model
best_model_name = comparison_df['Model'].iloc[0]
best_model = model_results[best_model_name]['model']

print(f"\nPerforming comprehensive evaluation of best model: {best_model_name}")
best_model_metrics = comprehensive_evaluation(
    best_model, 
    X_test_selected, 
    y_test, 
    best_model_name, 
    model_results[best_model_name]['y_prob']
)

# Save comparison to CSV
comparison_df.to_csv("model_comparison_results.csv", index=False)
print("Model comparison results saved to 'model_comparison_results.csv'")

# Model interpretation with SHAP
print("\nInterpreting best model with SHAP...")
interpret_model_with_shap(best_model, X_train_resampled, X_test_selected, selected_features)

# Partial dependence plots for top features
print("\nGenerating partial dependence plots...")
plt.figure(figsize=(12, 10))
display = PartialDependenceDisplay.from_estimator(
    best_model,
    X_test_selected,
    features=range(min(5, len(selected_features))),
    feature_names=selected_features,
    kind="both",
    subsample=200,
    n_jobs=-1,
    grid_resolution=50,
    random_state=42,
)
plt.suptitle(f'Partial Dependence Plots - {best_model_name}')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# LIME explanation for individual predictions
print("\nGenerating LIME explanations for individual predictions...")
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_resampled.values,
    feature_names=selected_features,
    class_names=['No Diabetes', 'Diabetes'],
    mode='classification'
)

# Explain some test instances
num_explanations = 3
for i in range(num_explanations):
    idx = np.random.randint(0, len(X_test_selected))
    exp = explainer.explain_instance(
        X_test_selected.iloc[idx].values, 
        best_model.predict_proba,
        num_features=10
    )
    
    print(f"\nLIME Explanation for Test Instance {i+1}:")
    print(f"True label: {'Diabetes' if y_test.iloc[idx] == 1 else 'No Diabetes'}")
    print(f"Predicted label: {'Diabetes' if model_results[best_model_name]['y_pred'][idx] == 1 else 'No Diabetes'}")
    print(f"Prediction probability: {model_results[best_model_name]['y_prob'][idx]:.4f}")
    
    # Plot explanation
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f"LIME Explanation for Test Instance {i+1}")
    plt.tight_layout()
    plt.show()

# Cross-validation evaluation of best model
print("\nPerforming cross-validation evaluation of best model...")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=cv, scoring='accuracy')

print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Learning curves
print("\nGenerating learning curves...")
train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X_train_selected,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

# Calculate mean and standard deviation for training scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')

# Add fills to represent one standard deviation
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

# Add labels and title
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title(f'Learning Curves - {best_model_name}')
plt.legend(loc='best')
plt.grid()
plt.show()

# Visualize feature importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    print("\nVisualing feature importance...")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance - {best_model_name}')
    plt.bar(range(len(selected_features)), importances[indices], align='center')
    plt.xticks(range(len(selected_features)), [selected_features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
elif hasattr(best_model, 'coef_'):
    print("\nVisualizing feature coefficients...")
    coeffs = best_model.coef_[0]
    indices = np.argsort(np.abs(coeffs))[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Coefficients - {best_model_name}')
    plt.bar(range(len(selected_features)), coeffs[indices], align='center')
    plt.xticks(range(len(selected_features)), [selected_features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# PCA visualization
print("\nPerforming PCA visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_selected)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='Diabetes Status')
plt.title('PCA Visualization of Test Set')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.grid()
plt.tight_layout()
plt.show()

# Create a 3D PCA visualization with Plotly
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_test_selected)

fig = px.scatter_3d(
    x=X_pca_3d[:, 0], 
    y=X_pca_3d[:, 1], 
    z=X_pca_3d[:, 2],
    color=y_test,
    opacity=0.8,
    title='3D PCA Visualization of Test Set',
    labels={
        'x': f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
        'y': f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
        'z': f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})',
        'color': 'Diabetes Status'
    }
)
fig.show()

# Save the best model, preprocessors, and feature list
print("\nSaving best model and preprocessing components...")
model_artifacts = {
    'model': best_model,
    'imputer': imputer,
    'scaler': scaler,
    'selected_features': selected_features,
    'feature_engineering_funcs': enhanced_feature_engineering,
    'model_comparison': comparison_df.to_dict()
}

joblib.dump(model_artifacts, f"diabetes_prediction_model_{best_model_name.replace(' ', '_').lower()}.pkl")
print(f"Model saved as 'diabetes_prediction_model_{best_model_name.replace(' ', '_').lower()}.pkl'")

# Create a prediction function for new data
def predict_diabetes_risk(data_dict, model_artifacts=model_artifacts):
    """
    Predict diabetes risk for a new patient
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing patient data with original features.
    model_artifacts : dict
        Dictionary containing model and preprocessors.
        
    Returns:
    --------
    prediction : int
        0 for No Diabetes, 1 for Diabetes
    probability : float
        Probability of having diabetes
    """
    # Convert input dictionary to DataFrame
    data = pd.DataFrame([data_dict])
    
    # Apply feature engineering
    data = model_artifacts['feature_engineering_funcs'](data)
    
    # Handle missing values
    data_imputed = pd.DataFrame(
        model_artifacts['imputer'].transform(data), 
        columns=data.columns
    )
    
    # Scale features
    data_scaled = pd.DataFrame(
        model_artifacts['scaler'].transform(data_imputed),
        columns=data_imputed.columns
    )
    
    # Select features
    data_selected = data_scaled[model_artifacts['selected_features']]
    
    # Make prediction
    prediction = model_artifacts['model'].predict(data_selected)[0]
    
    # Get probability if available
    if hasattr(model_artifacts['model'], 'predict_proba'):
        probability = model_artifacts['model'].predict_proba(data_selected)[0, 1]
    else:
        probability = prediction
    
    return prediction, probability

# Example of using the prediction function
example_patient = {
    'Pregnancies': 6,
    'Glucose': 140,
    'BloodPressure': 80,
    'SkinThickness': 30,
    'Insulin': 200,
    'BMI': 32.1,
    'DiabetesPedigreeFunction': 0.63,
    'Age': 50
}

prediction, probability = predict_diabetes_risk(example_patient)
print("\nExample prediction:")
print(f"Patient data: {example_patient}")
print(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
print(f"Probability of diabetes: {probability:.4f}")

# Create a simple prediction application interface function
def diabetes_prediction_interface():
    """Simple function to demonstrate the model usage with user input"""
    print("\n=== Diabetes Risk Prediction ===")
    
    # Collect user inputs
    pregnancies = int(input("Number of pregnancies: "))
    glucose = float(input("Plasma glucose concentration (mg/dL): "))
    blood_pressure = float(input("Diastolic blood pressure (mm Hg): "))
    skin_thickness = float(input("Triceps skin fold thickness (mm): "))
    insulin = float(input("2-Hour serum insulin (mu U/ml): "))
    bmi = float(input("Body mass index (weight in kg/(height in m)^2): "))
    diabetes_pedigree = float(input("Diabetes pedigree function: "))
    age = int(input("Age (years): "))
    
    # Create patient dictionary
    patient = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    # Make prediction
    prediction, probability = predict_diabetes_risk(patient)
    
    # Display results
    print("\n=== Prediction Results ===")
    print(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    print(f"Probability of diabetes: {probability:.4f}")
    
    # Risk categorization
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    print(f"Risk level: {risk_level}")
    
    # Provide some context
    if prediction == 1:
        print("\nThis model predicts that the patient may have diabetes.")
        print("Recommend follow-up clinical tests for confirmation.")
    else:
        print("\nThis model predicts that the patient likely does not have diabetes.")
        print("Recommend routine health monitoring.")
        
    print("\nNote: This is a predictive model and should not replace medical advice.")
    print("Please consult a healthcare professional for proper diagnosis.")

# Call the interface function if desired
if input("\nWould you like to try the prediction interface? (y/n): ").lower() == 'y':
    diabetes_prediction_interface()

print("\n=== Diabetes Prediction Model Analysis Complete ===")