#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Diabetes Prediction Model
A streamlined implementation with essential features and models
"""

# Import core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import warnings

# Import sklearn components
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Check for optional dependencies
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("Note: imbalanced-learn not available. Install with: pip install imbalanced-learn")


def load_data(filepath):
    """
    Load data from CSV file and perform basic preprocessing
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    # Load data
    data = pd.read_csv(filepath)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
    
    # Handle zeros in columns where zeros are not biologically plausible
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        # Count zeros in each column
        zero_count = (data[col] == 0).sum()
        if zero_count > 0:
            print(f"Replacing {zero_count} zeros in {col} with NaN")
            data.loc[data[col] == 0, col] = np.nan
    
    return data


def engineer_features(data):
    """
    Create new features to improve model performance
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # BMI categories
    df['BMI_Category'] = pd.cut(df['BMI'], 
                               bins=[0, 18.5, 24.9, 29.9, 100], 
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 30, 45, 60, 100], 
                            labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # Feature interactions (most important ones)
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Insulin'] = df['Age'] * df['Insulin']
    df['Glucose_to_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)  # Add 1 to avoid division by zero
    
    # Log transformations
    df['Log_Glucose'] = np.log1p(df['Glucose'])
    df['Log_BMI'] = np.log1p(df['BMI'])
    df['Log_Insulin'] = np.log1p(df['Insulin'] + 1)
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['BMI_Category', 'Age_Group'], drop_first=True)
    
    print(f"Features added: {df.shape[1] - data.shape[1]} new features created")
    return df


def visualize_data(data):
    """
    Create basic visualizations of the data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    """
    # Outcome distribution
    plt.figure(figsize=(10, 6))
    outcome_counts = data['Outcome'].value_counts()
    sns.countplot(x='Outcome', data=data)
    plt.title('Distribution of Diabetes Outcomes')
    
    # Add percentages
    total = len(data)
    for i, count in enumerate(outcome_counts):
        plt.text(i, count + 5, f"{count} ({count/total*100:.1f}%)", ha='center')
    
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = data.select_dtypes(include=['number']).corr()
    mask = np.triu(corr)
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Feature distributions by outcome
    features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'DiabetesPedigreeFunction']
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature in data.columns:
            sns.boxplot(x='Outcome', y=feature, data=data, ax=axes[i])
            axes[i].set_title(f'{feature} by Diabetes Outcome')
    
    plt.tight_layout()
    plt.show()


def prepare_data(data):
    """
    Split data and prepare for modeling
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with engineered features
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, imputer, scaler
    """
    # Split features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split into train and test sets (stratify to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Handle missing values with KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)
    
    print(f"Data prepared: {X_train_scaled.shape[0]} training samples, {X_test_scaled.shape[0]} test samples")
    return X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler


def balance_classes(X_train, y_train):
    """
    Handle class imbalance using SMOTE or SMOTEENN
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
        
    Returns:
    --------
    tuple
        X_train_resampled, y_train_resampled
    """
    if not IMBALANCED_AVAILABLE:
        print("Skipping class balancing - imbalanced-learn not available")
        return X_train, y_train
    
    # Print original class distribution
    print("Original class distribution:")
    print(y_train.value_counts())
    
    try:
        # Apply SMOTEENN (combination of oversampling and cleaning)
        sampler = SMOTEENN(random_state=RANDOM_STATE)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # Print new class distribution
        print("Balanced class distribution:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"Error in class balancing: {e}")
        # Fallback to SMOTE if SMOTEENN fails
        try:
            sampler = SMOTE(random_state=RANDOM_STATE)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        except:
            # Return original if all balancing methods fail
            return X_train, y_train


def train_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    tuple
        best_model, results_dict, comparison_df
    """
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'train_time': train_time,
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        results[name] = metrics
        
        # Print results
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Train time: {metrics['train_time']:.2f} seconds")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1 Score': [results[m]['f1'] for m in results],
        'AUC': [results[m]['auc'] for m in results],
        'Training Time (s)': [results[m]['train_time'] for m in results]
    }).sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    # Determine best model (by F1 score)
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name} (F1 Score: {results[best_model_name]['f1']:.4f})")
    
    # Visualize model performance
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    comparison_melted = pd.melt(comparison_df, 
                              id_vars=['Model'], 
                              value_vars=metrics_to_plot,
                              var_name='Metric', 
                              value_name='Score')
    
    sns.barplot(data=comparison_melted, x='Model', y='Score', hue='Metric')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    
    return best_model, results, comparison_df


def optimize_best_model(model, X_train, y_train):
    """
    Optimize hyperparameters for the best model
    
    Parameters:
    -----------
    model : object
        Best model to optimize
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
        
    Returns:
    --------
    object
        Optimized model
    """
    print(f"\nOptimizing hyperparameters for {type(model).__name__}...")
    
    # Define parameter grid based on model type
    if isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif isinstance(model, GradientBoostingClassifier):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10]
        }
    elif isinstance(model, LogisticRegression):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs', 'newton-cg'],
            'class_weight': [None, 'balanced']
        }
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    else:
        print(f"Unsupported model type: {type(model).__name__}")
        return model
    
    # Create randomized search
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings to try
        cv=5,  # Cross-validation folds
        scoring='f1',  # Optimize for F1 score
        n_jobs=-1,  # Use all available cores
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # Fit randomized search
    random_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def evaluate_final_model(model, X_test, y_test):
    """
    Perform detailed evaluation of the final model
    
    Parameters:
    -----------
    model : object
        Model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probability predictions if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print metrics
    print("\nFinal Model Evaluation:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plot feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_names = X_test.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
        # Print top 10 features
        print("\nTop 10 Features:")
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {metrics['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return metrics


def save_model(model, imputer, scaler, filepath='diabetes_model.pkl'):
    """
    Save model and preprocessing components
    
    Parameters:
    -----------
    model : object
        Trained model
    imputer : object
        Fitted imputer
    scaler : object
        Fitted scaler
    filepath : str
        File path to save model
        
    Returns:
    --------
    str
        File path where model was saved
    """
    # Create a dict with all components
    model_artifacts = {
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'model_type': type(model).__name__,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to disk
    joblib.dump(model_artifacts, filepath)
    print(f"Model saved to {filepath}")
    
    return filepath


def predict_diabetes(patient_data, model_artifacts):
    """
    Make diabetes prediction for new patient data
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary with patient data
    model_artifacts : dict
        Dictionary with model and preprocessing components
        
    Returns:
    --------
    dict
        Prediction results
    """
    # Extract components
    model = model_artifacts['model']
    imputer = model_artifacts['imputer']
    scaler = model_artifacts['scaler']
    
    # Convert input to DataFrame
    df = pd.DataFrame([patient_data])
    
    # Apply feature engineering
    df_engineered = engineer_features(df)
    
    # Handle missing values
    df_imputed = pd.DataFrame(
        imputer.transform(df_engineered),
        columns=df_engineered.columns
    )
    
    # Scale features
    df_scaled = pd.DataFrame(
        scaler.transform(df_imputed),
        columns=df_imputed.columns
    )
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(df_scaled)[0, 1]
    else:
        probability = float(prediction)
    
    # Determine risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    # Return results
    result = {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': risk_level,
        'diagnosis': 'Diabetes' if prediction == 1 else 'No Diabetes'
    }
    
    return result


def prediction_interface():
    """
    Simple interface for diabetes prediction
    """
    print("\n=== Diabetes Risk Prediction ===")
    
    try:
        # Load the model
        model_path = input("Enter path to saved model (or press Enter for default 'diabetes_model.pkl'): ")
        if not model_path:
            model_path = "diabetes_model.pkl"
        
        model_artifacts = joblib.load(model_path)
        print(f"Model loaded: {model_artifacts['model_type']}")
        
        # Collect user inputs
        pregnancies = int(input("Number of pregnancies: "))
        glucose = float(input("Plasma glucose concentration (mg/dL): "))
        blood_pressure = float(input("Diastolic blood pressure (mm Hg): "))
        skin_thickness = float(input("Triceps skin fold thickness (mm): "))
        insulin = float(input("2-Hour serum insulin (mu U/ml): "))
        bmi = float(input("Body mass index: "))
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
        result = predict_diabetes(patient, model_artifacts)
        
        # Display results
        print("\n=== Prediction Results ===")
        print(f"Diagnosis: {result['diagnosis']}")
        print(f"Probability of diabetes: {result['probability']:.4f}")
        print(f"Risk level: {result['risk_level']}")
        
        # Provide context
        if result['prediction'] == 1:
            print("\nThis model predicts that the patient may have diabetes.")
            print("Recommend follow-up clinical tests for confirmation.")
        else:
            print("\nThis model predicts that the patient likely does not have diabetes.")
            print("Recommend routine health monitoring.")
        
        print("\nNote: This is a predictive model and should not replace medical advice.")
        print("Please consult a healthcare professional for proper diagnosis.")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """
    Main function to run the diabetes prediction pipeline
    """
    print("=== Diabetes Prediction Pipeline ===")
    
    # Get data path
    data_path = input("Enter the path to the diabetes dataset CSV file (or press Enter for default path): ")
    if not data_path:
        data_path = '/Users/omniaabouhassan/Desktop/ML project/diabetes.csv'
    
    # Load and preprocess data
    data = load_data(data_path)
    
    # Visualize data
    visualize_data(data)
    
    # Engineer features
    data_engineered = engineer_features(data)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, imputer, scaler = prepare_data(data_engineered)
    
    # Balance classes
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    
    # Train models
    best_model, results, comparison_df = train_models(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Optimize best model
    optimize = input("\nWould you like to optimize the best model? (y/n): ").lower() == 'y'
    if optimize:
        optimized_model = optimize_best_model(best_model, X_train_balanced, y_train_balanced)
        best_model = optimized_model
    
    # Evaluate final model
    metrics = evaluate_final_model(best_model, X_test, y_test)
    
    # Save model
    save = input("\nWould you like to save the model? (y/n): ").lower() == 'y'
    if save:
        model_path = save_model(best_model, imputer, scaler)
        
        # Ask if user wants to test the model
        test_model = input("\nWould you like to test the model with new data? (y/n): ").lower() == 'y'
        if test_model:
            prediction_interface()
    
    print("\n=== Diabetes Prediction Pipeline Complete ===")


if __name__ == "__main__":
    main()