import pandas as pd
import numpy as np
import os
import ssl
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

# Fix SSL certificate issues for fetch_openml
ssl._create_default_https_context = ssl._create_unverified_context

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and Prepare Data
def load_data(filepath):
    """
    Load data from CSV file and perform basic validation
    """
    try:
        # Use absolute path from user
        data = pd.read_csv(filepath)
        if data.empty:
            raise ValueError("Data not loaded correctly. Please check the file path.")
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise  # Re-raise to handle in main code

# Use the correct path
data_path = '/Users/omniaabouhassan/Desktop/ML project/diabetes.csv'
print(f"Attempting to load data from: {data_path}")
print(f"File exists: {os.path.exists(data_path)}")

# Load the dataset
data = load_data(data_path)
print("Data Head:\n", data.head())

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    """
    Perform exploratory data analysis with enhanced visualizations
    """
    # Check for missing values
    missing = data.isnull().sum()
    print("Missing values in each column:\n", missing)
    
    # Check for zeros in columns where zero is not plausible
    zero_counts = {}
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if col in data.columns:
            zero_counts[col] = (data[col] == 0).sum()
    print("\nZero values in clinical measurements (potentially missing data):")
    for col, count in zero_counts.items():
        print(f"{col}: {count} zeros ({count/len(data)*100:.1f}%)")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(data.describe())
    
    # Outcome distribution
    plt.figure(figsize=(8, 6))
    outcome_counts = data['Outcome'].value_counts()
    ax = sns.countplot(x='Outcome', data=data, palette='viridis')
    plt.title('Distribution of Outcome (Diabetes)', fontsize=15)
    
    # Add count labels on bars
    for i, count in enumerate(outcome_counts):
        ax.text(i, count + 5, f"{count} ({count/len(data)*100:.1f}%)", 
                ha='center', fontsize=12)
    
    plt.xlabel('Diabetes Diagnosis (0=No, 1=Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.show()
    
    # Feature distributions by outcome
    plt.figure(figsize=(16, 12))
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for i, col in enumerate(numerical_cols):
        if col != 'Outcome':
            plt.subplot(3, 3, i+1)
            sns.histplot(data=data, x=col, hue='Outcome', kde=True, element='step',
                         palette=['green', 'red'], bins=20, alpha=0.6)
            plt.title(f'{col} by Outcome')
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(data.corr())
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', mask=mask, vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Heatmap', fontsize=15)
    plt.show()
    
    # Pairplot for key variables
    key_vars = ['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']
    sns.pairplot(data[key_vars], hue='Outcome', palette=['green', 'red'], diag_kind='kde')
    plt.suptitle('Pairplot of Key Variables', y=1.02, fontsize=16)
    plt.show()
    
    return data

# Improved Feature Engineering
def feature_engineering(data):
    """
    Enhanced feature engineering with additional features and proper handling of zeros
    """
    # Create a copy to avoid warnings
    df = data.copy()
    
    # Handle zeros in columns where zero is not a valid value
    # Replace with NaN first, then with median of non-zero values
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if col in df.columns:
            # Replace zeros with median of non-zero values
            median_value = df[df[col] != 0][col].median()
            df[col] = df[col].replace(0, median_value)
    
    # BMI Categories
    df['BMI_Category'] = pd.cut(df['BMI'], 
                                bins=[0, 18.5, 24.9, 29.9, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Age Categories
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 30, 45, 60, 100], 
                            labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # New ratio features
    df['Glucose_to_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)  # Add 1 to avoid division by zero
    df['BMI_to_Age_Ratio'] = df['BMI'] / df['Age']
    
    # Interaction terms
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Insulin'] = df['Age'] * df['Insulin']
    
    # Polynomial features for key predictors
    df['Glucose_Squared'] = df['Glucose'] ** 2
    df['BMI_Squared'] = df['BMI'] ** 2
    
    # Log transformations
    df['Log_Glucose'] = np.log1p(df['Glucose'])
    df['Log_Insulin'] = np.log1p(df['Insulin'])
    
    # Diabetes risk score (simplified)
    df['Diabetes_Risk_Score'] = (0.3 * df['Glucose'] + 
                                0.2 * df['BMI'] + 
                                0.1 * df['Age'] + 
                                0.15 * df['Insulin'] + 
                                0.15 * df['BloodPressure'])
    
    # Convert categorical variables to one-hot encoding
    df = pd.get_dummies(df, columns=['BMI_Category', 'Age_Group'], drop_first=True)
    
    return df

# Apply feature engineering
print("\nApplying feature engineering...")
data_engineered = feature_engineering(data)
print(f"Data shape after feature engineering: {data_engineered.shape}")

# Feature Selection
def select_features(X_train, y_train, X_test, method='rfe', n_features=15):
    """
    Advanced feature selection with multiple methods
    """
    if method == 'rfe':
        # Recursive Feature Elimination
        estimator = RandomForestClassifier(random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = X_train.columns[selector.support_]
        print("Selected features (RFE):", selected_features.tolist())
        
    elif method == 'tree_based':
        # Tree-based feature selection
        # First fit the estimator to get feature importances
        estimator = RandomForestClassifier(random_state=42)
        estimator.fit(X_train, y_train)
        
        # Now get feature importances
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(X_train.shape[1]), importances[indices])
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        plt.tight_layout()
        plt.show()
        
        # Now use SelectFromModel with the fitted estimator
        selector = SelectFromModel(estimator, max_features=n_features, prefit=True)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X_train.columns[selected_mask]
        
        print("Selected features (Tree-based):", selected_features.tolist())
        
    return X_train_selected, X_test_selected, selected_features

# Prepare data for modeling
def prepare_data(data):
    """
    Prepare data for modeling - separate features and target
    """
    # Drop non-numeric columns and outcome
    categorical_cols = ['BMI_Category', 'Age_Group']
    drop_cols = [col for col in categorical_cols if col in data.columns]
    drop_cols += ['Outcome']
    
    X = data.drop(drop_cols, axis=1)
    y = data['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to dataframes with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(data_engineered)

# Feature selection
X_train_selected, X_test_selected, selected_features = select_features(
    X_train, y_train, X_test, method='tree_based', n_features=15
)

# Create neural network models with advanced architectures and techniques
def create_mlp_model(input_shape):
    """
    Create improved MLP model with regularization and batch normalization
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_dnn_model(input_shape):
    """
    Create improved deep neural network with skip connections
    """
    inputs = Input(shape=(input_shape,))
    
    # First layer
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second layer with skip connection
    layer2 = Dense(64, activation='relu')(x)
    layer2 = BatchNormalization()(layer2)
    layer2 = Dropout(0.2)(layer2)
    
    # Skip connection
    skip1 = Concatenate()([x, layer2])
    
    # Third layer
    layer3 = Dense(32, activation='relu')(skip1)
    layer3 = BatchNormalization()(layer3)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(layer3)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_cnn_model(input_shape):
    """
    Create 1D CNN model for tabular data
    """
    model = Sequential([
        # Reshape input to have a single channel
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # 1D Convolutional layers
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(32, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),
        
        # Flatten and feed to dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_rnn_model(input_shape):
    """
    Create RNN model for tabular data
    """
    model = Sequential([
        # Reshape input to have a single feature
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # RNN layer
        SimpleRNN(64, activation='tanh', return_sequences=True),
        SimpleRNN(32, activation='tanh'),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Training settings
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Train models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, reshape_for_cnn_rnn=False):
    """
    Train and evaluate a model with proper validation and visualization
    """
    print(f"\n----- Training {model_name} Model -----")
    
    # Prepare data for CNN/RNN if needed
    if reshape_for_cnn_rnn:
        # Check if X_train is a DataFrame or ndarray and handle accordingly
        if hasattr(X_train, 'values'):
            X_train_model = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_model = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        else:
            # Already a numpy array
            X_train_model = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_model = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    # Train the model
    history = model.fit(
        X_train_model, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )
    
    # Evaluate the model
    print(f"\n{model_name} Evaluation:")
    results = model.evaluate(X_test_model, y_test, verbose=0)
    metrics = model.metrics_names
    
    for metric, value in zip(metrics, results):
        print(f"{metric}: {value:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(X_test_model)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # ROC curve
    y_prob = model.predict(X_test_model).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    average_precision = average_precision_score(y_test, y_prob)
    
    return {
        'model': model,
        'history': history,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'average_precision': average_precision
    }

# Create and train models
input_shape = X_train_selected.shape[1]

# Initialize models
mlp_model = create_mlp_model(input_shape)
dnn_model = create_dnn_model(input_shape)
cnn_model = create_cnn_model(input_shape)
rnn_model = create_rnn_model(input_shape)

# Train and evaluate models
mlp_results = train_and_evaluate_model(
    mlp_model, X_train_selected, y_train, X_test_selected, y_test, 'MLP'
)

dnn_results = train_and_evaluate_model(
    dnn_model, X_train_selected, y_train, X_test_selected, y_test, 'DNN'
)

cnn_results = train_and_evaluate_model(
    cnn_model, X_train_selected, y_train, X_test_selected, y_test, 'CNN', reshape_for_cnn_rnn=True
)

rnn_results = train_and_evaluate_model(
    rnn_model, X_train_selected, y_train, X_test_selected, y_test, 'RNN', reshape_for_cnn_rnn=True
)

# Compare all models
def compare_models(results_list, model_names):
    """
    Compare all models using ROC and PR curves
    """
    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    for result, name in zip(results_list, model_names):
        plt.plot(result['fpr'], result['tpr'], lw=2, 
                 label=f'{name} (AUC = {result["roc_auc"]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    
    # Plot Precision-Recall curves
    plt.subplot(2, 1, 2)
    
    for result, name in zip(results_list, model_names):
        plt.plot(result['recall'], result['precision'], lw=2,
                 label=f'{name} (AP = {result["average_precision"]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Models')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()
    
    # Summarize model performance
    print("\n--- Model Performance Summary ---")
    
    metrics_table = {
        'Model': [],
        'Accuracy': [],
        'AUC': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }
    
    for result, name in zip(results_list, model_names):
        y_pred = result['y_pred']
        y_test_data = y_test  # Assuming y_test is in the global scope
        
        metrics_table['Model'].append(name)
        metrics_table['Accuracy'].append(f"{accuracy_score(y_test_data, y_pred):.4f}")
        metrics_table['AUC'].append(f"{result['roc_auc']:.4f}")
        metrics_table['Precision'].append(f"{precision_score(y_test_data, y_pred):.4f}")
        metrics_table['Recall'].append(f"{recall_score(y_test_data, y_pred):.4f}")
        metrics_table['F1 Score'].append(f"{f1_score(y_test_data, y_pred):.4f}")
    
    # Display results as a table
    metrics_df = pd.DataFrame(metrics_table)
    print(metrics_df.to_string(index=False))

# Compare all models
all_results = [mlp_results, dnn_results, cnn_results, rnn_results]
model_names = ['MLP', 'DNN', 'CNN', 'RNN']

compare_models(all_results, model_names)

# Feature importance analysis for the best model (using a wrapper)
def analyze_feature_importance(model, X_train, y_train, feature_names):
    """
    Analyze feature importance using a custom approach for neural networks
    """
    print("\nAnalyzing feature importance for the best model...")
    
    # Since permutation_importance doesn't work well with our Keras wrapper,
    # let's use a simpler approach to estimate feature importance
    
    # Convert to numpy if needed
    if hasattr(X_train, 'values'):
        X_train_data = X_train.values
    else:
        X_train_data = X_train
        
    # Get baseline prediction and accuracy
    baseline_pred = model.predict(X_train_data)
    baseline_pred_class = (baseline_pred > 0.5).astype(int).ravel()
    baseline_accuracy = accuracy_score(y_train, baseline_pred_class)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Calculate importance for each feature
    importances = []
    feature_indices = range(X_train_data.shape[1])
    
    for i in feature_indices:
        # Create a copy of the data
        X_permuted = X_train_data.copy()
        
        # Permute the feature
        np.random.shuffle(X_permuted[:, i])
        
        # Predict with permuted feature
        permuted_pred = model.predict(X_permuted)
        permuted_pred_class = (permuted_pred > 0.5).astype(int).ravel()
        permuted_accuracy = accuracy_score(y_train, permuted_pred_class)
        
        # Calculate importance as drop in accuracy
        importance = baseline_accuracy - permuted_accuracy
        importances.append(importance)
        print(f"Feature {i} ({feature_names[i]}) importance: {importance:.4f}")
    
    # Convert to numpy array
    importances = np.array(importances)
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance (Drop in Accuracy When Permuted)')
    plt.title('Neural Network Feature Importance Analysis')
    plt.tight_layout()
    plt.show()
    
    # Print top important features
    print("\nTop 5 most important features:")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return importances, sorted_idx

# Determine the best model based on AUC
best_model_idx = np.argmax([result['roc_auc'] for result in all_results])
best_model_name = model_names[best_model_idx]
best_model = all_results[best_model_idx]['model']

print(f"\nThe best performing model is: {best_model_name}")

# Analyze feature importance for the best model if it's MLP or DNN
if best_model_name in ['MLP', 'DNN']:
    analyze_feature_importance(best_model, X_train_selected, y_train, selected_features)

# Save the best model
best_model.save(f'best_diabetes_model_{best_model_name}.h5')
print(f"\nBest model saved as 'best_diabetes_model_{best_model_name}.h5'")

# Sample prediction function
def predict_diabetes_risk(model, new_data, scaler, feature_names):
    """
    Make prediction on new patient data
    """
    # Ensure data has the right features
    new_data_df = pd.DataFrame([new_data], columns=feature_names)
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data_df)
    
    # Make prediction
    prediction_prob = model.predict(new_data_scaled)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0
    
    return {
        'probability': float(prediction_prob),
        'prediction': int(prediction),
        'risk_level': 'High' if prediction_prob > 0.7 else 
                      'Moderate' if prediction_prob > 0.3 else 'Low'
    }

print("\nModel improvement complete. The enhanced neural networks code now includes:")
print("1. Better data preprocessing with proper handling of zeros/missing values")
print("2. Enhanced feature engineering with more meaningful features")
print("3. Advanced model architectures with regularization, batch normalization, and dropout")
print("4. Early stopping and learning rate scheduling for better convergence")
print("5. Comprehensive model evaluation and comparison")
print("6. Feature importance analysis")
print("7. Better code organization with docstrings and comments")