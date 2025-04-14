import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN
import warnings
warnings.filterwarnings('ignore')

# Load and Prepare Data
def load_data(filepath):
    data = pd.read_csv(filepath)
    if data.empty:
        raise ValueError("Data not loaded correctly. Please check the file path.")
    return data

data = load_data('/Users/omniaabouhassan/Desktop/ML project/diabetes.csv')
print("Data Head:\n", data.head())

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    print(data.isnull().sum())
    sns.countplot(x='Outcome', data=data)
    plt.title('Distribution of Outcome')
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

perform_eda(data)

# Feature Engineering
def feature_engineering(data):
    data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    data['Glucose_BMI_Interaction'] = data['Glucose'] * data['BMI']
    data['Age_Insulin_Interaction'] = data['Age'] * data['Insulin']
    data['Log_Glucose'] = np.log(data['Glucose'] + 1)
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])
    return data

data = feature_engineering(data)

# Separate features and target variable
X = data.drop(['Outcome', 'BMI_Category', 'Age_Group'], axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection using RFE
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

# Cross-Validation
def cross_validate_model(model, X, y):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f'Cross-Validation Accuracy: {results.mean()}')

# Evaluate the Model
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {}'.format(accuracy))
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n{}'.format(cm))
    report = classification_report(y_test, y_pred)
    print('Classification Report:\n{}'.format(report))
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 Score: {}'.format(f1))
    print('ROC AUC: {}'.format(roc_auc))
    return cm

# Neural Networks
# MLP
mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_mlp = mlp.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
loss_mlp, accuracy_mlp = mlp.evaluate(X_test_scaled, y_test)
print('MLP Accuracy:', accuracy_mlp)

# DNN
model_dnn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_dnn = model_dnn.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
loss_dnn, accuracy_dnn = model_dnn.evaluate(X_test_scaled, y_test)
print('DNN Accuracy:', accuracy_dnn)

# CNN
model_cnn = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
history_cnn = model_cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_cnn, y_test)
print('CNN Accuracy:', accuracy_cnn)

# RNN
model_rnn = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train_rnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_rnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
history_rnn = model_rnn.fit(X_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.2)
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_rnn, y_test)
print('RNN Accuracy:', accuracy_rnn)

# Evaluate the models
def evaluate_nn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    return evaluate_model(y_test, y_pred)

# Evaluate MLP
print("Evaluating MLP Model")
evaluate_nn_model(mlp, X_test_scaled, y_test)

# Evaluate DNN
print("Evaluating DNN Model")
evaluate_nn_model(model_dnn, X_test_scaled, y_test)

# Evaluate CNN
print("Evaluating CNN Model")
evaluate_nn_model(model_cnn, X_test_cnn, y_test)

# Evaluate RNN
print("Evaluating RNN Model")
evaluate_nn_model(model_rnn, X_test_rnn, y_test)

# Plot ROC curve for neural networks
def plot_roc_curve_nn(model, X_test, y_test, model_name):
    y_prob = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curves for all neural networks
plot_roc_curve_nn(mlp, X_test_scaled, y_test, 'MLP')
plot_roc_curve_nn(model_dnn, X_test_scaled, y_test, 'DNN')
plot_roc_curve_nn(model_cnn, X_test_cnn, y_test, 'CNN')
plot_roc_curve_nn(model_rnn, X_test_rnn, y_test, 'RNN')

# Plot Precision-Recall curves for neural networks
def plot_precision_recall_curve_nn(model, X_test, y_test, model_name):
    y_prob = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'{model_name} Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

# Plot Precision-Recall curves for all neural networks
plot_precision_recall_curve_nn(mlp, X_test_scaled, y_test, 'MLP')
plot_precision_recall_curve_nn(model_dnn, X_test_scaled, y_test, 'DNN')
plot_precision_recall_curve_nn(model_cnn, X_test_cnn, y_test, 'CNN')
plot_precision_recall_curve_nn(model_rnn, X_test_rnn, y_test, 'RNN')

# Overlay ROC Curves
def plot_combined_roc_curve(models, X_tests, y_test, model_names):
    plt.figure(figsize=(10, 8))
    for model, X_test, model_name in zip(models, X_tests, model_names):
        y_prob = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Models and their corresponding test sets
models = [mlp, model_dnn, model_cnn, model_rnn]
X_tests = [X_test_scaled, X_test_scaled, X_test_cnn, X_test_rnn]
model_names = ['MLP', 'DNN', 'CNN', 'RNN']

# Plot combined ROC curve
plot_combined_roc_curve(models, X_tests, y_test, model_names)

# Overlay Precision-Recall Curves
def plot_combined_precision_recall_curve(models, X_tests, y_test, model_names):
    plt.figure(figsize=(10, 8))
    for model, X_test, model_name in zip(models, X_tests, model_names):
        y_prob = model.predict(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, lw=2, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Combined Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

# Plot combined Precision-Recall curve
plot_combined_precision_recall_curve(models, X_tests, y_test, model_names)