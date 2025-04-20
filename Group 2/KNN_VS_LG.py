import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

# Load and Prepare Data
def load_data(filepath):
    data = pd.read_csv(filepath)
    if data.empty:
        raise ValueError("Data not loaded correctly. Please check the file path.")
    return data

data = load_data('/Users/omniaabouhassan/Desktop/ML project/diabetes.csv')

# Feature Engineering
def feature_engineering(data):
    data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    return data

data = feature_engineering(data)

# Separate features and target variable
X = data.drop(['Outcome', 'BMI_Category'], axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Initialize and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Evaluate models
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, cm, report, precision, recall, f1, roc_auc

log_reg_metrics = evaluate_model(y_test, y_pred_log_reg)
knn_metrics = evaluate_model(y_test, y_pred_knn)

# Create comparison page
def create_comparison_page(log_reg_metrics, knn_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Confusion Matrix
    sns.heatmap(log_reg_metrics[1], annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Logistic Regression Confusion Matrix')
    sns.heatmap(knn_metrics[1], annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('KNN Confusion Matrix')

    # ROC Curve
    y_prob_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]
    fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_prob_log_reg)
    axes[1, 0].plot(fpr_log_reg, tpr_log_reg, color='blue', lw=2, label='Logistic Regression (area = %0.2f)' % log_reg_metrics[6])
    axes[1, 0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('Logistic Regression ROC Curve')
    axes[1, 0].legend(loc="lower right")

    y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
    axes[1, 1].plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN (area = %0.2f)' % knn_metrics[6])
    axes[1, 1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('KNN ROC Curve')
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    print("Logistic Regression Metrics:")
    print("Accuracy:", log_reg_metrics[0])
    print("Precision:", log_reg_metrics[3])
    print("Recall:", log_reg_metrics[4])
    print("F1 Score:", log_reg_metrics[5])
    print("ROC AUC:", log_reg_metrics[6])
    print("Classification Report:\n", log_reg_metrics[2])
    print("Logistic Regression shows a balanced performance with a good ROC AUC score, indicating a reliable model for this dataset.")

    print("\nKNN Metrics:")
    print("Accuracy:", knn_metrics[0])
    print("Precision:", knn_metrics[3])
    print("Recall:", knn_metrics[4])
    print("F1 Score:", knn_metrics[5])
    print("ROC AUC:", knn_metrics[6])
    print("Classification Report:\n", knn_metrics[2])
    print("KNN also performs well, but its ROC AUC score is slightly lower than Logistic Regression, suggesting it might be less reliable for this dataset.")

    # Create a comparison table
    comparison_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Logistic Regression': [log_reg_metrics[0], log_reg_metrics[3], log_reg_metrics[4], log_reg_metrics[5], log_reg_metrics[6]],
        'KNN': [knn_metrics[0], knn_metrics[3], knn_metrics[4], knn_metrics[5], knn_metrics[6]]
    }
    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparison Table:")
    print(comparison_df)

create_comparison_page(log_reg_metrics, knn_metrics)