# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.pipeline import Pipeline
import warnings
import shap
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE, ADASYN
import lime
import lime.lime_tabular
import optuna
import plotly.express as px
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

# Handling Missing Values
data.fillna(data.mean(), inplace=True)

# Feature Engineering
def feature_engineering(data):
    data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    poly = PolynomialFeatures(degree=2)
    data_poly = poly.fit_transform(data.drop(['Outcome', 'BMI_Category'], axis=1))
    return data, data_poly

data, data_poly = feature_engineering(data)

# Separate features and target variable
X = data.drop(['Outcome', 'BMI_Category'], axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Automated Hyperparameter Tuning with Optuna
def objective(trial):
    C = trial.suggest_loguniform('C', 0.1, 100)
    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    model = LogisticRegression(C=C, solver=solver, max_iter=1000)
    score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)

# Initialize Logistic Regression with best parameters
best_model = LogisticRegression(**study.best_params, max_iter=1000)
best_model.fit(X_train_scaled, y_train)

# Use the best model to make predictions
y_pred = best_model.predict(X_test_scaled)

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
    mcc = matthews_corrcoef(y_test, y_pred)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 Score: {}'.format(f1))
    print('ROC AUC: {}'.format(roc_auc))
    print('Matthews Correlation Coefficient: {}'.format(mcc))
    return cm

cm = evaluate_model(y_test, y_pred)

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print('Cross-Validation Scores:', cv_scores)
print('Mean Cross-Validation Score:', np.mean(cv_scores))

# Get feature importance
importance = best_model.coef_[0]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Compare with other models
models = [RandomForestClassifier(), SVC(probability=True), GradientBoostingClassifier()]
model_names = ['Random Forest', 'SVM', 'Gradient Boosting']

# Train and evaluate other models
for model, name in zip(models, model_names):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')

# Hyperparameter tuning for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}
grid_search_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train_scaled, y_train)
print("Best Parameters for SVM:", grid_search_svm.best_params_)
print("Best Score for SVM:", grid_search_svm.best_score_)

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train_scaled, y_train)
print("Best Parameters for Gradient Boosting:", grid_search_gb.best_params_)
print("Best Score for Gradient Boosting:", grid_search_gb.best_score_)

# Save the model
joblib.dump(best_model, 'best_model.pkl')

# Load the model
loaded_model = joblib.load('best_model.pkl')

# Make predictions with the loaded model
y_pred_loaded = loaded_model.predict(X_test_scaled)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Compute ROC curve and ROC area
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot distribution of predictions and actual outcomes
plt.figure(figsize=(10, 6))
sns.histplot(y_test, kde=True, color='blue', label='Actual')
sns.histplot(y_pred, kde=True, color='red', label='Predicted')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions and Actual Outcomes')
plt.legend()
plt.show()

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred_pipeline = pipeline.predict(X_test)

# Evaluate the pipeline
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
print('Pipeline Accuracy: {}'.format(accuracy_pipeline))

# Model Interpretation using SHAP
explainer = shap.Explainer(best_model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
# Feature Selection using RFE
from sklearn.feature_selection import RFE
selector = RFE(best_model, n_features_to_select=5, step=1)
selector = selector.fit(X_train_scaled, y_train)
selected_features = X.columns[selector.support_]
print("Selected Features:", selected_features)

# Train models with selected features
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Ensemble Methods
ensemble_model = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True)),
    ('gb', GradientBoostingClassifier())
], voting='soft')
ensemble_model.fit(X_train_selected, y_train)
y_pred_ensemble = ensemble_model.predict(X_test_selected)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print('Ensemble Model Accuracy:', accuracy_ensemble)

# Handling Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train the model with resampled data
best_model.fit(X_resampled, y_resampled)
y_pred_smote = best_model.predict(X_test_scaled)
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print('SMOTE Model Accuracy:', accuracy_smote)

# Handling Imbalanced Data using ADASYN
adasyn = ADASYN(random_state=42)
X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_train_scaled, y_train)

# Train the model with ADASYN resampled data
best_model.fit(X_resampled_adasyn, y_resampled_adasyn)
y_pred_adasyn = best_model.predict(X_test_scaled)
accuracy_adasyn = accuracy_score(y_test, y_pred_adasyn)
print('ADASYN Model Accuracy:', accuracy_adasyn)

# Advanced Visualization using Plotly
fig = px.histogram(data, x='BMI', color='Outcome', title='Distribution of BMI by Outcome')
fig.show()

# Model Interpretability using LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], verbose=True, mode='classification')
i = np.random.randint(0, X_test_scaled.shape[0])
exp = explainer.explain_instance(X_test_scaled[i], best_model.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True, show_all=False)

# Additional Visualizations
# Distribution plots for each feature by outcome class
for column in X.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data, x=column, hue='Outcome', kde=True)
    plt.title(f'Distribution of {column} by Outcome')
    plt.show()

# Partial dependence plots
PartialDependenceDisplay.from_estimator(best_model, X_train_scaled, features=[0, 1, 2, 3, 4, 5, 6, 7], feature_names=X.columns)
plt.show()

# Learning curves
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

# Calculate mean and standard deviation for training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc="best")
plt.show()