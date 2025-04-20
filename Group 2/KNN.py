# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiabetesPredictor:
    
    
    def __init__(self, filepath=None, data=None):
        
        if filepath:
            self.data = self.load_data(filepath)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Either filepath or data must be provided")
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self, filepath):
        

        try:
            data = pd.read_csv(filepath)
            if data.empty:
                raise ValueError("Data not loaded correctly. Please check the file path.")
            logger.info(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, save_plots=True):
        
        logger.info("Performing exploratory data analysis")
        
        # Basic info and statistics
        print("Data Shape:", self.data.shape)
        print("\nData Info:")
        print(self.data.info())
        print("\nData Description:")
        print(self.data.describe().round(2))
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print("\nMissing Values:")
        print(missing_values)
        
        # Check for zeros in columns that shouldn't have zeros
        zero_counts = (self.data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum()
        print("\nZero Counts (potential missing values):")
        print(zero_counts)
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Outcome', data=self.data)
        plt.title('Distribution of Outcome (Diabetes)')
        plt.xlabel('Diabetes (1: Yes, 0: No)')
        plt.ylabel('Count')
        if save_plots:
            plt.savefig(f"{self.results_dir}/target_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.data.corr().round(2)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt='.2f')
        plt.title('Correlation Heatmap')
        if save_plots:
            plt.savefig(f"{self.results_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Distribution of features by outcome
        features_to_plot = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features_to_plot):
            plt.subplot(2, 3, i+1)
            sns.histplot(data=self.data, x=feature, hue='Outcome', kde=True, element="step")
            plt.title(f'Distribution of {feature} by Outcome')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{self.results_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Pairplot for key features
        pairplot = sns.pairplot(self.data[features_to_plot + ['Outcome']], hue='Outcome', corner=True)
        plt.suptitle('Pairplot of Key Features', y=1.02)
        if save_plots:
            pairplot.savefig(f"{self.results_dir}/pairplot.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Box plots to detect outliers
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features_to_plot):
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='Outcome', y=feature, data=self.data)
            plt.title(f'Box Plot of {feature} by Outcome')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{self.results_dir}/boxplots.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Exploratory data analysis completed")
    
    def preprocess_data(self):
        
        logger.info("Preprocessing data")
        data_copy = self.data.copy()
        
        # Replace zero values with NaN for columns that shouldn't have zeros
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            data_copy.loc[data_copy[col] == 0, col] = np.nan
        
        # Add indicators for missing values
        for col in cols_with_zeros:
            data_copy[f'{col}_missing'] = data_copy[col].isna().astype(int)
        
        # Store the preprocessed data
        self.data_preprocessed = data_copy
        logger.info("Data preprocessing completed")
        
        return data_copy
    
    def engineer_features(self, data=None):
        
        if data is None:
            if hasattr(self, 'data_preprocessed'):
                data = self.data_preprocessed.copy()
            else:
                data = self.preprocess_data()
        else:
            data = data.copy()
        
        logger.info("Engineering features")
        
        # BMI categories
        data['BMI_Category'] = pd.cut(
            data['BMI'],
            bins=[0, 18.5, 24.9, 29.9, float('inf')],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # Age categories
        data['Age_Category'] = pd.cut(
            data['Age'],
            bins=[20, 30, 40, 50, 60, float('inf')],
            labels=['20-30', '30-40', '40-50', '50-60', '60+']
        )
        
        # Glucose-to-Insulin ratio (handling div by zero)
        mask = (data['Insulin'] > 0) & (~data['Insulin'].isna())
        data['Glucose_to_Insulin'] = np.nan
        data.loc[mask, 'Glucose_to_Insulin'] = data.loc[mask, 'Glucose'] / data.loc[mask, 'Insulin']
        
        # BMI and Age interaction
        data['BMI_Age'] = data['BMI'] * data['Age'] / 100
        
        # Polynomials for important features
        for feature in ['Glucose', 'BMI']:
            data[f'{feature}_squared'] = data[feature] ** 2
        
        # Convert categorical variables to dummy variables
        if 'BMI_Category' in data.columns:
            data = pd.get_dummies(data, columns=['BMI_Category', 'Age_Category'], drop_first=True)
        
        # Store the engineered data
        self.data_engineered = data
        logger.info(f"Feature engineering completed, new shape: {data.shape}")
        
        return data
    
    def prepare_modelling_data(self, data=None, test_size=0.2):
        
        if data is None:
            if hasattr(self, 'data_engineered'):
                data = self.data_engineered.copy()
            else:
                data = self.engineer_features()
        
        logger.info("Preparing data for modelling")
        
        # Select features and target
        y = data['Outcome']
        
        # Drop non-feature columns
        X = data.drop(['Outcome'], axis=1)
        
        # Store column names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Store data
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
        return X_train, X_test, y_train, y_test
    
    def build_pipeline(self):
        
        logger.info("Building KNN pipeline")
        
        # Create pipeline with preprocessing and model
        pipeline = ImbPipeline([
            # Replace missing values using iterative imputation
            ('imputer', IterativeImputer(max_iter=10, random_state=RANDOM_STATE)),
            
            # Scale features
            ('scaler', RobustScaler()),
            
            # Transform features to be more Gaussian-like
            ('transformer', PowerTransformer(standardize=True)),
            
            # Use SMOTE to handle class imbalance
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            
            # Feature selection
            ('feature_selector', SelectKBest(f_classif, k='all')),
            
            # KNN model
            ('model', KNeighborsClassifier())
        ])
        
        return pipeline
    
    def tune_model(self, pipeline=None, cv=5):
        
        if pipeline is None:
            pipeline = self.build_pipeline()
        
        logger.info("Tuning model hyperparameters")
        
        # Define parameter grid
        param_grid = {
            'feature_selector__k': [5, 7, 9, 'all'],
            'model__n_neighbors': [3, 5, 7, 9, 11],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan', 'minkowski'],
            'model__p': [1, 2]  # p=1 is manhattan, p=2 is euclidean
        }
        
        # Create grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Store best model
        self.model = grid_search.best_estimator_
        
        # Print results
        print("Best Parameters:", grid_search.best_params_)
        print("Best CV Score:", round(grid_search.best_score_, 4))
        
        logger.info(f"Model tuning completed. Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def evaluate_model(self, model=None, save_plots=True):
        
        if model is None:
            if self.model is None:
                raise ValueError("No model available. Please train a model first.")
            model = self.model
        
        logger.info("Evaluating model on test set")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        balanced_acc = balanced_accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_prob)
        avg_precision = average_precision_score(self.y_test, y_prob)
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        report = classification_report(self.y_test, y_pred)
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if save_plots:
            plt.savefig(f"{self.results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        if save_plots:
            plt.savefig(f"{self.results_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_prob)
        plt.plot(recall_curve, precision_curve, color='green', lw=2, 
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.axhline(y=np.sum(self.y_test) / len(self.y_test), color='gray', linestyle='--',
                   label=f'Baseline (AP = {np.sum(self.y_test) / len(self.y_test):.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        if save_plots:
            plt.savefig(f"{self.results_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create evaluation metrics dictionary
        evaluation_metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info("Model evaluation completed")
        
        return evaluation_metrics
    
    def analyze_feature_importance(self, model=None, save_plots=True):
       
        if model is None:
            if self.model is None:
                raise ValueError("No model available. Please train a model first.")
            model = self.model
        
        logger.info("Analyzing feature importance")
        
        # Get feature selector from pipeline
        if hasattr(model, 'named_steps') and 'feature_selector' in model.named_steps:
            if model.named_steps['feature_selector'].k != 'all':
                # Get selected feature indices
                support = model.named_steps['feature_selector'].get_support()
                selected_features = [f for i, f in enumerate(self.feature_names) if support[i]]
                print(f"\nSelected Features ({len(selected_features)}):")
                for feature in selected_features:
                    print(f"- {feature}")
        
        # Get the final estimator from the pipeline
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            final_estimator = model.named_steps['model']
        else:
            final_estimator = model
        
        # Calculate permutation importance
        logger.info("Calculating permutation importance")
        perm_importance = permutation_importance(
            model, self.X_test, self.y_test, 
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
        )
        
        # Sort features by importance
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        
        # Print permutation importance
        print("\nPermutation Importance:")
        for i in sorted_idx:
            if i < len(self.feature_names):
                print(f"{self.feature_names[i]}: {perm_importance.importances_mean[i]:.4f} ± {perm_importance.importances_std[i]:.4f}")
        
        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        bar_data = sorted(zip(perm_importance.importances_mean, self.feature_names), reverse=True)
        importances, names = zip(*bar_data[:15])  # Show top 15 features
        plt.barh(range(len(names)), importances, align='center')
        plt.yticks(range(len(names)), names)
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance (Permutation Importance)')
        if save_plots:
            plt.savefig(f"{self.results_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Feature importance analysis completed")
    
    def plot_learning_curves(self, model=None, save_plots=True):
        
        if model is None:
            if self.model is None:
                raise ValueError("No model available. Please train a model first.")
            model = self.model
        
        logger.info("Plotting learning curves")
        
        # Generate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        # Calculate mean and standard deviation
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std,
            alpha=0.1, color='blue'
        )
        plt.fill_between(
            train_sizes, test_mean - test_std, test_mean + test_std,
            alpha=0.1, color='orange'
        )
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
        plt.xlabel('Training Examples')
        plt.ylabel('Balanced Accuracy Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        if save_plots:
            plt.savefig(f"{self.results_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Learning curves plotted")
    
    def save_model(self, filename='knn_diabetes_model.pkl'):
        
        if self.model is None:
            raise ValueError("No model available. Please train a model first.")
        
        filepath = os.path.join(self.results_dir, filename)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename='knn_diabetes_model.pkl'):
        
        filepath = os.path.join(self.results_dir, filename)
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def predict_new_data(self, new_data):
       
        if self.model is None:
            raise ValueError("No model available. Please train a model first.")
        
        # Make predictions
        try:
            y_pred = self.model.predict(new_data)
            y_prob = self.model.predict_proba(new_data)[:, 1]
            logger.info(f"Predictions made for {len(new_data)} samples")
            return y_pred, y_prob
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor('/Users/omniaabouhassan/Desktop/ML project/diabetes.csv')
    
    # Explore data
    predictor.explore_data()
    
    # Preprocess and engineer features
    data_preprocessed = predictor.preprocess_data()
    data_engineered = predictor.engineer_features(data_preprocessed)
    
    # Prepare data for modelling
    predictor.prepare_modelling_data(data_engineered)
    
    # Build pipeline and tune model
    pipeline = predictor.build_pipeline()
    grid_search = predictor.tune_model(pipeline)
    
    # Evaluate model
    evaluation_metrics = predictor.evaluate_model()
    
    # Analyze feature importance
    predictor.analyze_feature_importance()
    
    # Plot learning curves
    predictor.plot_learning_curves()
    
    # Save model
    predictor.save_model()
    
    print("\nComplete pipeline execution finished successfully!")