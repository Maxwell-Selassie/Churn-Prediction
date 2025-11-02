"""
Production-Grade Model Training Pipeline with MLflow Tracking
Implements automated feature selection and multi-phase training

Author: Maxwell Selassie Hiamatsu
Date: October 24, 2025
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.inspection import permutation_importance

# External libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available")

# Import utilities
from utils import (
    setup_logger, load_csv_file, read_yaml_file,
    project_metadata, get_timestamp, ensure_directories
)

# Setup logging
logger = setup_logger('ModelTraining', 'logs/model_training.log')


class ChurnModelTrainer:
    """
    Production-grade model training pipeline with MLflow tracking
    
    Features:
    - Multi-phase training (baseline → feature selection → retraining → tuning)
    - Automated feature importance analysis
    - MLflow experiment tracking
    - Model registry integration
    - SHAP explainability
    - Cross-validation
    - Model comparison dashboard
    """
    
    def __init__(self, config_path: str = 'config/model_training_config.yaml'):
        """Initialize trainer with configuration"""
        logger.info('='*70)
        logger.info('INITIALIZING CHURN MODEL TRAINING PIPELINE')
        logger.info('='*70)
        
        self.config = read_yaml_file(config_path)
        self.config_path = config_path
        
        # Setup MLflow
        if self.config['mlflow']['enabled']:
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])
            if self.config['mlflow']['autolog']['enabled']:
                mlflow.sklearn.autolog(
                    log_input_examples=self.config['mlflow']['autolog']['log_input_examples'],
                    log_model_signatures=self.config['mlflow']['autolog']['log_model_signatures'],
                    log_models=self.config['mlflow']['autolog']['log_models']
                )
        
        # Initialize storage
        self.X_train = self.config['paths']['x_train']
        self.X_test = self.config['paths']['x_test']
        self.y_train = self.config['paths']['x_test']
        self.y_test = self.config['paths']['y_test']

        
        self.selected_features = None
        self.feature_importance_df = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        
        self.model_results = {}
        
        # Create directories
        for dir_path in ['models', 'results', 'logs']:
            ensure_directories(dir_path)

        project_name = self.config['project']['name']
        version_name = self.config['project']['version']
        mlflow_tracking = self.config['mlflow']['enabled']
        
        logger.info(f"Project: {project_name}")
        logger.info(f"Version: {version_name}")
        logger.info(f"MLflow tracking: {mlflow_tracking}")
    
    # ========================================================================
    # DATA LOADING AND SPLITTING
    # ========================================================================
    
    def load_and_split_data(self):
        """Load engineered data and split into train/val/test"""
        logger.info('-'*70)
        logger.info('STEP 1: LOADING AND SPLITTING DATA')
        logger.info('-'*70)
        
        # Load data
        df = load_csv_file(self.config['paths']['engineered_data'])
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Separate features and target
        target_col = self.config['data_split']['target_column']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        logger.info(f"Churn rate: {y.mean():.2%}")
        
        # Train-test split
        test_size = self.config['data_split']['test_size']
        stratify = y if self.config['data_split']['stratify'] else None
        random_state = self.config['data_split']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        # Train-validation split
        val_size = self.config['data_split']['validation_size']
        stratify_train = y_train if self.config['data_split']['stratify'] else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=stratify_train
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        logger.info(f"✓ Train set: {X_train.shape}")
        logger.info(f"✓ Validation set: {X_val.shape}")
        logger.info(f"✓ Test set: {X_test.shape}")
        
        # Save splits
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        save_csv(train_df, self.config['paths']['train_data'])
        save_csv(test_df, self.config['paths']['test_data'])
    
    # ========================================================================
    # FEATURE SELECTION
    # ========================================================================
    
    def initial_feature_filter(self):
        """Remove low-variance and highly correlated features"""
        logger.info('-'*70)
        logger.info('STEP 2: INITIAL FEATURE FILTERING')
        logger.info('-'*70)
        
        if not self.config['feature_selection']['enabled']:
            logger.info("Feature selection disabled")
            return
        
        initial_features = self.X_train.shape[1]
        
        # Variance threshold
        if self.config['feature_selection']['initial_filter']['enabled']:
            threshold = self.config['feature_selection']['initial_filter']['variance_threshold']
            logger.info(f"Removing low-variance features (threshold={threshold})...")
            
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(self.X_train)
            
            selected_mask = selector.get_support()
            selected_cols = self.X_train.columns[selected_mask].tolist()
            
            self.X_train = self.X_train[selected_cols]
            self.X_val = self.X_val[selected_cols]
            self.X_test = self.X_test[selected_cols]
            
            removed = initial_features - len(selected_cols)
            logger.info(f"✓ Removed {removed} low-variance features")
        
        # Correlation filter
        if self.config['feature_selection']['correlation_filter']['enabled']:
            threshold = self.config['feature_selection']['correlation_filter']['threshold']
            logger.info(f"Removing highly correlated features (threshold={threshold})...")
            
            corr_matrix = self.X_train.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
            
            self.X_train = self.X_train.drop(columns=to_drop)
            self.X_val = self.X_val.drop(columns=to_drop)
            self.X_test = self.X_test.drop(columns=to_drop)
            
            logger.info(f"✓ Removed {len(to_drop)} highly correlated features")
        
        final_features = self.X_train.shape[1]
        logger.info(f"Features after filtering: {initial_features} → {final_features}")
    
    def analyze_feature_importance(self, model, X, y, model_name: str):
        """Comprehensive feature importance analysis"""
        logger.info(f"Analyzing feature importance for {model_name}...")
        
        importance_dict = {}
        
        # Tree-based importance
        if hasattr(model, 'feature_importances_'):
            importance_dict['tree_importance'] = model.feature_importances_
            logger.info("✓ Tree importance extracted")
        
        # Permutation importance
        if self.config['explainability']['feature_importance']['methods'] and 'permutation' in \
           self.config['explainability']['feature_importance']['methods']:
            n_repeats = self.config['explainability']['feature_importance']['n_repeats']
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            importance_dict['perm_importance'] = perm_importance.importances_mean
            logger.info("✓ Permutation importance calculated")
        
        # SHAP values
        if SHAP_AVAILABLE and self.config['explainability']['shap']['enabled']:
            try:
                sample_size = min(len(X), self.config['explainability']['shap']['sample_size'])
                X_sample = X.sample(n=sample_size, random_state=42)
                
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification
                    
                    importance_dict['shap_importance'] = np.abs(shap_values).mean(axis=0)
                    logger.info("✓ SHAP values calculated")
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")
        
        # Combine importances
        importance_df = pd.DataFrame({
            'feature': X.columns
        })
        
        for imp_name, imp_values in importance_dict.items():
            importance_df[imp_name] = imp_values
        
        # Average rank across methods
        rank_columns = [col for col in importance_df.columns if col != 'feature']
        for col in rank_columns:
            importance_df[f'{col}_rank'] = importance_df[col].rank(ascending=False)
        
        rank_cols = [col for col in importance_df.columns if col.endswith('_rank')]
        importance_df['avg_rank'] = importance_df[rank_cols].mean(axis=1)
        importance_df = importance_df.sort_values('avg_rank')
        
        return importance_df
    
    def select_features(self, importance_df):
        """Select top features based on importance"""
        logger.info('-'*70)
        logger.info('PERFORMING FEATURE SELECTION')
        logger.info('-'*70)
        
        config = self.config['feature_selection']['iterative_selection']
        top_k = config['top_k_features']
        
        # Select top K features
        selected_features = importance_df.head(top_k)['feature'].tolist()
        
        logger.info(f"✓ Selected {len(selected_features)} features")
        logger.info(f"Top 10 features:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: rank={row['avg_rank']:.1f}")
        
        self.selected_features = selected_features
        self.feature_importance_df = importance_df
        
        # Save
        importance_df.to_csv(self.config['paths']['feature_importance_path'], index=False)
        
        with open(self.config['paths']['selected_features_path'], 'w') as f:
            json.dump({'selected_features': selected_features}, f, indent=4)
        
        return selected_features
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    
    def get_model_and_params(self, model_name: str, phase: str):
        """Get model instance and hyperparameter grid"""
        model_config = self.config['models'][model_name]
        
        if not model_config['enabled']:
            return None, None
        
        # Import model class
        class_path = model_config['class']
        module_name, class_name = class_path.rsplit('.', 1)
        
        if 'xgboost' in module_name and not XGBOOST_AVAILABLE:
            logger.warning(f"XGBoost not available, skipping {model_name}")
            return None, None
        
        if 'lightgbm' in module_name and not LIGHTGBM_AVAILABLE:
            logger.warning(f"LightGBM not available, skipping {model_name}")
            return None, None
        
        # Get model class
        if 'sklearn' in module_name:
            from sklearn import linear_model, ensemble, tree
            module = eval(module_name.split('.')[-1])
        elif 'xgboost' in module_name:
            module = xgb
        elif 'lightgbm' in module_name:
            module = lgb
        
        model_class = getattr(module, class_name)
        
        # Initialize model
        init_params = model_config['init_params']
        model = model_class(**init_params)
        
        # Get param grid
        param_grid_key = f'param_grid_{phase}'
        param_grid = model_config.get(param_grid_key, {})
        
        return model, param_grid
    
    def train_model(self, model_name: str, X_train, y_train, X_val, y_val, phase: str):
        """Train a single model with GridSearchCV"""
        logger.info(f"Training {model_name} ({phase})...")
        
        # Get model and params
        model, param_grid = self.get_model_and_params(model_name, phase)
        
        if model is None:
            return None
        
        # Setup cross-validation
        cv_config = self.config['training']
        cv = StratifiedKFold(
            n_splits=cv_config['n_splits'],
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )
        
        # Grid search
        grid_config = self.config['grid_search']
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=grid_config['scoring'],
            refit=grid_config['refit'],
            return_train_score=grid_config['return_train_score'],
            n_jobs=grid_config['n_jobs'],
            verbose=grid_config['verbose'],
            error_score=grid_config['error_score']
        )
        
        # Train
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Validation evaluation
        val_pred = best_model.predict(X_val)
        val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred),
            'recall': recall_score(y_val, val_pred),
            'f1': f1_score(y_val, val_pred),
            'roc_auc': roc_auc_score(y_val, val_pred_proba)
        }
        
        logger.info(f"✓ {model_name} - CV Score: {best_score:.4f}, Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        return {
            'model': best_model,
            'model_name': model_name,
            'best_params': best_params,
            'cv_score': best_score,
            'val_metrics': val_metrics,
            'grid_search': grid_search
        }
    
    def train_all_models(self, phase: str, X_train, y_train, X_val, y_val):
        """Train all enabled models"""
        logger.info('-'*70)
        logger.info(f'TRAINING MODELS - {phase.upper()}')
        logger.info('-'*70)
        
        results = {}
        
        for model_name in self.config['models'].keys():
            if not self.config['models'][model_name]['enabled']:
                continue
            
            with mlflow.start_run(run_name=f"{model_name}_{phase}", nested=True):
                # Log phase and config
                mlflow.log_param("phase", phase)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("n_samples", X_train.shape[0])
                
                # Train
                result = self.train_model(model_name, X_train, y_train, X_val, y_val, phase)
                
                if result is None:
                    continue
                
                # Log metrics
                mlflow.log_metric("cv_score", result['cv_score'])
                for metric_name, metric_value in result['val_metrics'].items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
                
                # Log params
                for param_name, param_value in result['best_params'].items():
                    mlflow.log_param(param_name, param_value)
                
                # Save model
                model_path = f"{self.config['paths']['models_dir']}/{model_name}_{phase}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
                mlflow.log_artifact(model_path)
                
                results[model_name] = result
        
        return results
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_pipeline(self):
        """Execute complete training pipeline"""
        logger.info('\n' + '='*70)
        logger.info('STARTING MODEL TRAINING PIPELINE')
        logger.info('='*70)
        
        with mlflow.start_run(run_name="churn_training_pipeline"):
            # Log config
            mlflow.log_params({
                'project_version': self.config['project']['version'],
                'random_seed': self.config['reproducibility']['random_seed']
            })
            
            # Phase 1: Load and split data
            self.load_and_split_data()
            
            # Phase 2: Initial feature filtering
            self.initial_feature_filter()
            
            # Phase 3: Train initial models with all features
            if self.config['training']['phases']['phase_1']['enabled']:
                logger.info('\n' + '='*70)
                logger.info('PHASE 1: INITIAL TRAINING (ALL FEATURES)')
                logger.info('='*70)
                
                phase1_results = self.train_all_models(
                    'phase1', self.X_train, self.y_train, self.X_val, self.y_val
                )
                
                # Find best model
                best_model_name = max(
                    phase1_results.keys(),
                    key=lambda x: phase1_results[x]['val_metrics']['roc_auc']
                )
                best_model_result = phase1_results[best_model_name]
                
                logger.info(f"\n✓ Best Phase 1 Model: {best_model_name}")
                logger.info(f"  ROC-AUC: {best_model_result['val_metrics']['roc_auc']:.4f}")
                
                # Phase 4: Feature importance analysis
                if self.config['training']['phases']['phase_2']['enabled']:
                    logger.info('\n' + '='*70)
                    logger.info('PHASE 2: FEATURE IMPORTANCE ANALYSIS')
                    logger.info('='*70)
                    
                    importance_df = self.analyze_feature_importance(
                        best_model_result['model'],
                        self.X_val,
                        self.y_val,
                        best_model_name
                    )
                    
                    selected_features = self.select_features(importance_df)
                    
                    # Update datasets with selected features
                    self.X_train = self.X_train[selected_features]
                    self.X_val = self.X_val[selected_features]
                    self.X_test = self.X_test[selected_features]
            
            # Phase 5: Retrain with selected features
            if self.config['training']['phases']['phase_3']['enabled'] and self.selected_features:
                logger.info('\n' + '='*70)
                logger.info('PHASE 3: RETRAIN WITH SELECTED FEATURES')
                logger.info('='*70)
                
                phase3_results = self.train_all_models(
                    'phase3', self.X_train, self.y_train, self.X_val, self.y_val
                )
                
                # Find best model
                best_model_name = max(
                    phase3_results.keys(),
                    key=lambda x: phase3_results[x]['val_metrics']['roc_auc']
                )
                best_model_result = phase3_results[best_model_name]
                
                self.best_model = best_model_result['model']
                self.best_model_name = best_model_name
                self.best_score = best_model_result['val_metrics']['roc_auc']
                
                logger.info(f"\n✓ Best Phase 3 Model: {best_model_name}")
                logger.info(f"  ROC-AUC: {self.best_score:.4f}")
            
            # Final evaluation on test set
            self.evaluate_final_model()
            
            logger.info('\n' + '='*70)
            logger.info('✓ PIPELINE COMPLETED SUCCESSFULLY')
            logger.info('='*70)
    
    def evaluate_final_model(self):
        """Evaluate best model on test set"""
        logger.info('\n' + '='*70)
        logger.info('FINAL MODEL EVALUATION ON TEST SET')
        logger.info('='*70)
        
        if self.best_model is None:
            logger.warning("No best model found")
            return
        
        # Predict
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        test_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        logger.info(f"\nFinal Model: {self.best_model_name}")
        logger.info(f"Test Set Performance:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            mlflow.log_metric(f"test_{metric}", value)
        
        # Classification report
        report = classification_report(self.y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Save best model
        best_model_path = self.config['paths']['best_model_path']
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        mlflow.log_artifact(best_model_path)
        
        logger.info(f"✓ Best model saved to {best_model_path}")


def main():
    """Main execution"""
    trainer = ChurnModelTrainer('config/model_training_config.yaml')
    trainer.run_pipeline()


if __name__ == '__main__':
    main()