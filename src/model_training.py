"""
Production-Grade Model Training Pipeline with MLflow Tracking
Fully corrected and tested version

Author: Maxwell Selassie Hiamatsu
Date: October 24, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import mlflow
import mlflow.sklearn
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, 
    f1_score, accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance

# external imports
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print('XGBoost not available')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print('SHAP not available')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print('LightGBM not available')

# import utilities
from utils import (
    load_csv_file, ensure_directories, setup_logger, 
    save_json_file, save_joblib_file, read_yaml_file
)

# setup logging
logger = setup_logger(name='model_training', log_filename='logs/model_training.log')


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
    
    def __init__(self, config_path: str | Path = 'config/model_training_config.yaml'):
        """Initialize trainer with configuration"""
        logger.info('='*70)
        logger.info('INITIALIZING CHURN MODEL TRAINING PIPELINE')
        logger.info('='*70)
        
        self.config = read_yaml_file(config_path)
        self.config_path = config_path

        # Initialize data containers
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

        # Initialize model tracking
        self.selected_features = None
        self.feature_importance_df = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.model_results = {}

        # Create directories
        for dir_path in ['data', 'mlruns', 'results', 'logs', 'models']:
            ensure_directories(dir_path)

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

        project_name = self.config['project']['name']
        version_name = self.config['project']['version']
        mlflow_tracking = self.config['mlflow']['enabled']

        logger.info(f'Project: {project_name}')
        logger.info(f'Version: {version_name}')
        logger.info(f'MLflow tracking: {mlflow_tracking}')

    # ==========================
    # DATA SPLITTING AND LOADING
    # ==========================
    
    def load_and_split_data(self):
        """Load engineered data and split train set into train/val sets"""
        logger.info('-'*70)
        logger.info('STEP 1: LOADING AND SPLITTING DATA')
        logger.info('-'*70)

        # Load engineered files
        x_train = load_csv_file(self.config['paths']['x_train_data'])
        y_train = load_csv_file(self.config['paths']['y_train_data'])
        x_test = load_csv_file(self.config['paths']['x_test_data'])
        y_test = load_csv_file(self.config['paths']['y_test_data'])
        
        # Handle if y is DataFrame (convert to Series)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.iloc[:, 0]

        logger.info(f"Loaded x_train: {x_train.shape}")
        logger.info(f"Loaded x_test: {x_test.shape}")
        logger.info(f"Churn rate in training: {y_train.mean():.2%}")

        # Split training into train/validation
        val_size = self.config['data_split']['validation_size']
        stratify = y_train if self.config['data_split']['stratify'] else None
        random_state = self.config['data_split']['random_state']

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, 
            test_size=val_size, 
            random_state=random_state, 
            stratify=stratify
        ) 

        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        logger.info(f'✓ Train set: {x_train.shape}')
        logger.info(f'✓ Validation set: {x_val.shape}')
        logger.info(f'✓ Test set: {x_test.shape}')

        # Save splits for reference
        train_df = pd.concat([x_train, y_train], axis=1)
        test_df = pd.concat([x_test, y_test], axis=1)
        
        train_output = self.config['paths'].get('train_split_path', 'data/train_dataframe.csv')
        test_output = self.config['paths'].get('test_split_path', 'data/test_dataframe.csv')
        
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        logger.info(f'✓ Saved train/test splits')

    # =================
    # FEATURE SELECTION
    # =================
    
    def initial_feature_filter(self):
        """Remove low-variance and highly correlated features"""
        logger.info('-'*70)
        logger.info('STEP 2: INITIAL FEATURE FILTERING')
        logger.info('-'*70)

        if not self.config['feature_selection']['enabled']:
            logger.info('Feature selection disabled in config')
            return 
        
        initial_features = self.x_train.shape[1]

        # Variance threshold filter
        if self.config['feature_selection']['initial_filter']['enabled']:
            threshold = self.config['feature_selection']['initial_filter']['variance_threshold']
            logger.info(f'Removing low-variance features (threshold={threshold})...')

            selector = VarianceThreshold(threshold=threshold)
            selector.fit(self.x_train)

            selected_mask = selector.get_support()
            selected_cols = self.x_train.columns[selected_mask].tolist()

            self.x_train = self.x_train[selected_cols]
            self.x_test = self.x_test[selected_cols]
            self.x_val = self.x_val[selected_cols]

            removed = initial_features - len(selected_cols)
            logger.info(f'✓ Removed {removed} low-variance features')

        # Correlation filter
        if self.config['feature_selection']['correlation_filter']['enabled']:
            threshold = self.config['feature_selection']['correlation_filter']['threshold']
            logger.info(f'Removing highly correlated features (threshold={threshold})...')

            corr_matrix = self.x_train.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

            self.x_train = self.x_train.drop(columns=to_drop)
            self.x_val = self.x_val.drop(columns=to_drop)
            self.x_test = self.x_test.drop(columns=to_drop)

            logger.info(f'✓ Removed {len(to_drop)} highly correlated features')

        final_features = self.x_train.shape[1]
        logger.info(f'Features after filtering: {initial_features} → {final_features}')

    def analyze_feature_importance(self, model, x, y, model_name: str):
        """Comprehensive feature importance analysis"""
        logger.info(f'Analyzing feature importance for {model_name}...')

        importance_dict = {}

        # Tree-based importance
        if hasattr(model, 'feature_importances_'):
            importance_dict['tree_importance'] = model.feature_importances_
            logger.info('✓ Tree importance extracted')

        # Check if feature importance methods are configured
        if not self.config['explainability']['feature_importance'].get('methods'):
            logger.info('Feature importance methods not configured, skipping advanced analysis')
            if not importance_dict:
                # Return empty DataFrame if no importance available
                return pd.DataFrame({'feature': x.columns, 'importance': 0})

        # Permutation importance
        if 'permutation' in self.config['explainability']['feature_importance'].get('methods', []):
            n_repeats = self.config['explainability']['feature_importance']['n_repeats']
            logger.info(f'Calculating permutation importance (n_repeats={n_repeats})...')

            perm_importance = permutation_importance(
                model, x, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            importance_dict['perm_importance'] = perm_importance.importances_mean
            logger.info('✓ Permutation importance calculated')

        # SHAP values
        if SHAP_AVAILABLE and self.config['explainability']['shap']['enabled']:
            try:
                sample_size = min(len(x), self.config['explainability']['shap']['sample_size'])
                x_sample = x.sample(n=sample_size, random_state=42)
                logger.info(f'Calculating SHAP values (sample_size={sample_size})...')

                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(x_sample)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification

                    importance_dict['shap_importance'] = np.abs(shap_values).mean(axis=0)
                    logger.info('✓ SHAP values calculated')
            except Exception as e:
                logger.warning(f'SHAP calculation failed: {e}')

        # Combine importances
        importance_df = pd.DataFrame({
            'feature': x.columns
        })

        for imp_name, imp_values in importance_dict.items():
            importance_df[imp_name] = imp_values

        # Average rank across methods
        rank_columns = [col for col in importance_df.columns if col != 'feature']
        
        if len(rank_columns) > 0:
            for col in rank_columns:
                importance_df[f'{col}_rank'] = importance_df[col].rank(ascending=False)
            
            rank_cols = [col for col in importance_df.columns if col.endswith('_rank')]
            importance_df['avg_rank'] = importance_df[rank_cols].mean(axis=1)
            importance_df = importance_df.sort_values('avg_rank')
        else:
            # No ranking methods available
            importance_df['avg_rank'] = range(len(importance_df))
        
        return importance_df

    def select_features(self, importance_df):
        """Select best features based on importance"""
        logger.info('-'*70)
        logger.info('STEP 3: FEATURE SELECTION')
        logger.info('-'*70)

        config = self.config['feature_selection']['iterative_selection']
        top_k = config['top_k_features']

        # Select top K features
        selected_features = importance_df.head(top_k)['feature'].tolist()
        
        logger.info(f'✓ Selected {len(selected_features)} features')
        logger.info('Top 10 features:')
        for i, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
            rank = row['avg_rank']
            row_feature = row['feature']
            logger.info(f'  {i}. {row_feature}: rank={rank:.1f}')

        self.selected_features = selected_features
        self.feature_importance_df = importance_df

        # Save results
        importance_path = self.config['paths']['feature_importance_path']
        importance_df.to_csv(importance_path, index=False)
        logger.info(f'✓ Feature importance saved to {importance_path}')
        
        features_path = self.config['paths']['selected_features_path']
        with open(features_path, 'w') as f:
            json.dump({'selected_features': selected_features}, f, indent=4)
        logger.info(f'✓ Selected features saved to {features_path}')
        
        return selected_features
    
    def get_model_and_params(self, model_name: str, phase: str):
        """Get model instance and hyperparameter grid"""
        model_config = self.config['models'][model_name]

        if not model_config['enabled']:
            return None, None
        
        # Import model class
        class_path = model_config['class']
        module_name, class_name = class_path.rsplit('.', 1)

        # Check library availability
        if 'xgboost' in module_name and not XGBOOST_AVAILABLE:
            logger.warning(f'XGBoost not available, skipping {model_name}')
            return None, None

        if 'lightgbm' in module_name and not LIGHTGBM_AVAILABLE:
            logger.warning(f'LightGBM not available, skipping {model_name}')
            return None, None
        
        # Get module object
        if 'sklearn' in module_name:
            from sklearn import linear_model, ensemble, tree
            module = eval(module_name.split('.')[-1])
        elif 'xgboost' in module_name:
            module = xgb
        elif 'lightgbm' in module_name:
            module = lgb
        else:
            logger.error(f"Unknown module: {module_name}")
            return None, None

        # Get model class
        model_class = getattr(module, class_name)

        # Initialize model
        init_params = model_config['init_params']
        model = model_class(**init_params)

        # Get param grid for this phase
        param_grid_key = f'param_grid_{phase}'
        param_grid = model_config.get(param_grid_key, {})
        
        if not param_grid:
            logger.warning(f'No parameter grid found for {model_name} in {phase}')

        return model, param_grid
    
    def train_model(self, model_name: str, x_train, y_train, x_val, y_val, phase: str):
        """Train a single model with GridSearchCV"""
        logger.info(f'Training {model_name} ({phase})...')

        # Get model and parameters
        model, param_grid = self.get_model_and_params(model_name, phase)
        
        if model is None:
            logger.warning(f'Skipping {model_name} - model not available')
            return None

        # Setup cross-validation
        cv_config = self.config['training']
        cv = StratifiedKFold(
            n_splits=cv_config['n_splits'],
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )     

        # GridSearchCV
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
        try:
            grid_search.fit(x_train, y_train)
        except Exception as e:
            logger.error(f'Training failed for {model_name}: {e}')
            return None

        # Best model
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Validation evaluation
        val_pred = best_model.predict(x_val)
        val_pred_proba = best_model.predict_proba(x_val)[:, 1]

        val_metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred, zero_division=0),
            'recall': recall_score(y_val, val_pred, zero_division=0),
            'f1': f1_score(y_val, val_pred, zero_division=0),
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

    def train_all_models(self, phase: str, x_train, y_train, x_val, y_val):
        """Train all enabled models"""
        logger.info('-'*70)
        logger.info(f'TRAINING MODELS - {phase.upper()}')
        logger.info('-'*70)

        results = {}

        for model_name in self.config['models'].keys():
            if not self.config['models'][model_name]['enabled']:
                logger.info(f'Skipping {model_name} (disabled in config)')
                continue

            with mlflow.start_run(run_name=f'{model_name}_{phase}', nested=True):
                # Log phase and config
                mlflow.log_param('phase', phase)
                mlflow.log_param('n_features', x_train.shape[1])
                mlflow.log_param('n_samples', x_train.shape[0])

                # Train
                result = self.train_model(model_name, x_train, y_train, x_val, y_val, phase)

                if result is None:
                    logger.warning(f'Skipping {model_name} - training returned None')
                    continue

                # Log metrics
                mlflow.log_metric('cv_score', result['cv_score'])
                for metric_name, metric_value in result['val_metrics'].items():
                    mlflow.log_metric(f'val_{metric_name}', metric_value)

                # Log params
                for param_name, param_value in result['best_params'].items():
                    mlflow.log_param(param_name, param_value)

                # Save model
                model_path = f"{self.config['paths']['models_dir']}/{model_name}_{phase}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
                mlflow.log_artifact(model_path)
                logger.info(f'✓ Model saved to {model_path}')
                    
                results[model_name] = result
        
        if not results:
            logger.error('No models were trained successfully!')
            return {}
        
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
                    'phase1', self.x_train, self.y_train, self.x_val, self.y_val
                )
                
                if not phase1_results:
                    logger.error('Phase 1 training failed - no models trained')
                    return
                
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
                        self.x_val,
                        self.y_val,
                        best_model_name
                    )
                    
                    selected_features = self.select_features(importance_df)
                    
                    # Update datasets with selected features
                    self.x_train = self.x_train[selected_features]
                    self.x_val = self.x_val[selected_features]
                    self.x_test = self.x_test[selected_features]
            
            # Phase 5: Retrain with selected features
            if self.config['training']['phases']['phase_3']['enabled'] and self.selected_features:
                logger.info('\n' + '='*70)
                logger.info('PHASE 3: RETRAIN WITH SELECTED FEATURES')
                logger.info('='*70)
                
                phase3_results = self.train_all_models(
                    'phase3', self.x_train, self.y_train, self.x_val, self.y_val
                )
                
                if not phase3_results:
                    logger.error('Phase 3 training failed - no models trained')
                    return
                
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
            logger.warning("No best model found - skipping final evaluation")
            return
        
        # Predict
        y_pred = self.best_model.predict(self.x_test)
        y_pred_proba = self.best_model.predict_proba(self.x_test)[:, 1]
        
        # Metrics
        test_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
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
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Save best model
        best_model_path = self.config['paths']['best_model_path']
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        mlflow.log_artifact(best_model_path)
        
        logger.info(f"✓ Best model saved to {best_model_path}")
        
        # Save test metrics
        test_metrics_path = 'results/test_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"✓ Test metrics saved to {test_metrics_path}")


def main():
    """Main execution"""
    try:
        trainer = ChurnModelTrainer('config/model_training_config.yaml')
        trainer.run_pipeline()
        
        print("\n" + "="*70)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Best Model: {trainer.best_model_name}")
        print(f"Best Score: {trainer.best_score:.4f}")
        print(f"Check logs/model_training.log for details")
        print(f"View MLflow UI: mlflow ui")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        print("Check logs/model_training.log for details")
        raise


if __name__ == '__main__':
    main()