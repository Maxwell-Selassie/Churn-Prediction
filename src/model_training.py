import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import mlflow
import mlflow.sklearn
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# sklearn imports
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, f1_score,accuracy_score,
    brier_score_loss, log_loss, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
)
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.inspection import permutation_importance
import pickle
# external imports
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print('XGBoost Not Available')

try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    print('SHAP not available')

try:
    import lightgbm
    light_gbm_available = True
except ImportError:
    light_gbm_available = False
    print('LightGBM not found')
# import utilities
from utils import (
    load_csv_file, ensure_directories, load_joblib_file, load_json_file,
    setup_logger, save_json_file, save_joblib_file, read_yaml_file, save_yaml_file
)

# setup logging
log = setup_logger(name='model_training',log_filename='logs/model_training.log')

class ChurnModelTrainer:
    '''
    Production-grade model training pipeline with MLFlow tracking

    Features:
    - Multi-phase training (baseline → feature selection → retraining → tuning)
    - Automated feature importance analysis
    - MLflow experiment tracking
    - Model registry integration
    - SHAP explainability
    - Cross-validation
    - Model comparison dashboard
    '''
    def __init__(self, config_path: str | Path = 'config/model_training_config.yaml'):
        '''Initialize trainer with configuration'''
        log.info(f'INITIALIZING CHURN MODEL TRAINING PIPELINE')
        
        self.config = read_yaml_file(config_path)
        self.config_path = config_path

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

        self.selected_features = None
        self.feature_importance_df = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf

        self.model_results = {}

        # create directories
        for dir_path in ['data','mlrun','results','logs']:
            ensure_directories(dir_path)

        project_name = self.config['project']['name']
        version_name = self.config['project']['version']
        mlflow_tracking = self.config['mlflow']['enabled']

        log.info(f'Project : {project_name}')
        log.info(f'Version : {version_name}')
        log.info(f'MLFlow tracking : { mlflow_tracking}')

    # ==========================
    # DATA SPLITTING AND LOADING
    # ==========================
    def load_and_split_data(self):
        '''Load engineered data and split train set into train/val sets'''
        log.info('LOADING AND SPLITTING DATA')

        # load engineered file
        X_train = load_csv_file(self.config['paths']['x_train_data'])
        Y_train = load_csv_file(self.config['paths']['y_train_data'])
        X_test = load_csv_file(self.config['paths']['x_test_data'])
        Y_test = load_csv_file(self.config['paths']['y_test_data'])

        val_size = self.config['data_split']['validation_size']
        stratify = Y_train if self.config['data_split']['stratify'] else None
        random_state = self.config['data_split']['random_state']

        x_train, x_val, y_train, y_val = train_test_split(
            X_train, Y_train, test_size=val_size, random_state=random_state, stratify=stratify
        ) 

        self.x_train = x_train
        self.x_test = X_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = Y_test
        self.y_val = y_val

        log.info(f'Train set : {x_train.shape}')
        log.info(f'Test set : {X_test.shape}')
        log.info(f'Validation set : {x_val.shape}')

        train_df = pd.concat([x_train, y_train], axis=1)
        test_df = pd.concat([X_test, Y_test], axis=1)
        train_df.to_csv('data/train_dataframe.csv',index=False)
        test_df.to_csv('data/test_dataframe.csv',index=False)

    # =================
    # FEATURE SELECTION
    # =================
    def initial_feature_filter(self):
        '''Remove low-variance and highly correlated features'''
        log.info('='*50)
        log.info(f'INITIAL FEATURE FILTERING')
        log.info('='*50)

        if not self.config['feature_selection']['enabled']:
            log.info('Feature selection disabled')
            return 
        
        initial_features = self.x_train.shape[1]

        # variance threshold
        if self.config['feature_selection']['initial_filter']['enabled']:
            threshold = self.config['feature_selection']['initial_filter']['variance_threshold']
            log.info(f'Removing low-variance features (threshold = {threshold})')

            selector = VarianceThreshold(threshold=threshold)
            selector.fit(self.x_train)

            selected_mask = selector.get_support()
            selected_cols = self.x_train.columns[selected_mask].tolist()

            self.x_train = self.x_train[selected_cols]
            self.x_test = self.x_test[selected_cols]
            self.x_val = self.x_val[selected_cols]

            removed = initial_features - len(selected_cols)
            log.info(f'Removed {removed} quasi-constant features')

        # correlation features
        if self.config['feature_selection']['correlation_filter']['enabled']:
            threshold = self.config['feature_selection']['correlation_filter']['threshold']
            log.info(f'Removing highly correlated features (threshold = {threshold})')

            corr_matrix = self.x_train.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

            self.x_train = self.x_train.drop(columns=to_drop)
            self.x_val = self.x_val.drop(columns=to_drop)
            self.x_test = self.x_test.drop(columns=to_drop)

            log.info(f'Removed {len(to_drop)} highly correlated features')

        final_features = self.x_train.shape[1]
        log.info(f'Features after filtering: {initial_features} -> {final_features}')


    def analyze_feature_importance(self, model, x, y, model_name: str):
        '''Comprehensive feature importance analysis'''
        log.info(f'Analyzing feature importance for {model_name}')

        importance_dict = {}

        # tree based models
        if hasattr(model, 'feature_importances_'):
            importance_dict['tree_importance'] = model.feature_importances_
            log.info('Tree importance extracted')

        # permutation importance
        if not self.config['explainability']['feature_importance']['methods']:
            log.info(f'Feature importance disabled!')

        if 'permutation' in self.config['explainability']['feature_importance']['methods']:
            n_repeats = self.config['explainability']['feature_importance']['n_repeats']

            perm_importance = permutation_importance(
                model, x, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            importance_dict['perm_importance'] = perm_importance.importances_mean
            log.info(f'Permutation importance calculated!')

        # SHAP values
        if shap_available and self.config['explainability']['shap']['enabled']:
            try:
                sample_size = min(len(x), self.config['explainability']['shap']['sample_size'])
                x_sample = x.sample(n=sample_size, random_state = 42)

                if hasattr(model,'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(x_sample)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]

                    importance_dict['shap_importance'] = np.abs(shap_values).mean(axis=0)
                    log.info('SHAP values calculated')
            except Exception as e:
                log.warning(f'SHAP calculation failed: {e}')


        # combine importances
        importance_df = pd.DataFrame({
            'features' : x.columns
        })

        for imp_name, imp_values in importance_dict.items():
            importance_df[imp_name] = imp_values

        # average rank across methods
        rank_columns = [col for col in importance_df.columns if col != 'feature']
        for col in rank_columns:
            importance_df[f'{col}_rank'] = importance_df[col].rank(ascending=False)
        
        rank_cols = [col for col in importance_df.columns if col.endswith('_rank')]
        importance_df['avg_rank'] = importance_df[rank_cols].mean(axis=1)
        importance_df = importance_df.sort_values('avg_rank')
        
        return importance_df

    def select_features(self, importance_df):
        '''select best features based on importance'''
        log.info('='*50)
        log.info('PERFORMING FEATURE SELECTION')
        log.info('='*50)

        config = self.config['feature_selection']['iterative_selection']
        top_k = self.config['feature_selection']['top_k_features']

        selected_features = importance_df.head(top_k)['feature'].tolist()
        log.info(f'Selected {len(selected_features)} features')
        log.info('Top 10 features : ')
        for i, row in importance_df.head(10).iterrows():
            rank = row['avg_rank']
            row_feature = row['feature']
            log.info(f'{row_feature}: rank={rank:.1f}')

        self.selected_features = selected_features
        self.feature_importance_df = importance_df

        importance_df.to_csv(self.config['paths']['feature_importance_path'], index=False)
        
        with open(self.config['paths']['selected_features_path'], 'w') as f:
            json.dump({'selected_features': selected_features}, f, indent=4)
        
        return selected_features
    
    def get_params_and_model(self, model_name: str, phase: str):
        '''Get model instance and hyperparameter grid'''
        model_config = self.config['models'][model_name]

        if not model_config['enabled']:
            return None, None
        
        # import model class
        class_path = model_config['class']
        module_name, class_name = class_path.rsplit('.',1)

        if 'xgboost' in module_name and not xgboost_available:
            log.warning(f'XGBoost not available. Skipping {model_name}')
            return None, None

        if 'lightgbm' in module_name and not light_gbm_available:
            log.warning(f'LightGBM not available. Skipping {model_name}')
            return None, None
        
        # get model class
        if 'sklearn' in module_name:
            from sklearn import linear_model, ensemble, tree
            module = eval(module_name.split('.')[-1])
        elif 'xgboost' in module_name:
            module = XGBClassifier

        elif 'lightgbm' in module_name:
            module = lightgbm

        model_class = getattr(module, class_name)

        # initialize model
        init_params = model_config['init_params']
        model = model_class(**init_params)

        # get params grid
        params_grid_key = f'params_grid_{phase}'
        params_grid = model_config.get(params_grid_key, {})

        return model, params_grid
    
    def train_model(self, model_name: str, x_train, y_train, x_val, y_val, phase: str):
        '''Train a single model with GridSearchCV'''
        log.info(f'Training {model_name} : ({phase})...')

        # get model and parameters
        model, params_grid = self.get_params_and_model(model_name, phase)
        if model is None:
            return None

        # setup cross-validation
        cv_config = self.config['training']
        cv = StratifiedKFold(
            n_splits=cv_config.get('n_splits',5),
            shuffle=cv_config.get('shuffle', True),
            random_state=cv_config.get('random_state',1)
        )     

        # GridSearchCV
        grid_config = self.config['grid_search']
        grid_search = GridSearchCV(
            estimator= model,
            param_grid=params_grid,
            cv=cv,
            scoring=grid_config.get('scoring','f1'),
            refit= grid_config.get('refit', True),
            return_train_score=grid_config.get('return_train_score',True),
            n_jobs= grid_config.get('n_jobs',-1),
            verbose=grid_config.get('verbose',1),
            error_score=grid_config.get('error_score','raise')
        )

        grid_search.fit(x_train, y_train)

        # best model
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # validation evaluation
        val_pred = best_model.predict(x_val)
        val_pred_proba = best_model.predict_proba(x_val)[:,1]

        val_metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'precision': precision_score(y_val, val_pred),
            'recall': recall_score(y_val, val_pred),
            'f1': f1_score(y_val, val_pred),
            'roc_auc': roc_auc_score(y_val, val_pred_proba)
        }

        log.info(f"✓ {model_name} - CV Score: {best_score:.4f}, Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        return {
            'model': best_model,
            'model_name': model_name,
            'best_params': best_params,
            'cv_score': best_score,
            'val_metrics': val_metrics,
            'grid_search': grid_search
        }

    def train_all_models(self,phase: str, x_train, y_train, x_val, y_val):
        '''Train all enabled models'''
        log.info('='*50)
        log.info(f'TRAINING MODELS - {phase.upper()}')
        log.info('='*50)

        results = {}

        for model_name in self.config['models'].keys():
            if not self.config['models'][model_name]['enabled']:
                continue

            with mlflow.start_run(run_name=f'{model_name}_{phase}', nested=True):
                # log phase and config
                mlflow.log_param('phase',phase)
                mlflow.log_param('n_features',x_train.shape[1])
                mlflow.log_param('n_samples',x_train.shape[0])

            # train
            result = self.train_model(model_name, x_train, y_train, x_val, y_val)

            if result is None:
                continue

            # log metrics
            mlflow.log_metric('cv_score', results['cv_score'])
            for metric_name, metric_value in results['val_metrics'].items():
                mlflow.metric(f'val_{metric_name}', metric_value)

            # log params
            for params_name, params_value in results['best_params'].items():
                mlflow.log_param(params_name, params_value)

            # save model
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
        log.info('\n' + '='*70)
        log.info('STARTING MODEL TRAINING PIPELINE')
        log.info('='*70)
        
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
                log.info('\n' + '='*70)
                log.info('PHASE 1: INITIAL TRAINING (ALL FEATURES)')
                log.info('='*70)
                
                phase1_results = self.train_all_models(
                    'phase1', self.x_train, self.y_train, self.x_val, self.y_val
                )
                
                # Find best model
                best_model_name = max(
                    phase1_results.keys(),
                    key=lambda x: phase1_results[x]['val_metrics']['roc_auc']
                )
                best_model_result = phase1_results[best_model_name]
                
                log.info(f"\n✓ Best Phase 1 Model: {best_model_name}")
                log.info(f"  ROC-AUC: {best_model_result['val_metrics']['roc_auc']:.4f}")
                
                # Phase 4: Feature importance analysis
                if self.config['training']['phases']['phase_2']['enabled']:
                    log.info('\n' + '='*70)
                    log.info('PHASE 2: FEATURE IMPORTANCE ANALYSIS')
                    log.info('='*70)
                    
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
                log.info('\n' + '='*70)
                log.info('PHASE 3: RETRAIN WITH SELECTED FEATURES')
                log.info('='*70)
                
                phase3_results = self.train_all_models(
                    'phase3', self.x_train, self.y_train, self.x_val, self.y_val
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
                
                log.info(f"\n✓ Best Phase 3 Model: {best_model_name}")
                log.info(f"  ROC-AUC: {self.best_score:.4f}")
            
            # Final evaluation on test set
            self.evaluate_final_model()
            
            log.info('\n' + '='*70)
            log.info('✓ PIPELINE COMPLETED SUCCESSFULLY')
            log.info('='*70)
    
    def evaluate_final_model(self):
        """Evaluate best model on test set"""
        log.info('\n' + '='*70)
        log.info('FINAL MODEL EVALUATION ON TEST SET')
        log.info('='*70)
        
        if self.best_model is None:
            log.warning("No best model found")
            return
        
        # Predict
        y_pred = self.best_model.predict(self.x_test)
        y_pred_proba = self.best_model.predict_proba(self.x_test)[:, 1]
        
        # Metrics
        test_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        log.info(f"\nFinal Model: {self.best_model_name}")
        log.info(f"Test Set Performance:")
        for metric, value in test_metrics.items():
            log.info(f"  {metric}: {value:.4f}")
            mlflow.log_metric(f"test_{metric}", value)
        
        # Classification report
        report = classification_report(self.y_test, y_pred)
        log.info(f"\nClassification Report:\n{report}")
        
        # Save best model
        best_model_path = self.config['paths']['best_model_path']
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        mlflow.log_artifact(best_model_path)
        
        log.info(f"✓ Best model saved to {best_model_path}")


def main():
    """Main execution"""
    trainer = ChurnModelTrainer('config/model_training_config.yaml')
    trainer.run_pipeline()


if __name__ == '__main__':
    main()