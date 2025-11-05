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
    train_test_split, GridSearchCV, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
)
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.inspection import permutation_importance
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
        log.info(f'{'INITIALIZING CHURN MODEL TRAINING PIPELINE':_^30}')
        
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
    def laod_and_split_data(self):
        '''Load engineered data and split train set into train/val sets'''
        log.info(f'{'LOADING AND SPLITTING DATA':_^39}')

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
            threshold = self.config['feature_selection']['correlation_filter']['enabled']
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
            log.info(f'{row['feature']}: rank={row['avg_rank']:.1f}')

        self.selected_features = selected_features
        self.feature_importance_df = importance_df

        importance_df.to_csv(self.config['paths']['feature_importance_path'], index=False)
        
        with open(self.config['paths']['selected_features_path'], 'w') as f:
            json.dump({'selected_features': selected_features}, f, indent=4)
        
        return selected_features
