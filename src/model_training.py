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
