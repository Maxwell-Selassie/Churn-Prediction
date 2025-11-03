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
    def __init__(self, config_file: str | Path = 'config/model_training_config.yaml'):
        '''Initialize trainer with configuration'''
        log.info(f'INITIALIZING CHURN MODEL TRAINING PIPELINE')