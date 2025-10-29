import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')
from utils import (
    setup_logger, get_timestamp, ensure_directories, read_yaml_file
)

log = setup_logger(name='preprocessing', log_filename='logs/ preprocessing.log',level=logging.INFO)

class DataPreprocesserPipeline():
    '''Data preprocessing Pipeline driven by YAML configuration
    
    Workflow: 
        - Handling missing values
        - Handling duplicates
        - Handling outliers
        - Feature Scaling
        - Featur Encoding
        
    '''
    def __init__(self, config_file: str | Path) -> None:
        '''Initialize DataPreprocessorPipeline
        
        Args:
            config_file : YAML configuration file path
            
        '''
        log.info("="*50)
        log.info('INITIALIZING DATA PREPROCESSING PIPELINE')
        log.info("="*50)

        self.config = read_yaml_file(config_file)
        self.config_file = config_file

        self.scalar = None

        # project metadata
        self.log_transformations = []
        self.metadata = {
            'config_version' : self.config['project']['version'],
            'config_filepath' : self.config_file,
            'timestamp_start' : get_timestamp()
        }

        log.info(f"Project name : {self.config['project']}")
        log.info(f"Version : {self.config["project"]["version"]}")
        log.info(f"Configuration file successfully loaded from {Path(config_file)}")

    def log_transformations(self, step: str, details: Dict[str,Any]):
        '''Log transformation steps for audit trails'''
        entry = {
            'step' : step,
            'timestamp' : get_timestamp(),
            **details
        }
        self.log_transformations.append(entry)
        log.info(f"{step} : {details}")

    def validate_data(self, df: pd.DataFrame):
        '''Initial data validation'''
        log.info("="*50)
        log.info('DATA VALIDATION')
        log.info("="*50)

        if df.empty:
            log.error('DataFrame is Empty')
            raise ValueError('DataFrane is Empty!')
        
        self.log_transformations('data_validation',{
            'initial_rows' : df.shape[0],
            'initial_features' : df.shape[1],
            'memory_usage' : round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)
        })
        return df
    
    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Drop columns based on Config'''
        log.info("="*50)
        log.info("DROP COLUMNS")
        log.info("="*50)

        if 'columns_to_drop' not in self.config or not self.config["columns_to_drop"]:
            log.info('No columns to drop')
            return df
        
        cols_to_drop = []
        for item in self.config["columns_to_drop"]:
            col_name = item['name']
            reason = item['reason']
        
            if col_name in df.columns:
                cols_to_drop.append(col_name)
                log.info(f"Dropping {col_name} : {reason}")
            else:
                log.error(f'Column {col_name} not found in dataframe')

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.log_transformations('Drop_columns', {
                'dropped_columns' : cols_to_drop,
                'remaining_columns' : df.shape[1]
            })

        return df
    
    def handling_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Handle missing values in dataframe'''
        log.info("="*50)
        log.info("HANDLING MISSNG VALUES")
        log.info("="*50) 

        missing_before = df.isnull().sum().sum()
        log.info(f"Missing values before : {missing_before}")

        if 'missing_values' in self.config or self.config['missing_values']:
            for col, strategy in self.config['missing_values'].items():
                if col not in df.columns:
                    log.warning(f'Column "{col}" not in dataFrame')
                    continue

                missing_count = df[col].isnull().sum()
                if len(missing_count) == 0:
                    log.info(f'Column "{col}" has no missing values')
                    continue

                method = strategy['method']
                log.info(f'-{col} : {missing_count} missing values | Method : {method}')

                if method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == 'mode':
                    df[col].fillna(df[col].mode(),inplace=True)
        else:
            log.error(f"No missing columns")
            raise ValueError(f'No missing columns!')
        
        missing_after = df.isnull().sum().sum()
        log.info(f'Total missing values after: {missing_after}')

        self.log_transformations('handle_missing_values', {
            'missing_before' : int(missing_before),
            'missing_after' : int(missing_after),
            'missing_removed' : int(missing_before - missing_after)
        })

        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Handle outliers in the dataset'''
        log.info('='*50)
        log.info('HANDLING OUTLIERS')
        log.info('='*50)

        if 'outliers' not in self.config or not self.config['outliers']:
            log.warning('No outliers in the dataset')
            return df
        
        for col, strategy in self.config['outliers'].items():
            if col not in df.columns:
                log.warning(f'-{col} column not found in the dataset')
                continue
        
            action = strategy['action']
            reason = strategy['reason']

            log.info(f'{col} : Action: {action} | Reason : {reason}')

            if action == 'none':
                continue
            elif action == 'cap':
                df[col] = np.clip()