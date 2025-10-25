'''
helper functions for this churn prediction end-to-end machine learning project
    * Basic helper functions
    - Setup folder and file paths
    - Setup logging

    * I/O helper functions
    - Load csv file
    - Load json file
    - Load yaml config file
    - Load joblib file
    - Save csv file
    - Save yaml config file
    - Save json file
    - Save joblib file
    - Ensure directories exist

    * Other helper functions
    - Project metadata
    - Timestamp
    - validate dataframe and required columns
    - get memory usage
    - generate quick data profile
'''
# import librabries
import pandas as pd
import numpy as np
import json
import warnings
import logging
import yaml
import joblib
from typing import Dict, Any, List, Optional
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================
# DIRECTORIES AND FILE SETUP
# ==============================

base_dir = Path.cwd() # current working directory
list_of_directories = ['data','data/raw','data/processed','plots','data/splits','models','logs','config']
for directory in list_of_directories:
    Path(directory).mkdir(exist_ok=True)

data_dir = base_dir / 'data'
logs_dir = base_dir / 'logs'
plots_dir = base_dir / 'plots'
models_dir = base_dir / 'models'

def setup_logger(name: str, log_filename: str | Path, level = logging.INFO) -> logging.log:
    ''' Setup a dedicated timedrotatingfilehandler logging system that logs information to both file and console

    Args: 
        name : logger name (e.g. EDA, preprocessing, feature_engineering)
        log_filename: Log output file
        level: Logging level (e.g. INFO, WARNING, ERROR, DEBUG)

    Examples:
        log = setup_logger(name="EDA",log_filename="logs/EDA_pipeline.log", level=logging.INFO)
        log.info("Dedicated logging system setup successful")
    '''
    log = logging.getLogger(name)
    # prevent adding handlers multiple times if handlers already exist
    if log.handlers:
        return log
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s : %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
        )
    # Time rotating file handler
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',
        interval=1,
        backupCount=7
    )
    file_handler.suffix = "_%Y%m%d"
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    log.propagate = False # don't propagate to root logger
    log.setLevel(level)

    log.addHandler(file_handler)
    log.addHandler(console_handler)
    
    return log

# setup utils log
log = setup_logger(name='Utility',log_filename='logs/utility.log',level=logging.INFO)

# ==============================
# FILE I/0 HELPER FUNCTIONS
# ==============================

def load_csv_file(filename : str | Path) -> pd.DataFrame:
    '''Load csv file into python environment as a pandas dataframe
    
    Args:
        filename : Path to csv file
        
    Returns: 
        pd.dataframe : A pandas dataframe
        
    Raises:
        FileNotFoundError: if file is not found
        pd.errors.EmptyDataError: if dataframe is empty
        pd.errors.ParseError: if dataframe is malformed
    '''
    try:
        filepath = Path(filename)
        df = pd.read_csv(filepath)
        log.info(f'✅Data loaded from {filepath} | Shape {df.shape}')
        return df
    except FileNotFoundError:
        log.info('❌File Not Found! Check file path and try again!')
        raise
    except pd.errors.EmptyDataError as e:
        log.info(f'❌Data is empty : {e}')
        raise
    except pd.errors.ParserError as e: 
        log.info(f'❌CSV file is malformed : {e}')
        raise
    except Exception as e:
        log.info(f'❌Error parsing CSV file : {e}')

def load_json_file(filename : str | Path) -> Any:
    '''Loads json data from file
    
    Args:
        filename: Path to json file
        
    Raises:
        FileNotFoundError: If file does not exist
        JSONDecodeError: If file is malformed
    '''
    try:
        filepath = Path(filename)
        with open(filepath,'r') as file:
            data = json.load(file)
        log.info(f'✅Data loaded successfully from {filepath}')
        return data
    except FileNotFoundError:
        log.info(f'❌File not found! Check filepath and try again')
        raise
    except json.JSONDecodeError as e:
        log.info(f'❌Json file is malformed : {e}')
        raise
    except Exception as e:
        log.info(f'❌Error parsing json file : {e}')
        raise

def read_yaml_file(filename : str | Path) -> Any:
    '''Loads configuration info from a yaml file
    
    Args:
        filename : Path to yaml file
        
    Raises:
        FileNotFoundError : If file does not exist
        YAMLError: If yaml file is malformed
    '''
    try:
        filepath = Path(filename)
        with open(filepath,'r') as file:
            config = yaml.safe_load(file)
        log.info(f'✅Configuration Info. loaded from {filepath}')
        return config
    except FileNotFoundError:
        log.info(f'❌File not found! Check file path and try again')
        raise
    except yaml.YAMLError:
        log.info(f'❌YAML file is malformed : {e}')
        raise
    except Exception as e:
        log.info(f'❌Error parsing yaml file : {e}')
        raise

def load_joblib_file(filename : str | Path) -> Any:
    '''Loads binary data from joblib file
    
    Args:
        filename: Path to joblib file
        
    Raises:
        FileNotFoundError: If file does not exist
    '''
    try:
        filepath = Path(filename)
        with open(filepath,'r') as file:
            model = joblib.load(file)
        log.info(f'✅Model loaded from {filepath}')
        return model
    except FileNotFoundError:
        log.info(f'❌File not found! Check file path and try again')
        raise
    except Exception as e:
        log.info(f'❌Error parsing joblib file : {e}')
        raise

