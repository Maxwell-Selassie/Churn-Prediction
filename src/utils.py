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
    - Save json file
    - Save yaml config file
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
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================
# DIRECTORIES AND FILE SETUP
# ==============================

base_dir = Path.cwd() # current working directory
list_of_directories = ['data','data/raw','data/processed','plots','data/splits','models','logs','config','tests']
for directory in list_of_directories:
    Path(directory).mkdir(exist_ok=True)

data_dir = base_dir / 'data'
logs_dir = base_dir / 'logs'
plots_dir = base_dir / 'plots'
models_dir = base_dir / 'models'

def setup_logger(name: str, log_filename: str | Path, level = logging.INFO) -> logging.Logger:
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
        log.error('❌File Not Found! Check file path and try again!')
        raise
    except pd.errors.EmptyDataError as e:
        log.error(f'❌Data is empty : {e}')
        raise
    except pd.errors.ParserError as e: 
        log.error(f'❌CSV file is malformed : {e}')
        raise
    except Exception as e:
        log.error(f'❌Error parsing CSV file : {e}')

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
        log.error(f'❌File not found! Check filepath and try again')
        raise
    except json.JSONDecodeError as e:
        log.error(f'❌Json file is malformed : {e}')
        raise
    except Exception as e:
        log.error(f'❌Error parsing json file : {e}')
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
        log.error(f'❌File not found! Check file path and try again')
        raise
    except yaml.YAMLError:
        log.error(f'❌YAML file is malformed : {e}')
        raise
    except Exception as e:
        log.error(f'❌Error parsing yaml file : {e}')
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
        model = joblib.load(filepath)
        log.info(f'✅Model loaded from {filepath}')
        return model
    except FileNotFoundError:
        log.error(f'❌File not found! Check file path and try again')
        raise
    except Exception as e:
        log.error(f'❌Error parsing joblib file : {e}')
        raise

def save_csv_file(data: pd.DataFrame, filename: str | Path,index: bool = False) -> None:
    '''Save dataframe to a csv file
    
    Args:
        data: Dataframe to be saved
        filename: Ouput file path
        
    Example:
        save_csv_file(data=df, filename="data/cleaned_data.csv")    
    '''
    try:
        filepath = Path(filename)
        data.to_csv(filepath, index=index)
        log.info(f'✅Data successfully saved to {filepath}')
    except Exception as e:
        log.error(f'❌Error saving CSV data to {filepath} : {e}')
        raise

def save_json_file(data : Any, filename: str | Path, indent: int = 4) -> None:
    '''Save data to a json file
    
    Args:
        data : Data to be saved (Data must be JSON serializable)
        filename : Output file path

    Example: 
        save_json_file(data=data, filename="data/outlier_summary.json",indent=4)
    '''
    try:
        filepath = Path(filename)
        with open(filepath,'w') as file:
            json.dump(data, file, indent=indent)
        log.info(f'✅Data successfully saved to {filepath}')
    except Exception as e:
        log.error(f'❌Error saving JSON data to {filepath} : {e}')
        raise

def save_yaml_file(config : Dict[str, Any], filename: str | Path, sort_keys: bool = False) -> None:
    '''Save data to a yaml config yaml
    
    Args:
        config: configuration information to be saved
        filename: Output file path
    '''
    try:
        filepath = Path(filename)
        with open(filepath,'w') as file:
            yaml.dump(config,file, sort_keys=sort_keys)
        log.info(f'✅YAML configuration saved to {filepath}')
    except Exception as e:
        log.error(f'❌Error saving YAML config data to {filepath} : {e}')
        raise

def save_joblib_file(data: Any, filename: str | Path) -> None:
    '''Saves binary data to a joblib model
    
    Args:
        data : Binary data to be saved
        filename: Output file path
    '''
    try:
        filepath = Path(filename)
        with open(filepath,'w') as file:
            joblib.dump(data, file)
        log.info(f'✅Data successfullt saved to {filepath}')
    except Exception as e:
        log.error(f'❌Error saving binary data to {filepath} : {e}')
        raise

def ensure_directories(directory: str) -> Path:
    '''Ensure all directories exist, if not, then create the directory
    
    Args:
        directory: Name of folder

    Returns:
        Path : created directory
    '''
    path = Path(directory)
    path.mkdir(parents=True,exist_ok=True)
    log.info(f'✅Directory ensured : {path}')
    return path

def get_timestamp(format: str = '%Y-%m-%d %H:%M:%S') -> datetime:
    '''Get timestamp for file naming
    
    Args:
        format: Date formatting style
        
    Returns:
        datatime: Date, formatted correctly
    '''
    return datetime.now().strftime(format)


def project_metadata(output_file : str | Path) -> Dict[str,Any]:
    ''' Get and save project metadata

    Args:
        metadata : Project metadata 
        filename : Ouptut file path

    Returns:
        Dictionary: A dictionary containing project metadata
    '''
    metadata = {
        'Timestamp' : get_timestamp(),
        'Project Name' : "👨👩Customer Churn Prediction",
        'Author' : "💎Maxwell Selasie Hiamatsu",
        'Python Version' : __import__('sys').version,
        'Pandas version' : pd.__version__,
        'Numpy version' : pd.__version__
    }
    save_json_file(data=metadata, filename=output_file)
    log.info(f"✅Project metadata saved to {output_file}")
    return metadata

def validate_df(df: pd.DataFrame, required_cols: Optional[List[str]]) -> None:
    '''Validate dataframe and required columns
    
    Args:
        df : Dataframe to validate
        required_cols: columns to validate

    Raises:
        ValueError : If dataframe is empty
    '''
    if df.empty:
        log.error(f'❌The dataframe is empty')
        raise ValueError(f'✖️The dataframe is empty!')

    if required_cols is None:
        log.info('No required columns specified')
        return

    if required_cols:
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            log.error(f'❌The dataframe is missing required columns : {missing_cols}')
            raise ValueError(f'✖️Missing required columns : {missing_cols}')

        else:
            log.info(f'✅Dataframe has no required missing columns')

def get_memory_usage(df: pd.DataFrame) -> Dict[str,float | int]:
    '''Get the memory usage statistics of the dataframe
    
    Args:
        df : Dataframe to analyze
        
    Returns:
        Dictionary with memory usage details
    '''
    memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    per_row_usage_in_kb = (df.memory_usage(deep=True).sum() / len(df)) / 1024

    return {
        'm_usage_in_mb' : memory_usage_mb.round(2),
        'per_row_m_usage_in_kb': per_row_usage_in_kb.round(2),
        'rows' : len(df),
        'columns' : len(df.columns)
    }

def data_profile(df: pd.DataFrame) -> Dict[str,Any]:
    '''Generate a quick profile of the dataframe
    
    Args:
        df : Dataframe to profile
        
    Returns:
        Dictionary with data profile'''
    profile = {
        'Observations' : df.shape[0],
        'Features' : df.shape[1],
        'Numeric_columns' : df.select_dtypes(include=[np.number]).columns.tolist(),
        'Categorical_columns' : df.select_dtypes(exclude=[np.number]).columns.tolist(),
        'Missing_values' : df.isnull().sum().to_dict(),
        'Duplicates' : df.duplicated().sum(),
        'Data_types' : df.dtypes.astype(str).to_dict()
    }
    log.info(f'✅Data Profile generated for Dataframe with {df.shape}')
    return profile

if __name__ == '__main__':
    log.info(f"Utility module loaded successfully")