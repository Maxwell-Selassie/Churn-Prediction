'''
helper functions for this churn prediction end-to-end machine learning project
    * Basic helper functions
    - Setup folder and file paths
    - Setup logging

    * I/O helper functions
    - Load csv file
    - Load json file
    - Load yaml config file
    - Save csv file
    - Save yaml config file
    - Save json file
    - Load pickle file
    - Save pickle file
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
from typing import Dict, Any, List, Optional
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================
# DIRECTORIES AND FILE SETUP
# ==============================

base_dir = Path.cwd() # current working directory
list_of_directories = ['data','data/raw','data/processed','plots','data/splits','models','logs']
for directory in list_of_directories:
    Path(directory).mkdir(exist_ok=True)

data_dir = base_dir / 'data'
logs_dir = base_dir / 'logs'
plots_dir = base_dir / 'plots'
models_dir = base_dir / 'models'

# def setup_logger(name: str, log_filename: str | Path, level = logging.INFO) -> logging.log:
#     ''' Setup a dedicated timedrotatingfilehandler logging system that logs information to both file and console

#     Args: 
#         name : logger name (e.g. EDA, preprocessing, feature_engineering)
#         log_filename: Log output file
#         level: Logging level (e.g. INFO, WARNING, ERROR, DEBUG)

#     Examples:
#         log = setup_logger(name="EDA",log_filename="logs/EDA_pipeline.log", level=logging.INFO)
#         log.info("Dedicated logging system setup successful")
#     '''
#     log = logging.getLogger(name)
#     # prevent adding handlers multiple times if handlers already exist
#     if log.handlers:
#         return log
    
#     formatter = logging.Formatter(
#         "%(asctime)s - %(levelname)s : %(message)s",
#         datefmt='%H:%M:%S'
#         )
#     # Time rotating file handler
#     file_handler = TimedRotatingFileHandler(
#         filename=log_filename,
#         when='midnight',
#         interval=1,
#         backupCount=7
#     )
#     file_handler.suffix("_%Y%m%d")
#     file_handler.setFormatter(formatter)
    
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)

#     log.propagate(False) # don't propagate to root logger
#     log.setLevel(level)

#     log.addHandler(file_handler)
#     log.addHandler(console_handler)
    
#     return log

# # setup utils log
# log = setup_logger(name='Utility',log_filename='logs/utility.log',level=logging.INFO)