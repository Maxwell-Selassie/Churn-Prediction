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
            
        }
