# load the datasets
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')

def load_data():
    '''Load the train,test and preprocessed datasets'''
    try:
        x_train = pd.read_parquet('data/x_train.parquet')
        x_test = pd.read_parquet('data/x_test.parquet')
        y_train = pd.read_parquet('data/y_train.parquet')['y_train']
        y_test = pd.read_parquet('data/y_test.parquet')['y_test']
        logging.info('Files successfully loaded!')
        return x_train,x_test,y_train,y_test
    except FileNotFoundError as e:
        raise FileNotFoundError('Files not found!') from e
