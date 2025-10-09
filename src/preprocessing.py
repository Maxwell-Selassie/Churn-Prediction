
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging
from sklearn.model_selection import train_test_split
import json

logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s : %(message)s', datefmt='%H:%M:%S')

def preprocessing():
    # load preprocessed csv file 
    try:
        df = pd.read_csv('../data/Preprocessed_e-commerce.csv')
        logging.info('File opened successfully!')
    except FileNotFoundError:
        logging.info('File was not found! Please check filepath and try again')
    
    y = df['Churn'] # traget output
    x = df.drop(columns=['Churn','CustomerID']).copy() #drop target column from dataset

    # splits the x and y datasets into training and testing sets
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.3,random_state=42,stratify=y
    )
    # save train and test splits 
    x_train.to_parquet('../data/x_train.parquet',index=False)
    x_test.to_parquet('../data/x_test.parquet',index=False)
    y_train.to_frame('y_train').to_parquet('../data/y_train.parquet',index=False)
    y_test.to_frame('y_test').to_parquet('../data/y_test.parquet',index=False)

    feature_names = x_train.columns.tolist()
    with open('../models/feature_names.json','w') as file:
        json.dump(feature_names,file, indent=4)        


    logging.info('Train/test splits and feature names successfully saved!')
