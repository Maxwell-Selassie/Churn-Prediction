#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from autoviz.AutoViz_Class import AutoViz_Class
import pkg_resources
import os



import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')



def load_data(filepath: str):
    'Returns the dataframe'
    try:
        df = pd.read_csv(filepath) # load csv into the env as a pandas dataFrame
        #prints out the number of rows and columns (5630 rows, 20 columns)
        logging.info(f'Data successfully loaded with {df.shape[0]} rows and {df.shape[1]} columns\n')
        return df
    except FileNotFoundError:
        logging.info('File Not Found! Please check filepath and try again')
        raise



# ----dataset overview--------
def dataset_overview(df: pd.DataFrame):
    '''Returns the shape of the dataset (i.e. number of rows and columns), 
    alongside a short descriptive summary statistcs of the dataset'''
    logging.info(f'Number of observations : {df.shape[0]}')
    logging.info(f'Number of features : {df.shape[1]}')
    return df.describe(include='all').T


# -------numeric columns-----------
def numeric_columns(df: pd.DataFrame):
    '''Returns numeric columns, together with their minimum and maximum values'''
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    cols = [col for col in numeric_cols[1:]]
    logging.info(f'\nNumber of Numeric columns : {len(cols)} | Examples : {cols[:3]}\n')

    for i,col in enumerate(numeric_cols,1):
        logging.info(f'\n{i}. {col} - Min: {df[col].min()} - Max: {df[col].max()}\n')
    return numeric_cols


# ------------categorical columns---------
def categorical_columns(df: pd.DataFrame):
    '''Returns categorical columns and their respective unique values'''
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    cols = [col for col in categorical_cols]
    logging.info(f'\nNumber of Categorical columns : {len(cols)} | Examples : {cols[:3]}\n')

    for i, col in enumerate(categorical_cols,1):
        uniques = df[col].unique()
        logging.info(f'\n{i}. {col} - Unique: {df[col].nunique()} | Examples : {uniques[:3]}\n')
    return categorical_cols


# Numerical Columns Description
# Churn : Target variable (0 = customer stayed, 1 = customer churned/left)
# Tenure : How long the customer has stayed with the company
# CityTier : Classification of the city where the customer lives
# WarehouseToHome : Distance between the warehouse and the customer's home
# HoursSpendOnApp : Average hours the customer spends on the app per day/week
# NumberOfDeviceRegistered : Number of devices registered to a cutomer's account
# SatisfactionScore: A customer's satisfaction rating 
# NumberOfAddress : How many addresses the customer has saved
# Complain : Whether the customer has filed a complaint or not
# OrderAmountHikeFromlastYear : Percentage increase in order compared to last year
# CouponUsed : Number of coupons used by the customer
# OrderCount : Number of orders placed by the customer
# DaysSinceLastOrder : Number of days since the customer's last order
# CashbackAmount : Total cashback the customer has received
# Categorical columns description
# PreferredLoginDevice - The device most often used to log into the app/site
# PreferredPaymentMode - Payment method most often used
# Gender - Sex of the customer (male/female)
# PreferredOrderCat - Most frequent product category ordered
# MaritalStatus - Marital status of the customer


#missing data
def missing_data(df: pd.DataFrame):
    '''Returns the sum of missing data alongside the percentage of 
    missing values with proportion to the length of the dataframe
    '''
    missing = df.isnull().sum()
    missing = missing[missing>0].sort_values(ascending=False)
    missing_pct = missing / len(df) * 100
    logging.info(f'\nMissing Data \n')
    missing_df = pd.DataFrame({
        'missing value' : missing,
        'missing pct' : missing_pct.round(2)
    })
    return missing_df

#------duplicated rows--------
def duplicate(df: pd.DataFrame):
    '''Returns the duplicates found in the dataset'''
    duplicates  =  df[df.duplicated()]
    logging.info(f'\nNumber of duplicates : {len(duplicates)}\n')
    if len(duplicates) == 0:
        logging.info(f'No duplicates found\n')
    else:
        return duplicates
    



# ---------outlier detection using IQR--------
def check_outlier(df: pd.DataFrame, col: str):
    '''
        Detects outliers in numeric columns using IQR
    '''
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers, lower_bound, upper_bound



def outlier_summary(df: pd.DataFrame, numeric_cols: list[str]):
    result = []
    logging.info('Outlier Summary\n')
    for i,col in enumerate(numeric_cols,1):
        outlier, lower, upper = check_outlier(df, col)
        result .append({
            'index': i,
            'columns' : col,
            'outlier' : len(outlier),
            'Lower Range' : lower,
            'Upper Range' : upper
        })
    summary_df = pd.DataFrame(result)
    return summary_df




def one_hot_encode(df: pd.DataFrame, categorical_columns):
    '''One hot encode all categorical columns'''
    return pd.get_dummies(data=df, columns=categorical_columns,dtype=float)



import os
def save_summary(df: pd.DataFrame, name: str):
    os.makedirs('eda_reports',exist_ok=True)
    path = f'eda_reports/{name}.csv'
    df.to_csv(path, index=False)
    logging.info(f'Saved report: {path}')

def autoviz_report(
        df: pd.DataFrame = None,
        filename: str = None,
        target: str = None,
        output_dir: str = 'autoviz_reports',
        dep_var: str = None
):
    '''
        Generates a comprehensive Autoviz EDA report from either a dataframe or a file.
        Saves HTML plots in the specified output directory. 
    '''
    os.makedirs(output_dir,exist_ok=True)
    logging.info(f'Running AutoViz eda....Output directory : {output_dir}')

    AV = AutoViz_Class()
    # ---- if a dataframe is passed---
    if df is not None:
        logging.info(f'Using in-memory dataframe with {df.shape[0]} rows and {df.shape[1]} columns')
        dft = AV.AutoViz(
            filename='',
            depVar=target,
            dfte=df,
            verbose=2,
            chart_format='html',
            save_plot_dir=output_dir
        )

    # if a filename is passed instead of a dataframe
    elif filename:
        dft = AV.AutoViz(
            filename='../data/e-commerce.csv',
            depVar=target,
            dfte=None,
            verbose=2,
            chart_format='html',
            save_plot_dir=output_dir
        )

    else:
        raise ValueError('You must either provide a dataframe or a filepath!')
    
    logging.info(f'Autoviz EDA completed. Reports saved successfully')
    return dft


def run_eda(filepath: str = '../data/e-commerce.csv'):
    df = load_data(filepath)
    overview = dataset_overview(df)
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)

    if 'CouponUsed' in df.columns:
        df['CouponUsed'].fillna(0,inplace=True)

    if 'HourSpendOnApp' in df.columns:
        df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mean(), inplace=True)
    
    if 'WarehouseToHome' in df.columns:
        df = df.query('WarehouseToHome <= 36')

    if 'Tenure' in df.columns:
        df['Tenure'].fillna(df['Tenure'].median(), inplace=True)

    if 'DaySinceLastOrder' in df.columns:
        df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median(), inplace=True)
    
    if 'OrderAmountHikeFromlastYear' in df.columns:
        df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].mean(), inplace=True)

    if 'OrderCount' in df.columns:
        df['OrderCount'].fillna(df['OrderCount'].mean(), inplace=True)

    missing = missing_data(df)
    duplicates = duplicate(df)
    outliers = outlier_summary(df, num_cols)
    df = one_hot_encode(df, cat_cols)
    dft = autoviz_report()
    logging.info(f'EDA completed successfully!')

    save_summary(overview,'overview')
    save_summary(missing, 'missing_data')
    save_summary(outliers, 'outlier_summary')
    if duplicates is not None:
        save_summary(duplicates,'duplicates')
    
    print({
        'data' : df,
        'overview' : overview,
        'num_cols' :num_cols,
        'cat_cols' : cat_cols,
        'missing' : missing,
        'duplicates' : duplicates,
        'outliers' : outliers,
    })
    print(dft)
