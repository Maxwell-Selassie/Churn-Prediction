#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from autoviz.AutoViz_Class import AutoViz_Class
from pathlib import Path
from typing import List, Any, Dict, Optional
import logging
from dataclasses import dataclass
from autoviz import AutoViz_Class
import pkg_resources

from utils import (
    load_csv_file,save_json_file,ensure_directories,setup_logger, project_metadata,data_profile,get_timestamp,
    validate_df,save_csv_file
)

log = setup_logger('EDA_pipeline','logs/EDA_pipeline.log',level=logging.INFO)


# -------numeric columns-----------
def numeric_columns(df: pd.DataFrame) -> List[str]:
    '''Returns numeric columns, together with their minimum and maximum values
    
    Args: 
        df : DataFrame to be analyzed

    Returns:
        List of numerical values
    '''
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    log.info("="*50)
    log.info(f"NUMERICAL COLUMNS")
    log.info("="*50)

    cols = [col for col in numeric_cols[1:]]
    log.info(f'\nNumber of Numeric columns : {len(cols)} | Examples : {cols[:3]}\n')

    for i,col in enumerate(numeric_cols,1):
        log.info(f'\n{i}. {col} | Min: {df[col].min()} | Max: {df[col].max()}\n')
    return numeric_cols


# ------------categorical columns---------
def categorical_columns(df: pd.DataFrame) -> List[str]:
    '''Returns categorical columns and their respective unique values
    
    Args:
        df : DataFrame to be analyzed
        
    Returns:
        List of categorical columns
    '''
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    log.info("="*50)
    log.info(f"CATEGORICAL COLUMNS")
    log.info("="*50)

    cols = [col for col in categorical_cols]
    log.info(f'\nNumber of Categorical columns : {len(cols)} | Examples : {cols[:3]}\n')

    for i, col in enumerate(categorical_cols,1):
        uniques = df[col].unique()
        log.info(f'\n{i}. {col} - Unique: {df[col].nunique()} | Examples : {uniques[:3]}\n')
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
def missing_values(df : pd.DataFrame) -> pd.DataFrame:
    '''Analyze the missing values in the dataset
        Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing value statistics'''
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        log.info('No missing values detected in dataset')
        return pd.DataFrame(columns=['missing_values','missing_pct'])
    
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_values' : missing,
        'missing_pct' : missing_pct.round(2)
    })
    log.info(f'Dataset shape: {df.shape}, Missing columns: {missing_df.index.tolist()}')
    return missing_df

# ------------plot missing values -----------
def plt_missing_values(missing_summary : pd.DataFrame) -> None:
    """Visualize missing value distribution
    
    Args:
        missing_summary: DataFrame from missing_values() function
    """
    if missing_summary['missing_values'].sum() == 0:
        log.info(f'No missing values detected. Skipping plots')
        return None
    try:
        missing_summary['missing_values'].plot(kind='barh',figsize=(12,7),
                title='Distribution of missing values',
                xlabel='Frequency',color='indigo')
        
        output_path = f'plots/plots_missing_values.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f'Missing values plot successfully plotted and saved to {output_path}')
        plt.show()
        plt.close()
    except Exception as e:
        log.error(f'Error creating missing value plots : {e}')
        plt.close()

#------duplicated rows--------
def duplicate(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    '''Returns the duplicates found in the dataset
    
    Args:
        df : DataFrame to analyze
    
    Returns:
        A dataframe of duplicate rows or None if no duplicates'''
    if df.empty:
        log.warning(f'❌DataFrame is empty!')
        raise ValueError(f'❌DataFrame is empty!')
    

    duplicates  =  df[df.duplicated()]
    log.info(f'\nNumber of duplicates : {len(duplicates)}\n')

    if len(duplicates) == 0:
        log.info(f'No duplicates found\n')
    else:
        return duplicates
    
def plt_histogram(df: pd.DataFrame, numeric_col: List[str]) -> None:
    '''Plot distributions of numeric columns
    
    Args:
        df: Data to be plotted
        numeric_col: column to be plotted
    '''
    if df.empty:
        log.warning(f'DataFrame is empty. Cannot plot histogram distributions')
        raise

    for col in numeric_col:
        plt.figure(figsize=(20,15))    
        sns.histplot(data=df, x= col, kde=True, color='indigo', alpha=0.7)
        plt.title(f'Distribution of {col.title()}', fontsize=14, fontweight='bold')
        plt.ylabel(f'Frequency', fontsize=12)
        plt.xlabel(f'{col}', fontsize=12)
        plt.grid(True, alpha= 0.4)
        plt.tight_layout()

        output_dir = f'plots/{col}.png'
        plt.savefig(output_dir, dpi=300, bbox_inches='tight')
        plt.show()
        log.info(f'{col} histogram plots saved to {output_dir}')
        plt.close()

def plt_heatmap(df: pd.DataFrame) -> None:
        '''Plot heatmap
        
        Args:
            df: Data to be plotted
        '''
        if not df.empty:
            corr = df.corr(numeric_only=True, method='spearman')

            plt.figure(figsize=(20,17))
            sns.heatmap(data=corr, annot= True, fmt= '.2f', linecolor='green', cmap='Blues')
            plt.title('CORRELATION MATRIX')

            output_dir = 'plots/correlation_matrix.png'
            plt.savefig(output_dir, dpi=300, bbox_inches='tight')

            log.info(f'Correlation matrix plot successfully save to {output_dir}')
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            log.error(f'DataFrame is Empty! Cannot plot correlation matrix')
            raise

def plt_boxplots(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    '''Plot boxplots
    
    Args:
        df: Data to be plotted
        numeric_col: column to be plotted
    '''
    if not isinstance(df, pd.DataFrame):
        log.error('Data must be a pandas DataFrame')
        raise
    if not df.empty:
        for col in numeric_cols:
            plt.figure(figsize=(12,7))
            sns.boxplot(data=df, y= col, linecolor='blue', color='green')
            plt.title(f'Boxplot - {col}')
            plt.tight_layout()
            output_dir = f'plots/{col}_boxplot.png'
            plt.savefig(output_dir, dpi=300, bbox_inches='tight')
            log.info(f'{col} boxplot successfully saved to {output_dir}')
            plt.show()
            plt.close()
    else:
        log.error(f'Dataframe is empty! Cannot plot boxplots')
        raise


# ---------outlier detection using IQR--------
def check_outlier(df: pd.DataFrame, col: str) -> tuple:
    '''
        Detects outliers in numeric columns using IQR

    Args:
        df: DataFrame to be analyzed
        col: Column to be analzyed for outliers

    Returns:
        A tuple of outliers, lower range and upper range values
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

    log.info("="*50)
    log.info(f"OUTLIER SUMMARY")
    log.info("="*50)

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


def autoviz_report(
        df: pd.DataFrame = None,
        filename: str = None,
        target: str = None,
        output_dir: str = None,
        dep_var: str = None
):
    '''
        Generates a comprehensive Autoviz EDA report from either a dataframe or a file.
        Saves HTML plots in the specified output directory. 
    '''

    log.info(f'Running AutoViz eda....Output directory : {output_dir}')

    AV = AutoViz_Class()
    # ---- if a dataframe is passed---
    if df is not None:
        log.info(f'Using in-memory dataframe with {df.shape[0]} rows and {df.shape[1]} columns')
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
            filename='data/raw/e-commerce.csv',
            depVar=target,
            dfte=None,
            verbose=2,
            chart_format='html',
            save_plot_dir=output_dir
        )

    else:
        raise ValueError('You must either provide a dataframe or a filepath!')
    
    log.info(f'Autoviz EDA completed. Reports saved successfully')
    return dft

@dataclass
class EDAResults:
    '''
    Attributes: 
        data: Dataframe to be analyzed
        validate_data : Data quality checks
        data_profile : Quick summary of DataFrame
        numeric_columns : List of numeric columns
        categorical_columns : List of categorical columns
        missing_data : A Dictionary of columns with missing values
        duplicates: A dataframe containing duplicate rows
        outliers : Outlier summary
    '''
    data: str
    validate_data: Optional[Dict]
    data_profiles : Dict
    numerical_columns : List[str]
    category_columns : List[str]
    missing_data_ : Optional[pd.DataFrame]
    duplicates_ : Optional[pd.DataFrame]
    outliers_ : Optional[Dict]



def run_eda(filepath: str = 'data/raw/e-commerce.csv') -> EDAResults:
    '''Run EDA'''
    log.info("="*50)
    log.info("🎇STARTING EXPLORATORY ANALYSIS")
    log.info("="*50)

    # project metadata
    project_metadata(output_file='data/project_metadata.json')

    # data input and validation
    df = load_csv_file(filepath)
    required_cols = ['CustomerID',
    'Churn',
    'Tenure',
    'PreferredLoginDevice',
    'CityTier',
    'WarehouseToHome',
    'PreferredPaymentMode',
    'Gender',
    'HourSpendOnApp',
    'NumberOfDeviceRegistered',
    'PreferedOrderCat',
    'SatisfactionScore',
    'MaritalStatus',
    'NumberOfAddress',
    'Complain',
    'OrderAmountHikeFromlastYear',
    'CouponUsed',
    'OrderCount',
    'DaySinceLastOrder',
    'CashbackAmount'
    ]
    save_json_file(required_cols,'data/required_cols.json',indent=4)
    
    validate = validate_df(df, required_cols)
    profile = data_profile(df)

    # data exploration
    numeric_cols = numeric_columns(df)
    categorical_cols = categorical_columns(df)

    missing = missing_values(df)
    missing.to_csv('data/missing_summary.csv')

    # visualization
    plt_missing_values(missing)
    plt_histogram(df, numeric_cols)
    plt_heatmap(df)
    plt_boxplots(df, numeric_cols)

    duplicates = duplicate(df)
    save_json_file(duplicates, 'data/duplicated_data.json')

    outliers = outlier_summary(df, numeric_cols)
    save_csv_file(outliers, 'data/outliers.csv')

    # autoviz reports
    output_dir = ensure_directories('autoviz_reports')
    autoviz_report(df=df, target='Churn',output_dir=output_dir)

    results = EDAResults(
        data= df,
        validate_data=validate,
        data_profiles=profile,
        numerical_columns=numeric_cols,
        category_columns=categorical_cols,
        missing_data_=missing,
        duplicates_=duplicates,
        outliers_=outliers
    )

    log.info("="*50)
    log.info("💯EDA COMPLETED SUCCESSULLY!")
    log.info('='*50)

    return results

if __name__ == '__main__':
    run_eda()

