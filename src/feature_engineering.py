"""
Feature Engineering Pipeline for Customer Churn Prediction
Creates new features from preprocessed data to improve churn prediction

Author: Maxwell Selassie Hiamatsu
Date: October 24, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils import (
    setup_logger, load_csv_file, read_yaml_file,
    project_metadata, get_timestamp, save_json_file
)

# Setup logging
logger = setup_logger('FeatureEngineering', 'logs/feature_engineering.log')


class ChurnFeatureEngineer:
    """
    Production-ready feature engineering pipeline for customer churn prediction.
    
    This class handles:
    - Arithmetic features (ratios, logs, products)
    - Aggregate features (groupby statistics)
    - Interaction features (between columns)
    - Domain-specific features (churn business logic)
    - Polynomial features
    - Binning/discretization
    - Statistical features
    - Custom churn risk indicators
    
    All decisions are documented in config/feature_engineering_config.yaml
    """
    
    def __init__(self, config_path: str = 'config/feature_engineering_config.yaml'):
        """
        Initialize feature engineer with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        logger.info('='*70)
        logger.info('INITIALIZING CHURN FEATURE ENGINEERING PIPELINE')
        logger.info('='*70)
        
        self.config = read_yaml_file(config_path)
        self.config_path = config_path
        
        # Track created features
        self.created_features = []
        self.feature_log = []
        
        # Metadata
        self.metadata = {
            'config_version': self.config['project']['version'],
            'config_file': config_path,
            'timestamp_start': get_timestamp('%Y-%m-%d %H:%M:%S')
        }
        
        project_name = self.config['project']['name']
        version_name = self.config['project']['version']
        business_goal = self.config['project']['business_goal']

        logger.info(f"Project: {project_name}")
        logger.info(f"Version: {version_name}")
        logger.info(f"Business Goal: {business_goal}")
        logger.info(f"Configuration loaded successfully")
    
    def _log_feature_creation(self, feature_name: str, feature_type: str, description: str):
        """Log feature creation for audit trail"""
        entry = {
            'feature_name': feature_name,
            'feature_type': feature_type,
            'description': description,
            'timestamp': get_timestamp('%Y-%m-%d %H:%M:%S')
        }
        self.feature_log.append(entry)
        self.created_features.append(feature_name)
        logger.info(f"✓ Created: {feature_name} ({feature_type})")
    
    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """Check if required columns exist in dataframe"""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True
    
    def _safe_division(self, numerator: pd.Series, denominator: pd.Series, fill_value: float = 0) -> pd.Series:
        """Safely divide two series, handling division by zero"""
        # Replace zero denominators with NaN to avoid division errors
        result = numerator / denominator.replace(0, np.nan)
        # Fill NaN with specified value
        return result.fillna(fill_value)
    
    # ========================================================================
    # ARITHMETIC FEATURES
    # ========================================================================
    
    def create_arithmetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features using basic arithmetic operations
        Handles ratios, products, differences, logs, etc.
        """
        logger.info('-'*70)
        logger.info('STEP 1: CREATING ARITHMETIC FEATURES')
        logger.info('-'*70)
        
        if 'arithmetic' not in self.config or not self.config['arithmetic'].get('enabled', False):
            logger.info("Arithmetic features disabled in config")
            return df
        
        for feature in self.config['arithmetic'].get('features', []):
            name = feature['name']
            operation = feature['operation']
            operands = feature['operands']
            reason = feature.get('reason', 'No reason provided')
            
            # Validate columns exist
            if not self._validate_columns(df, operands):
                continue
            
            try:
                if operation == 'ratio':
                    # Safe division
                    df[name] = self._safe_division(df[operands[0]], df[operands[1]], fill_value=0)
                
                elif operation == 'product':
                    df[name] = df[operands[0]] * df[operands[1]]
                
                elif operation == 'difference':
                    df[name] = df[operands[0]] - df[operands[1]]
                
                elif operation == 'sum':
                    df[name] = df[operands[0]] + df[operands[1]]
                
                elif operation == 'power':
                    power = feature.get('power', 2)
                    df[name] = df[operands[0]] ** power
                
                elif operation == 'log':
                    # Use log1p to handle log(0) gracefully
                    df[name] = np.log1p(df[operands[0]])
                
                elif operation == 'sqrt':
                    df[name] = np.sqrt(df[operands[0]].clip(lower=0))
                
                else:
                    logger.warning(f"Unknown operation: {operation}")
                    continue
                
                self._log_feature_creation(name, 'arithmetic', reason)
            
            except Exception as e:
                logger.error(f"Error creating {name}: {e}")
        
        return df
    
    # ========================================================================
    # AGGREGATE FEATURES
    # ========================================================================
    
    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features using groupby aggregations
        Maps group-level statistics back to individual rows
        """
        logger.info('-'*70)
        logger.info('STEP 2: CREATING AGGREGATE FEATURES')
        logger.info('-'*70)
        
        if 'aggregates' not in self.config or not self.config['aggregates'].get('enabled', False):
            logger.info("Aggregate features disabled in config")
            return df
        
        for agg in self.config['aggregates'].get('features', []):
            group_by = agg['group_by']
            agg_column = agg['agg_column']
            agg_func = agg['agg_function']
            feature_name = agg['feature_name']
            reason = agg.get('reason', 'No reason provided')
            
            # Validate columns
            if not self._validate_columns(df, [group_by, agg_column]):
                continue
            
            try:
                # Perform aggregation
                if agg_func == 'mean':
                    grouped = df.groupby(group_by)[agg_column].mean()
                elif agg_func == 'sum':
                    grouped = df.groupby(group_by)[agg_column].sum()
                elif agg_func == 'count':
                    grouped = df.groupby(group_by)[agg_column].count()
                elif agg_func == 'std':
                    grouped = df.groupby(group_by)[agg_column].std()
                elif agg_func == 'min':
                    grouped = df.groupby(group_by)[agg_column].min()
                elif agg_func == 'max':
                    grouped = df.groupby(group_by)[agg_column].max()
                elif agg_func == 'median':
                    grouped = df.groupby(group_by)[agg_column].median()
                else:
                    logger.warning(f"Unknown aggregation function: {agg_func}")
                    continue
                
                # Map aggregated values back to original dataframe
                df[feature_name] = df[group_by].map(grouped)
                
                self._log_feature_creation(feature_name, 'aggregate', reason)
            
            except Exception as e:
                logger.error(f"Error creating {feature_name}: {e}")
        
        return df
    
    # ========================================================================
    # INTERACTION FEATURES
    # ========================================================================
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between columns
        Captures joint effects and non-linear relationships
        """
        logger.info('-'*70)
        logger.info('STEP 3: CREATING INTERACTION FEATURES')
        logger.info('-'*70)
        
        if 'interactions' not in self.config or not self.config['interactions'].get('enabled', False):
            logger.info("Interaction features disabled in config")
            return df
        
        for interaction in self.config['interactions'].get('features', []):
            feature_name = interaction['name']
            columns = interaction['columns']
            reason = interaction.get('reason', 'No reason provided')
            
            # Validate columns
            if not self._validate_columns(df, columns):
                continue
            
            try:
                # Multiply all specified columns together
                df[feature_name] = df[columns].prod(axis=1)
                
                self._log_feature_creation(feature_name, 'interaction', reason)
            
            except Exception as e:
                logger.error(f"Error creating {feature_name}: {e}")
        
        return df
    
    # ========================================================================
    # DOMAIN-SPECIFIC FEATURES (Churn Prediction Logic)
    # ========================================================================
    
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on e-commerce churn domain knowledge
        These are custom features specific to customer retention
        """
        logger.info('-'*70)
        logger.info('STEP 4: CREATING DOMAIN-SPECIFIC CHURN FEATURES')
        logger.info('-'*70)
        
        if 'domain_features' not in self.config or not self.config['domain_features'].get('enabled', False):
            logger.info("Domain features disabled in config")
            return df
        
        feature_list = self.config['domain_features'].get('features', [])
        logic = self.config.get('domain_feature_logic', {})
        
        # Customer Lifecycle Stage
        if 'customer_lifecycle_stage' in feature_list and 'customer_lifecycle_stage' in logic:
            logger.info("Creating customer_lifecycle_stage...")
            
            def assign_lifecycle_stage(row):
                tenure = row['Tenure']
                order_count = row['OrderCount']
                days_since = row['DaySinceLastOrder']
                
                if tenure < 3:
                    return 'New'
                elif tenure >= 3 and tenure < 12 and order_count < 10:
                    return 'Growing'
                elif tenure >= 12 and order_count >= 10 and days_since < 30:
                    return 'Mature'
                elif days_since >= 30 and days_since < 60:
                    return 'At-Risk'
                elif days_since >= 60:
                    return 'Dormant'
                else:
                    return 'Growing'
            
            if self._validate_columns(df, ['Tenure', 'OrderCount', 'DaySinceLastOrder']):
                df['customer_lifecycle_stage'] = df.apply(assign_lifecycle_stage, axis=1)
                self._log_feature_creation('customer_lifecycle_stage', 'domain', 
                                        'Customer lifecycle segmentation')
        
        # Value Tier
        if 'value_tier' in feature_list and 'value_tier' in logic:
            logger.info("Creating value_tier...")
            
            def assign_value_tier(row):
                order_count = row['OrderCount']
                cashback = row['CashbackAmount']
                
                if order_count >= 20 and cashback >= 200:
                    return 'High'
                elif order_count >= 10 and cashback >= 100:
                    return 'Medium'
                else:
                    return 'Low'
            
            if self._validate_columns(df, ['OrderCount', 'CashbackAmount']):
                df['value_tier'] = df.apply(assign_value_tier, axis=1)
                self._log_feature_creation('value_tier', 'domain', 
                                        'Customer value segmentation')
        
        # Engagement Score
        if 'engagement_score' in feature_list and 'engagement_score' in logic:
            logger.info("Creating engagement_score...")
            
            required_cols = ['OrderCount', 'HourSpendOnApp', 'SatisfactionScore', 'NumberOfDeviceRegistered']
            if self._validate_columns(df, required_cols):
                df['engagement_score'] = (
                    (df['OrderCount'] * 2) + 
                    (df['HourSpendOnApp'] * 1.5) + 
                    (df['SatisfactionScore'] * 10) + 
                    (df['NumberOfDeviceRegistered'] * 5)
                )
                # Normalize to 0-100
                df['engagement_score'] = (df['engagement_score'] - df['engagement_score'].min()) / \
                                        (df['engagement_score'].max() - df['engagement_score'].min()) * 100
                
                self._log_feature_creation('engagement_score', 'domain', 
                                        'Composite engagement metric (0-100)')
        
        # Churn Risk Score
        if 'churn_risk_score' in feature_list and 'churn_risk_score' in logic:
            logger.info("Creating churn_risk_score...")
            
            required_cols = ['DaySinceLastOrder', 'Complain', 'SatisfactionScore', 'WarehouseToHome']
            if self._validate_columns(df, required_cols):
                df['churn_risk_score'] = (
                    (df['DaySinceLastOrder'] * 2) + 
                    (df['Complain'] * 15) + 
                    ((5 - df['SatisfactionScore']) * 10) + 
                    (df['WarehouseToHome'] * 0.5)
                )
                # Normalize to 0-100
                df['churn_risk_score'] = (df['churn_risk_score'] - df['churn_risk_score'].min()) / \
                                        (df['churn_risk_score'].max() - df['churn_risk_score'].min()) * 100
                
                self._log_feature_creation('churn_risk_score', 'domain', 
                                    'Composite churn risk metric (0-100)')
        
        # Loyalty Score
        if 'loyalty_score' in feature_list and 'loyalty_score' in logic:
            logger.info("Creating loyalty_score...")
            
            required_cols = ['Tenure', 'OrderCount', 'SatisfactionScore', 'Complain']
            if self._validate_columns(df, required_cols):
                df['loyalty_score'] = (
                    (df['Tenure'] * 2) + 
                    (df['OrderCount'] * 3) + 
                    (df['SatisfactionScore'] * 10) - 
                    (df['Complain'] * 5)
                )
                # Normalize to 0-100
                df['loyalty_score'] = (df['loyalty_score'] - df['loyalty_score'].min()) / \
                                     (df['loyalty_score'].max() - df['loyalty_score'].min()) * 100
                
                self._log_feature_creation('loyalty_score', 'domain', 
                                        'Composite loyalty metric (0-100)')
        
        # Binary Flags
        if 'price_sensitivity_flag' in feature_list:
            logger.info("Creating price_sensitivity_flag...")
            if self._validate_columns(df, ['CouponUsed', 'OrderCount']):
                df['price_sensitivity_flag'] = (
                    (df['CouponUsed'] / df['OrderCount'].replace(0, 1)) > 0.7
                ).astype(int)
                self._log_feature_creation('price_sensitivity_flag', 'domain', 
                                        'Heavy coupon user indicator')
        
        if 'multi_device_user_flag' in feature_list:
            logger.info("Creating multi_device_user_flag...")
            if 'NumberOfDeviceRegistered' in df.columns:
                df['multi_device_user_flag'] = (df['NumberOfDeviceRegistered'] > 1).astype(int)
                self._log_feature_creation('multi_device_user_flag', 'domain', 
                                        'Multi-device user indicator')
        
        if 'complaint_unresolved_flag' in feature_list:
            logger.info("Creating complaint_unresolved_flag...")
            if self._validate_columns(df, ['Complain', 'SatisfactionScore']):
                df['complaint_unresolved_flag'] = (
                    (df['Complain'] > 0) & (df['SatisfactionScore'] < 3)
                ).astype(int)
                self._log_feature_creation('complaint_unresolved_flag', 'domain', 
                                        'Unresolved complaint indicator')
        
        if 'dormant_customer_flag' in feature_list:
            logger.info("Creating dormant_customer_flag...")
            if 'DaySinceLastOrder' in df.columns:
                df['dormant_customer_flag'] = (df['DaySinceLastOrder'] > 30).astype(int)
                self._log_feature_creation('dormant_customer_flag', 'domain', 
                                        'Dormant customer indicator (30+ days)')
        
        if 'at_risk_flag' in feature_list:
            logger.info("Creating at_risk_flag...")
            if self._validate_columns(df, ['DaySinceLastOrder', 'SatisfactionScore', 'Complain']):
                df['at_risk_flag'] = (
                    (df['DaySinceLastOrder'] > 30) & 
                    (df['SatisfactionScore'] < 3) & 
                    (df['Complain'] > 0)
                ).astype(int)
                self._log_feature_creation('at_risk_flag', 'domain', 
                                        'High churn risk indicator (multiple signals)')
        
        return df
    
    # ========================================================================
    # BINNING FEATURES
    # ========================================================================
    
    def create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned/discretized versions of continuous features
        Useful for capturing non-linear patterns and interpretability
        """
        logger.info('-'*70)
        logger.info('STEP 5: CREATING BINNING FEATURES')
        logger.info('-'*70)
        
        if 'binning' not in self.config or not self.config['binning'].get('enabled', False):
            logger.info("Binning features disabled in config")
            return df
        
        for bin_feature in self.config['binning'].get('features', []):
            source_col = bin_feature['column']
            feature_name = bin_feature['name']
            bins = bin_feature['bins']
            labels = bin_feature.get('labels', None)
            reason = bin_feature.get('reason', 'No reason provided')
            
            # Validate column exists
            if source_col not in df.columns:
                logger.warning(f"Column {source_col} not found")
                continue
            
            try:
                df[feature_name] = pd.cut(
                    df[source_col],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )
                
                self._log_feature_creation(feature_name, 'binning', reason)
            
            except Exception as e:
                logger.error(f"Error creating {feature_name}: {e}")
        
        return df
    
    # ========================================================================
    # POLYNOMIAL FEATURES
    # ========================================================================
    
    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial features (squared, cubed, etc.)
        Useful for capturing non-linear relationships
        """
        logger.info('-'*70)
        logger.info('STEP 6: CREATING POLYNOMIAL FEATURES')
        logger.info('-'*70)
        
        if 'polynomial' not in self.config or not self.config['polynomial'].get('enabled', False):
            logger.info("Polynomial features disabled in config")
            return df
        
        for poly in self.config['polynomial'].get('features', []):
            source_col = poly['column']
            degrees = poly['degrees']
            reason = poly.get('reason', 'No reason provided')
            
            # Validate column exists
            if source_col not in df.columns:
                logger.warning(f"Column {source_col} not found")
                continue
            
            try:
                for degree in degrees:
                    feature_name = f"{source_col}_power_{degree}"
                    df[feature_name] = df[source_col] ** degree
                    
                    self._log_feature_creation(feature_name, 'polynomial', 
                                            f"{source_col} to power {degree} - {reason}")
            
            except Exception as e:
                logger.error(f"Error creating polynomial features for {source_col}: {e}")
        
        return df
    
    # ========================================================================
    # STATISTICAL FEATURES
    # ========================================================================
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features (percentile ranks, z-scores)
        """
        logger.info('-'*70)
        logger.info('STEP 7: CREATING STATISTICAL FEATURES')
        logger.info('-'*70)
        
        if 'statistical' not in self.config or not self.config['statistical'].get('enabled', False):
            logger.info("Statistical features disabled in config")
            return df
        
        feature_types = self.config['statistical'].get('features', [])
        
        # Percentile rank features
        if 'percentile_ranks' in feature_types:
            percentile_cols = self.config['statistical'].get('percentile_columns', [])
            for col in percentile_cols:
                if col in df.columns:
                    feature_name = f"{col}_percentile_rank"
                    df[feature_name] = df[col].rank(pct=True)
                    
                    self._log_feature_creation(feature_name, 'statistical', 
                                            f"Percentile rank of {col}")
        
        # Z-score features
        if 'z_scores' in feature_types:
            zscore_cols = self.config['statistical'].get('zscore_columns', [])
            for col in zscore_cols:
                if col in df.columns:
                    feature_name = f"{col}_zscore"
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[feature_name] = (df[col] - mean) / std
                    else:
                        df[feature_name] = 0
                    
                    self._log_feature_creation(feature_name, 'statistical', 
                                            f"Z-score of {col}")
        
        return df
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info('\n' + '='*70)
        logger.info('STARTING CHURN FEATURE ENGINEERING PIPELINE')
        logger.info('='*70)
        
        initial_shape = df.shape
        logger.info(f"Initial shape: {initial_shape}")
        
        # Apply all feature engineering steps
        df = self.create_arithmetic_features(df)
        df = self.create_aggregate_features(df)
        df = self.create_interaction_features(df)
        df = self.create_domain_features(df)
        df = self.create_binning_features(df)
        df = self.create_polynomial_features(df)
        df = self.create_statistical_features(df)
        
        # Handle any NaN created during feature engineering
        if self.config.get('handle_nan_after_engineering', True):
            nan_count_before = df.isnull().sum().sum()
            if nan_count_before > 0:
                logger.info(f"Handling {nan_count_before} NaN values created during feature engineering")
                # Fill numeric NaN with 0
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(0)
                logger.info("✓ NaN values filled with 0")
        
        final_shape = df.shape
        new_features = final_shape[1] - initial_shape[1]
        
        # Save metadata
        self.metadata['timestamp_end'] = get_timestamp('%Y-%m-%d %H:%M:%S')
        self.metadata['initial_shape'] = initial_shape
        self.metadata['final_shape'] = final_shape
        self.metadata['new_features_count'] = new_features
        self.metadata['created_features'] = self.created_features
        self.metadata['feature_log'] = self.feature_log
        
        save_json_file(self.metadata, self.config['paths']['feature_engineering_metadata'])
        
        logger.info('='*70)
        logger.info('✓ CHURN FEATURE ENGINEERING COMPLETED')
        logger.info(f"Initial features: {initial_shape[1]}")
        logger.info(f"New features created: {new_features}")
        logger.info(f"Final features: {final_shape[1]}")
        logger.info('='*70)
        
        return df
    


def main():
    """Main feature engineering execution"""
    # Load configuration
    config = read_yaml_file('config/feature_engineering.yaml')
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    df = load_csv_file(config['paths']['preprocessed_data'])
    
    # Initialize feature engineer
    engineer = ChurnFeatureEngineer('config/feature_engineering.yaml')
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    # Save engineered data
    df_engineered.to_csv(config['paths']['engineered_data'], index=False)
    
    logger.info(f"\n✅ Feature engineering completed successfully!")
    logger.info(f"📁 Engineered data saved to: {config['paths']['engineered_data']}")
    logger.info(f"📊 Final shape: {df_engineered.shape}")
    logger.info(f"🆕 New features: {len(engineer.created_features)}")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ CHURN FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"📁 Output: {config['paths']['engineered_data']}")
    print(f"📊 Initial shape: {engineer.metadata['initial_shape']}")
    print(f"📊 Final shape: {df_engineered.shape}")
    print(f"🆕 New features created: {len(engineer.created_features)}")
    print("\n📋 Created Features by Type:")
    
    # Group features by type
    feature_types = {}
    for log in engineer.feature_log:
        ftype = log['feature_type']
        if ftype not in feature_types:
            feature_types[ftype] = []
        feature_types[ftype].append(log['feature_name'])
    
    for ftype, features in feature_types.items():
        print(f"\n  {ftype.upper()} ({len(features)} features):")
        for i, feature in enumerate(features[:5], 1):  # Show first 5
            print(f"    {i}. {feature}")
        if len(features) > 5:
            print(f"    ... and {len(features) - 5} more")
    

if __name__ == '__main__':
    main()