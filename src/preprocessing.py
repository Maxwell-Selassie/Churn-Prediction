# preprocessing_pipeline.py
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# utils assumed to provide read_yaml_file, save_json_file, save_joblib_file, load_joblib_file, load_csv_file, setup_logger, get_timestamp, ensure_directories
from utils import (
    setup_logger, get_timestamp, read_yaml_file,
    save_json_file, save_joblib_file, load_joblib_file, load_csv_file
)

log = setup_logger(name='preprocessing', log_filename='logs/data_preprocessing.log', level=logging.INFO)


class DataPreprocessorPipeline:
    """Config-driven data preprocessing pipeline."""

    def __init__(self, config_file: str | Path) -> None:
        log.info("=" * 50)
        log.info("INITIALIZING DATA PREPROCESSING PIPELINE")
        log.info("=" * 50)

        self.config_path = Path(config_file)
        self.config = read_yaml_file(self.config_path)
        self.scaler: StandardScaler | None = None

        # record of transformations (audit)
        self.transform_log: list[Dict[str, Any]] = []

        self.metadata: Dict[str, Any] = {
            "config_version": self.config.get("project", {}).get("version"),
            "config_filepath": str(self.config_path),
            "timestamp_start": get_timestamp(),
        }

        log.info(f"Project config loaded from {self.config_path}")
        log.info(f"Project name/version: {self.config.get('project', {}).get('name')} / {self.config.get('project', {}).get('version')}")

    def _log_transform(self, step: str, details: Dict[str, Any]) -> None:
        entry = {"step": step, "timestamp": get_timestamp(), **details}
        self.transform_log.append(entry)
        log.info(f"{step} - {details}")

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("=" * 50)
        log.info("DATA VALIDATION")
        log.info("=" * 50)

        if df is None or df.empty:
            log.error("DataFrame is empty or None")
            raise ValueError("Input dataframe is empty")

        self._log_transform("data_validation", {
            "initial_rows": int(df.shape[0]),
            "initial_features": int(df.shape[1]),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)
        })
        return df

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("=" * 50)
        log.info("DROP COLUMNS")
        log.info("=" * 50)

        cols_to_drop = []
        for item in self.config.get("columns_to_drop", []):
            col_name = item.get("column")
            reason = item.get("reason", "")
            if col_name in df.columns:
                cols_to_drop.append(col_name)
                log.info(f"Dropping column {col_name} - ({reason})")
            else:
                log.warning(f"Column {col_name} not found in DataFrame")

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self._log_transform("drop_columns", {
                "dropped_columns": cols_to_drop,
                "remaining_columns": int(df.shape[1])
            })

        return df

    def handling_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("=" * 50)
        log.info("HANDLING MISSING VALUES")
        log.info("=" * 50)

        missing_before = int(df.isnull().sum().sum())
        log.info(f"Total missing before: {missing_before}")

        missing_cfg = self.config.get("missing_values", {})
        if not missing_cfg:
            log.info("No missing_values configuration provided. Skipping missing handling.")
            return df

        for col, strategy in missing_cfg.items():
            if col not in df.columns:
                log.warning(f"Configured missing-col '{col}' not in dataframe. Skipping.")
                continue

            missing_count = int(df[col].isnull().sum())
            if missing_count == 0:
                log.info(f"No missing values for {col}")
                continue

            method = strategy.get("method", "mean")
            log.info(f"Filling missing for {col}: method={method} | count={missing_count}")

            if method == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "mode":
                modes = df[col].mode()
                if not modes.empty:
                    df[col].fillna(modes.iloc[0], inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
            elif method == "constant":
                df[col].fillna(strategy.get("fill_value", 0), inplace=True)
            else:
                log.warning(f"Unknown missing value method '{method}' for column {col}")

        missing_after = int(df.isnull().sum().sum())
        log.info(f"Total missing after: {missing_after}")

        self._log_transform("handle_missing_values", {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "missing_removed": missing_before - missing_after
        })

        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("=" * 50)
        log.info("HANDLING OUTLIERS")
        log.info("=" * 50)

        outlier_cfg = self.config.get("outliers", {})
        if not outlier_cfg:
            log.info("No outliers configuration. Skipping.")
            return df

        handled = {}
        for col, strategy in outlier_cfg.items():
            if col not in df.columns:
                log.warning(f"Outlier-configured column {col} not in df. Skipping.")
                continue

            action = strategy.get("action", "none")
            reason = strategy.get("reason", "")
            log.info(f"Outlier - {col}: action={action} reason={reason}")

            if action == "none":
                continue
            elif action == "cap":
                # expect config to have 'lower' and 'upper' or provide percentiles
                lower = strategy.get("lower", None)
                upper = strategy.get("upper", None)
                if lower is None or upper is None:
                    # fallback to 1st/99th percentiles if not explicitly provided
                    lower = df[col].quantile(strategy.get("lower_q", 0.01))
                    upper = df[col].quantile(strategy.get("upper_q", 0.99))
                df[col] = np.clip(df[col], a_min=lower, a_max=upper)
                handled[col] = {"action": "cap", "lower": float(lower), "upper": float(upper)}
            elif action == "remove":
                lower = strategy.get("lower", df[col].quantile(0.01))
                upper = strategy.get("upper", df[col].quantile(0.99))
                before = int(df.shape[0])
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                after = int(df.shape[0])
                handled[col] = {"action": "remove", "rows_before": before, "rows_after": after}
            else:
                log.warning(f"Unknown outlier action '{action}' for {col}")

        self._log_transform("handle_outliers", {"handled": handled})
        log.info("Outlier handling complete.")
        return df

    def _encode_features(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Encode categorical features"""
        log.info('-'*70)
        log.info('STEP 6: FEATURE ENCODING')
        log.info('-'*70)
        
        if 'encoding' not in self.config:
            log.info("No encoding configuration found")
            return df
        
        # One-hot encoding
        if 'one_hot' in self.config['encoding'] or 'onehot' in self.config['encoding']:
            onehot_config = self.config['encoding'].get('one_hot', self.config['encoding'].get('onehot', []))
            
            for item in onehot_config:
                col = item['column']
                if col not in df.columns:
                    log.warning(f"Column '{col}' not found, skipping")
                    continue
                
                drop_first = item.get('drop_first', True)
                log.info(f"One-hot encoding '{col}' (drop_first={drop_first})")
                
                # Perform one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                
                log.info(f"Created {len(dummies.columns)} dummy columns")
        
        
        self._log_transformation('encoding', {
            'onehot_columns': len(self.config['encoding'].get('one_hot', [])),
            'target_encoded': len(self.target_encoders)
        })
        
        return df

    def scaling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("=" * 50)
        log.info("SCALING NUMERIC FEATURES")
        log.info("=" * 50)

        scaling_cfg = self.config.get("scaling", {})
        if not scaling_cfg:
            log.info("No scaling configuration. Skipping.")
            return df

        method = scaling_cfg.get("method", "standard")
        columns_to_scale = scaling_cfg.get("columns_to_scale", [])
        exclude = scaling_cfg.get("exclude", [])

        cols_to_scale = [c for c in columns_to_scale if c in df.columns and c not in exclude]
        if not cols_to_scale:
            log.info("No columns to scale. Skipping.")
            return df

        if self.scaler is None:
            if method == "standard":
                self.scaler = StandardScaler()
            else:
                log.warning(f"Unknown scaling method '{method}', defaulting to StandardScaler")
                self.scaler = StandardScaler()
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            log.info("Fitted scaler and transformed columns.")
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            log.info("Applied existing scaler transform.")

        self._log_transform("scaling", {"method": method, "columns_scaled": cols_to_scale})
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("=" * 50)
        log.info("STARTING FIT_TRANSFORM (training data)")
        log.info("=" * 50)

        df = self.validate_data(df)
        df = self.drop_columns(df)
        df = self.handling_missing_values(df)
        df = self.handle_outliers(df)
        df = self.encoding_features(df)
        df = self.scaling_features(df)

        # save transformers (if any)
        self.save_transformers()

        self.metadata.update({
            "timestamp_end": get_timestamp(),
            "final_shape": df.shape,
            "transformations": self.transform_log
        })

        # save metadata path safely
        meta_path = self.config.get("file_paths", {}).get("preprocessing_metadata")
        if meta_path:
            save_json_file(self.metadata, meta_path)
            log.info(f"Saved preprocessing metadata to {meta_path}")
        else:
            log.warning("No metadata path configured; skipping metadata save.")

        log.info("FIT_TRANSFORM completed.")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("=" * 50)
        log.info("STARTING TRANSFORM (new/test data)")
        log.info("=" * 50)

        df = self.validate_data(df)
        df = self.drop_columns(df)
        df = self.handling_missing_values(df)
        df = self.handle_outliers(df)
        df = self.encoding_features(df)
        df = self.scaling_features(df)

        log.info("TRANSFORM completed.")
        return df

    def save_transformers(self) -> None:
        models_dir = Path(self.config.get("file_paths", {}).get("models_dir", "models"))
        models_dir.mkdir(parents=True, exist_ok=True)

        if self.scaler is not None:
            scaler_path = self.config.get("file_paths", {}).get("scaler_path")
            if scaler_path:
                save_joblib_file(self.scaler, scaler_path)
                log.info(f"Saved scaler to {scaler_path}")
            else:
                log.warning("Scaler exists but scaler_path not configured in file_paths.")

    def load_transformers(self) -> None:
        scaler_path = Path(self.config.get("file_paths", {}).get("scaler_path", ""))
        if scaler_path.exists():
            self.scaler = load_joblib_file(scaler_path)
            log.info(f"Loaded scaler from {scaler_path}")
        else:
            log.info("No scaler artifact found at configured path; continuing without it.")


def main():
    config = read_yaml_file("config/preprocessing_config.yaml")
    log.info("Loading raw data...")
    raw_path = config.get("file_paths", {}).get("raw_data")
    if not raw_path:
        raise ValueError("raw_data path not configured in preprocessing_config.yaml")

    df = load_csv_file(raw_path)

    if "Churn" not in df.columns:
        log.error("Expected target column 'Churn' not found in raw data.")
        raise ValueError("Missing 'Churn' target column")

    y = df["Churn"]
    X = df.drop(columns=["Churn"]).copy()

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    x_train.to_csv("data/splits/x_train.csv", index=False)
    x_test.to_csv("data/splits/x_test.csv", index=False)
    y_train.to_csv("data/splits/y_train.csv", index=False)
    y_test.to_csv("data/splits/y_test.csv", index=False)

    preprocessor = DataPreprocessorPipeline("config/preprocessing_config.yaml")
    X_processed = preprocessor.fit_transform(x_train)

    preproc_out = config.get("file_paths", {}).get("preprocessed_data")
    if preproc_out:
        Path(preproc_out).parent.mkdir(parents=True, exist_ok=True)
        X_processed.to_csv(preproc_out, index=False)
        log.info(f"Saved preprocessed data to {preproc_out}")
    else:
        log.warning("preprocessed_data path not configured; skipping save.")

    log.info("✅ Preprocessing completed successfully!")
    print("✅ Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
