# optimized_feature_engineering.py - UPDATED FOR COMPATIBILITY
import pandas as pd
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestRegressor

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def optimized_feature_engineering(df):
    """
    Enhanced feature engineering with focus on performance and quality features
    """
    logging.info("Starting optimized feature engineering")

    try:
        # Create a copy of the dataframe
        features_df = df.copy()

        # 1. Derived metrics (keep the most useful ones)
        if 'total_bytes_processed' in features_df.columns and 'elapsed_time_ms' in features_df.columns:
            # Processing throughput (bytes/ms)
            features_df['bytes_per_ms'] = features_df['total_bytes_processed'] / np.maximum(
                features_df['elapsed_time_ms'], 1)

            # Computational intensity (slot_ms/byte)
            features_df['compute_intensity'] = features_df['total_slot_ms'] / np.maximum(
                features_df['total_bytes_processed'], 1)

            # Parallelism ratio
            features_df['parallelism'] = features_df['total_slot_ms'] / np.maximum(features_df['elapsed_time_ms'], 1)

        # 2. Time-based features
        # Create cyclical time features for better modeling of time patterns
        if 'hour_of_day' in features_df.columns:
            # Convert hour to cyclical features to preserve the circular nature
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24.0)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24.0)

        if 'day_of_week' in features_df.columns:
            # Convert day of week to cyclical features
            features_df['day_sin'] = np.sin(2 * np.pi * (features_df['day_of_week'] - 1) / 7.0)
            features_df['day_cos'] = np.cos(2 * np.pi * (features_df['day_of_week'] - 1) / 7.0)

            # Add weekend indicator
            features_df['is_weekend'] = (features_df['day_of_week'] >= 6).astype(int)

        # 3. Handle statement_type with controlled encoding
        if 'statement_type' in features_df.columns:
            # Map to standard types - focusing on the most common ones
            standard_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'MERGE']

            # Create dummy variables only for the common types
            for stmt_type in standard_types:
                features_df[f'statement_{stmt_type}'] = (
                        features_df['statement_type'].str.upper() == stmt_type).astype(int)

            # Drop the original column after creating dummies
            features_df.drop(columns=['statement_type'], inplace=True)

        # 4. Dataset and table features - more controlled approach
        if 'dataset_id' in features_df.columns:
            # Identify the most common datasets (top 5)
            top_datasets = features_df['dataset_id'].value_counts().nlargest(5).index

            # Create binary indicators for the top datasets
            for dataset in top_datasets:
                features_df[f'dataset_{dataset}'] = (features_df['dataset_id'] == dataset).astype(int)

            # Add an "other_dataset" category
            features_df['dataset_Other'] = (~features_df['dataset_id'].isin(top_datasets)).astype(int)

            # Drop the original column
            features_df.drop(columns=['dataset_id'], inplace=True)

        # 5. Process table_id similarly
        if 'table_id' in features_df.columns or 'le_id' in features_df.columns:
            # Determine the correct column name
            table_col = 'table_id' if 'table_id' in features_df.columns else 'le_id'

            # Top tables
            top_tables = features_df[table_col].value_counts().nlargest(5).index

            # Binary indicators
            for table in top_tables:
                features_df[f'table_{table}'] = (features_df[table_col] == table).astype(int)

            # Other tables
            features_df['table_Other'] = (~features_df[table_col].isin(top_tables)).astype(int)

            # Drop original
            features_df.drop(columns=[table_col], inplace=True)

        # 6. Handle pipeline features
        if 'pipeline_name' in features_df.columns:
            # Top pipelines
            top_pipelines = features_df['pipeline_name'].value_counts().nlargest(3).index

            # Binary indicators
            for pipeline in top_pipelines:
                features_df[f'pipeline_{pipeline}'] = (features_df['pipeline_name'] == pipeline).astype(int)

            # Other pipelines
            features_df['pipeline_Other'] = (~features_df['pipeline_name'].isin(top_pipelines)).astype(int)

            # Drop original
            features_df.drop(columns=['pipeline_name'], inplace=True)

        # 7. Create utilization buckets if utilization_percentage exists
        if 'utilization_percentage' in features_df.columns:
            # Create more fine-grained utilization features
            features_df['util_low'] = (features_df['utilization_percentage'] <= 25).astype(int)
            features_df['util_medium'] = ((features_df['utilization_percentage'] > 25) &
                                          (features_df['utilization_percentage'] <= 50)).astype(int)
            features_df['util_high'] = ((features_df['utilization_percentage'] > 50) &
                                        (features_df['utilization_percentage'] <= 75)).astype(int)
            features_df['util_very_high'] = (features_df['utilization_percentage'] > 75).astype(int)

        # 8. Referenced tables count handling
        if 'referenced_le_count' in features_df.columns:
            features_df.rename(columns={'referenced_le_count': 'referenced_table_count'}, inplace=True)

        # 9. Drop unnecessary columns that don't contribute to prediction
        cols_to_drop = [
            'job_id', 'user_email', 'utc_start_time', 'utc_end_time',
            'est_start_time', 'est_end_time', 'error_message', 'parent_job_id',
            'reservation_id', 'tag_name', 'referenced_les', 'referenced_tables',
            'day_of_week_text'
        ]

        # Only drop if they exist
        for col in cols_to_drop:
            if col in features_df.columns:
                features_df.drop(columns=[col], inplace=True)

        # 10. Handle any remaining object/categorical columns by dropping
        object_cols = features_df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            logging.warning(f"Dropping {len(object_cols)} remaining object columns: {list(object_cols)}")
            features_df.drop(columns=object_cols, inplace=True)

        # 11. Fill missing values
        features_df.fillna(0, inplace=True)

        # Log feature statistics
        logging.info(f"Engineered feature set: {len(features_df.columns)} features, {len(features_df)} rows")

        # Save engineered features
        features_df.to_csv('optimized_features.csv', index=False)
        return features_df

    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise


def select_important_features(features_df, target_col='total_slot_ms', n_features=30):
    """
    Perform feature selection to reduce dimensionality
    """
    try:
        if target_col not in features_df.columns:
            logging.error(f"Target column {target_col} not found")
            return features_df

        # Copy the dataframe
        df = features_df.copy()

        # Get a list of columns that start with target names to exclude
        target_patterns = ['total_slot_ms', 'elapsed_time_ms']
        target_columns = [c for c in df.columns if c in target_patterns or
                          any(c.startswith(pat) for pat in target_patterns)]

        # Separate features and target
        X = df.drop(columns=target_columns)
        y = df[target_col]

        # Use a random forest to select important features
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Get feature importances and sort them
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Select top N features
        top_n = min(n_features, len(X.columns))
        selected_indices = indices[:top_n]
        selected_features = X.columns[selected_indices].tolist()

        # Add the target columns back
        selected_features.extend(target_columns)

        # Keep only the selected features
        reduced_df = features_df[selected_features]

        logging.info(f"Feature selection: reduced from {len(X.columns)} to {len(selected_features)} features")

        return reduced_df

    except Exception as e:
        logging.error(f"Error in feature selection: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return features_df  # Return original if there's an error


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_preprocess_data

    data_path = config.TRAINING_DATA_PATH
    df = load_and_preprocess_data(data_path)
    features_df = optimized_feature_engineering(df)
    reduced_df = select_important_features(features_df)

    print(f"Original features: {len(features_df.columns)}")
    print(f"Selected features: {len(reduced_df.columns)}")