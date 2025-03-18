# optimized_model_training.py - UPDATED TO SAVE ONLY BEST MODELS
import pandas as pd
import numpy as np
import os
import logging
import time
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings

import config

warnings.filterwarnings('ignore')

# Try to import lightgbm, fall back to GradientBoosting if not available
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available, using GradientBoosting instead")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Optimized configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_jobs': -1,
    'models_dir': 'models',
    'target_columns': ['total_slot_ms', 'elapsed_time_ms'],
    'model_types': ['ridge', 'rf', 'gb'],  # Using gb instead of lgb for compatibility
    'max_samples': 50000,  # Cap for training samples
    'n_iter_search': 10,  # Reduced parameter combinations
    'model_params': {
        'ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'rf': {
            'n_estimators': [50, 100],  # Reduced from [100, 200]
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt', 'log2']
        },
        'gb': {  # Using GradientBoosting instead of LightGBM
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8]
        }
    }
}

# If LightGBM is available, update configuration
if LIGHTGBM_AVAILABLE:
    MODEL_CONFIG['model_types'] = ['ridge', 'rf', 'lgb']
    MODEL_CONFIG['model_params']['lgb'] = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7],
        'num_leaves': [31, 63],
        'subsample': [0.8]
    }


def train_models(features_df, config=MODEL_CONFIG):
    """
    Train machine learning models and store only the best model for each target
    """
    start_time = time.time()
    logging.info("Starting optimized model training (saving only best models)")

    # Create models directory
    os.makedirs(config['models_dir'], exist_ok=True)
    results = {}

    try:
        # Sample data if it's too large (improves training speed)
        if len(features_df) > config['max_samples']:
            logging.info(f"Sampling {config['max_samples']} rows from {len(features_df)} for faster training")
            features_df = features_df.sample(config['max_samples'], random_state=config['random_state'])

        # Log transform target variables (handles extreme values)
        for target in config['target_columns']:
            if target in features_df.columns:
                # Store original values for inverse transform later
                features_df[f'{target}_original'] = features_df[target].copy()

                # Apply log transform to target
                logging.info(f"Applying log transform to {target}")
                features_df[target] = np.log1p(features_df[target])

        # Drop any remaining non-numeric columns
        non_numeric_cols = features_df.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_cols:
            logging.warning(f"Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
            features_df = features_df.drop(columns=non_numeric_cols)

        # For each target variable
        for target in config['target_columns']:
            if target not in features_df.columns:
                logging.error(f"Target column {target} not found in features DataFrame")
                continue

            logging.info(f"Training models for target: {target}")

            # Prepare data
            X = features_df.drop(
                columns=[c for c in features_df.columns if c.startswith(tuple(config['target_columns']))
                         or c in config['target_columns']])
            y = features_df[target]

            # Save feature columns for prediction
            feature_columns = X.columns.tolist()
            joblib.dump(feature_columns, os.path.join(config['models_dir'], f'{target}_feature_columns.joblib'))

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config['test_size'], random_state=config['random_state']
            )

            # Fill any NaN values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            joblib.dump(scaler, os.path.join(config['models_dir'], f'{target}_scaler.joblib'))

            # Create a dict to store results for this target
            target_results = {}
            best_model = None
            best_score = -float('inf')
            best_model_info = None
            best_model_type = None
            best_params = None
            feature_importance_data = None

            # Train each model type
            for model_type in config['model_types']:
                logging.info(f"Training {model_type} model for {target}")

                # Initialize model
                if model_type == 'ridge':
                    model = Ridge(random_state=config['random_state'])
                elif model_type == 'rf':
                    model = RandomForestRegressor(
                        random_state=config['random_state'],
                        n_jobs=config['n_jobs'],
                        # Use early stopping for faster training
                        warm_start=True,
                        oob_score=True
                    )
                elif model_type == 'gb':
                    model = GradientBoostingRegressor(
                        random_state=config['random_state'],
                        # Use early stopping for faster training
                        warm_start=True
                    )
                elif model_type == 'lgb' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMRegressor(
                        random_state=config['random_state'],
                        n_jobs=config['n_jobs']
                    )
                else:
                    continue  # Skip this model type if not available

                # Get parameters
                param_grid = config['model_params'][model_type]

                # Use RandomizedSearchCV (faster than GridSearchCV)
                search = RandomizedSearchCV(
                    model, param_grid,
                    n_iter=config['n_iter_search'],
                    cv=3, scoring='neg_mean_squared_error',
                    n_jobs=config['n_jobs'],
                    random_state=config['random_state']
                )

                # Time the training
                model_start_time = time.time()
                search.fit(X_train_scaled, y_train)
                model_train_time = time.time() - model_start_time

                # Get best model
                best_model_cv = search.best_estimator_

                # Evaluate
                y_pred = best_model_cv.predict(X_test_scaled)

                # Convert predictions and actual values back to original scale for metrics
                y_pred_original = np.expm1(y_pred)
                y_test_original = np.expm1(y_test)

                # Calculate metrics on original scale (compatible with older scikit-learn)
                mae = mean_absolute_error(y_test_original, y_pred_original)
                mse = mean_squared_error(y_test_original, y_pred_original)
                rmse = np.sqrt(mse)  # Calculate RMSE manually
                r2 = r2_score(y_test_original, y_pred_original)

                # Also calculate metrics on log scale
                mae_log = mean_absolute_error(y_test, y_pred)
                mse_log = mean_squared_error(y_test, y_pred)
                rmse_log = np.sqrt(mse_log)  # Calculate RMSE manually
                r2_log = r2_score(y_test, y_pred)

                # Log results
                logging.info(
                    f"{model_type} for {target} - Original Scale - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
                logging.info(
                    f"{model_type} for {target} - Log Scale - RMSE: {rmse_log:.4f}, MAE: {mae_log:.4f}, R²: {r2_log:.4f}")
                logging.info(f"Best parameters: {search.best_params_}")
                logging.info(f"Training time: {model_train_time:.2f} seconds")

                # Create model info
                model_info = {
                    'model_type': model_type,
                    'target': target,
                    'metrics': {
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'r2': float(r2),
                        'rmse_log': float(rmse_log),
                        'mae_log': float(mae_log),
                        'r2_log': float(r2_log)
                    },
                    'best_params': search.best_params_,
                    'training_time': model_train_time,
                }

                # Extract feature importance (if available)
                importance_data = None
                if hasattr(best_model_cv, 'feature_importances_'):
                    importances = best_model_cv.feature_importances_
                    importance_data = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)

                    # Add top features to model info
                    top_n = min(20, len(importance_data))
                    model_info['feature_importance'] = importance_data.head(top_n).to_dict()

                # Store in results
                target_results[model_type] = model_info

                # Update best model if this one is better (using log scale R² which is more stable)
                if r2_log > best_score:
                    best_score = r2_log
                    best_model = best_model_cv
                    best_model_type = model_type
                    best_model_info = model_info
                    best_params = search.best_params_
                    feature_importance_data = importance_data

            # Save ONLY the best model
            if best_model:
                logging.info(f"Saving best model for {target}: {best_model_type} with R² of {best_score:.4f}")

                # Save the model file
                joblib.dump(best_model, os.path.join(config['models_dir'], f'{target}_best_model.joblib'))

                # Record which model type was best
                target_results['best_model_type'] = best_model_type

                # Create feature importance plot for best model only
                if feature_importance_data is not None:
                    top_n = min(20, len(feature_importance_data))
                    plt.figure(figsize=(12, 8))
                    plt.barh(feature_importance_data.head(top_n)['Feature'],
                             feature_importance_data.head(top_n)['Importance'])
                    plt.xlabel('Importance')
                    plt.title(f'Top Features - {best_model_type.upper()} for {target}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config['models_dir'], f'{target}_best_model_importance.png'))
                    plt.close()

                # Save log transform info to use during prediction
                transform_info = {
                    'transform_type': 'log',
                    'inverse_transform': 'expm1'
                }
                joblib.dump(transform_info, os.path.join(config['models_dir'], f'{target}_transform.joblib'))

            results[target] = target_results

        # Save overall results
        with open(os.path.join(config['models_dir'], 'model_results.json'), 'w') as f:
            def convert_to_serializable(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                return obj

            # More robust serialization
            results_json = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    results_json[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            results_json[k][k2] = {}
                            for k3, v3 in v2.items():
                                results_json[k][k2][k3] = convert_to_serializable(v3)
                        else:
                            results_json[k][k2] = convert_to_serializable(v2)
                else:
                    results_json[k] = convert_to_serializable(v)

            json.dump(results_json, f, indent=4)

        total_time = time.time() - start_time
        logging.info(f"Model training completed successfully in {total_time:.2f} seconds")
        return results

    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    from feature_engineering import optimized_feature_engineering, select_important_features

    data_path = config.TRAINING_DATA_PATH

    # Load and preprocess
    df = load_and_preprocess_data(data_path)

    # Engineer features
    features_df = optimized_feature_engineering(df)

    # Select important features
    features_df = select_important_features(features_df, 'total_slot_ms', n_features=30)

    # Train models
    results = train_models(features_df)