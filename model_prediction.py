# model_prediction.py
import pandas as pd
import numpy as np
import re
import joblib
import logging
import os
from datetime import datetime
import time

from config import BQ_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QueryPredictor:
    """Enhanced class to predict resources for BigQuery queries with log-transformed models"""

    def __init__(self, models_dir='models', cost_per_slot_ms=BQ_CONFIG['cost_per_slot_ms']):
        """
        Initialize the prediction module with trained models

        Args:
            models_dir: Directory containing the trained models
            cost_per_slot_ms: Cost per slot millisecond for cost calculations
        """
        self.models_dir = models_dir
        self.cost_per_slot_ms = cost_per_slot_ms

        # Initialize model components
        self.slot_model = None
        self.execution_model = None
        self.slot_scaler = None
        self.execution_scaler = None
        self.slot_features = None
        self.execution_features = None
        self.slot_transform = None
        self.execution_transform = None

        # Load all model components
        self._load_models()

    def _load_models(self):
        """Load all necessary model components"""
        try:
            # Load slot model and associated components
            slot_model_path = os.path.join(self.models_dir, 'total_slot_ms_model.joblib')
            if os.path.exists(slot_model_path):
                self.slot_model = joblib.load(slot_model_path)
                logging.info(f"Loaded slot model from {slot_model_path}")

                # Load scaler
                slot_scaler_path = os.path.join(self.models_dir, 'total_slot_ms_scaler.joblib')
                if os.path.exists(slot_scaler_path):
                    self.slot_scaler = joblib.load(slot_scaler_path)

                # Load feature list
                slot_features_path = os.path.join(self.models_dir, 'total_slot_ms_feature_columns.joblib')
                if os.path.exists(slot_features_path):
                    self.slot_features = joblib.load(slot_features_path)
                    logging.info(f"Loaded {len(self.slot_features)} slot features")

                # Load transform info
                slot_transform_path = os.path.join(self.models_dir, 'total_slot_ms_transform.joblib')
                if os.path.exists(slot_transform_path):
                    self.slot_transform = joblib.load(slot_transform_path)
            else:
                logging.warning(f"Slot model not found at {slot_model_path}")

            # Load execution model and associated components
            exec_model_path = os.path.join(self.models_dir, 'elapsed_time_ms_model.joblib')
            if os.path.exists(exec_model_path):
                self.execution_model = joblib.load(exec_model_path)
                logging.info(f"Loaded execution model from {exec_model_path}")

                # Load scaler
                exec_scaler_path = os.path.join(self.models_dir, 'elapsed_time_ms_scaler.joblib')
                if os.path.exists(exec_scaler_path):
                    self.execution_scaler = joblib.load(exec_scaler_path)

                # Load feature list
                exec_features_path = os.path.join(self.models_dir, 'elapsed_time_ms_feature_columns.joblib')
                if os.path.exists(exec_features_path):
                    self.execution_features = joblib.load(exec_features_path)
                    logging.info(f"Loaded {len(self.execution_features)} execution features")

                # Load transform info
                exec_transform_path = os.path.join(self.models_dir, 'elapsed_time_ms_transform.joblib')
                if os.path.exists(exec_transform_path):
                    self.execution_transform = joblib.load(exec_transform_path)
            else:
                logging.warning(f"Execution model not found at {exec_model_path}")

            logging.info("QueryPredictor initialized successfully")

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            logging.error("Using fallback estimation mode")

    def extract_query_features(self, query_text, total_bytes_processed=None):
        """
        Extract features from a query text for prediction

        Args:
            query_text: The SQL query text
            total_bytes_processed: Optional estimate of processed bytes

        Returns:
            Dictionary of features for the query
        """
        try:
            # 1. Extract query complexity metrics
            features = {}

            # Basic complexity metrics
            features['query_length'] = len(query_text)
            features['join_count'] = len(re.findall(r'\bJOIN\b', query_text, re.IGNORECASE))
            features['where_count'] = len(re.findall(r'\bWHERE\b', query_text, re.IGNORECASE))
            features['group_by_count'] = len(re.findall(r'\bGROUP\s+BY\b', query_text, re.IGNORECASE))
            features['order_by_count'] = len(re.findall(r'\bORDER\s+BY\b', query_text, re.IGNORECASE))
            features['having_count'] = len(re.findall(r'\bHAVING\b', query_text, re.IGNORECASE))
            features['distinct_count'] = len(re.findall(r'\bDISTINCT\b', query_text, re.IGNORECASE))
            features['subquery_count'] = query_text.count('(SELECT')
            features['window_function_count'] = len(re.findall(r'OVER\s*\(', query_text, re.IGNORECASE))

            # 2. Statement type detection
            for stmt_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'MERGE']:
                feature_name = f'statement_{stmt_type}'
                features[feature_name] = 1 if re.search(fr'^\s*{stmt_type}\b', query_text, re.IGNORECASE) else 0

            # 3. Time-based features (using current time for prediction)
            now = datetime.now()

            # Basic time features
            features['hour_of_day'] = now.hour
            features['day_of_week'] = now.weekday() + 1  # 1-7 format
            features['minute_of_hour'] = now.minute
            features['day'] = now.day  # Day of month

            # Cyclical encoding of time features (same as in training)
            features['hour_sin'] = np.sin(2 * np.pi * now.hour / 24.0)
            features['hour_cos'] = np.cos(2 * np.pi * now.hour / 24.0)
            features['day_sin'] = np.sin(2 * np.pi * (now.weekday()) / 7.0)
            features['day_cos'] = np.cos(2 * np.pi * (now.weekday()) / 7.0)
            features['is_weekend'] = 1 if now.weekday() >= 5 else 0

            # 4. Process bytes information
            if total_bytes_processed is None:
                # Estimate based on query complexity
                total_bytes_processed = (
                        features['query_length'] * 100000 +
                        features['join_count'] * 500000000 +
                        features['group_by_count'] * 200000000 +
                        features['window_function_count'] * 300000000 +
                        features['subquery_count'] * 400000000
                )
                # Ensure reasonable minimum
                total_bytes_processed = max(1000000, total_bytes_processed)

            features['total_bytes_processed'] = total_bytes_processed

            # 5. Derived metrics (matches training features)
            # Estimated execution time for derived features
            estimated_exec_ms = max(1000,
                                    features['join_count'] * 5000 +
                                    features['group_by_count'] * 3000 +
                                    features['window_function_count'] * 8000)

            features['bytes_per_ms'] = total_bytes_processed / estimated_exec_ms

            # Compute intensity
            features['compute_intensity'] = (features['join_count'] * 10 +
                                             features['group_by_count'] * 8 +
                                             features['window_function_count'] * 15) / max(1, features[
                'query_length'] / 100)

            # Parallelism approximation
            features['parallelism'] = min(64, 2 + features['join_count'] * 3 + features['group_by_count'])

            # Utilization buckets approximation based on complexity
            utilization = min(95, 40 + features['join_count'] * 5 + features['window_function_count'] * 10)
            features['utilization_percentage'] = utilization
            features['util_low'] = 1 if utilization <= 25 else 0
            features['util_medium'] = 1 if 25 < utilization <= 50 else 0
            features['util_high'] = 1 if 50 < utilization <= 75 else 0
            features['util_very_high'] = 1 if utilization > 75 else 0

            return features

        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            return {"query_length": len(query_text), "error": str(e)}

    def align_features(self, features, target_features):
        """
        Ensure extracted features match the format expected by the model

        Args:
            features: Dictionary of extracted features
            target_features: List of feature names expected by the model

        Returns:
            DataFrame with aligned features
        """
        # Create a DataFrame with proper structure (all zeros)
        df = pd.DataFrame(0, index=[0], columns=target_features)

        # Fill in values we have
        for col in target_features:
            if col in features:
                df[col] = features[col]

        return df

    def predict(self, query_text, total_bytes_processed=None):
        """
        Predict resources for a query

        Args:
            query_text: The SQL query text
            total_bytes_processed: Optional, estimated bytes processed

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        try:
            # Extract features
            features = self.extract_query_features(query_text, total_bytes_processed)

            # Calculate complexity metrics for response
            complexity = {
                "join_count": features.get('join_count', 0),
                "group_by_count": features.get('group_by_count', 0),
                "order_by_count": features.get('order_by_count', 0),
                "window_function_count": features.get('window_function_count', 0),
                "query_length": features.get('query_length', 0),
                "subquery_count": features.get('subquery_count', 0),
                "distinct_count": features.get('distinct_count', 0)
            }

            # Determine query category
            query_category = "UNKNOWN"
            for stmt_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'MERGE']:
                if features.get(f'statement_{stmt_type}', 0) == 1:
                    query_category = stmt_type
                    break

            # Initialize prediction results
            results = {
                "complexity": complexity,
                "query_category": query_category,
                "prediction_time_ms": 0
            }

            # Predict slot_ms if model is available
            if self.slot_model and self.slot_scaler and self.slot_features:
                # Align features with model expectations
                slot_features_df = self.align_features(features, self.slot_features)

                # Scale features
                slot_features_scaled = self.slot_scaler.transform(slot_features_df)

                # Make prediction
                slot_ms_pred_log = self.slot_model.predict(slot_features_scaled)[0]

                # Transform back to original scale if needed
                if self.slot_transform and self.slot_transform.get('transform_type') == 'log':
                    slot_ms_pred = np.expm1(slot_ms_pred_log)
                else:
                    slot_ms_pred = slot_ms_pred_log

                results["predicted_slot_ms"] = float(slot_ms_pred)
                results["predicted_slot_ms_log"] = float(slot_ms_pred_log)
            else:
                # Fallback prediction based on query complexity
                logging.warning("Using fallback prediction for slot_ms")
                slot_ms_pred = (
                        complexity['join_count'] * 100000 +
                        complexity['group_by_count'] * 80000 +
                        complexity['window_function_count'] * 120000 +
                        complexity['query_length'] * 100 +
                        complexity['subquery_count'] * 150000 +
                        50000
                )
                results["predicted_slot_ms"] = float(slot_ms_pred)
                results["prediction_type"] = "fallback"

            # Predict execution_ms if model is available
            if self.execution_model and self.execution_scaler and self.execution_features:
                # Align features with model expectations
                exec_features_df = self.align_features(features, self.execution_features)

                # Scale features
                exec_features_scaled = self.execution_scaler.transform(exec_features_df)

                # Make prediction
                exec_ms_pred_log = self.execution_model.predict(exec_features_scaled)[0]

                # Transform back to original scale if needed
                if self.execution_transform and self.execution_transform.get('transform_type') == 'log':
                    exec_ms_pred = np.expm1(exec_ms_pred_log)
                else:
                    exec_ms_pred = exec_ms_pred_log

                results["predicted_execution_ms"] = float(exec_ms_pred)
                results["predicted_execution_ms_log"] = float(exec_ms_pred_log)
            else:
                # Fallback prediction
                logging.warning("Using fallback prediction for execution_ms")
                exec_ms_pred = (
                        complexity['join_count'] * 5000 +
                        complexity['group_by_count'] * 3000 +
                        complexity['window_function_count'] * 8000 +
                        complexity['subquery_count'] * 10000 +
                        2000
                )
                results["predicted_execution_ms"] = float(exec_ms_pred)

            # Calculate cost estimate
            results["estimated_cost_usd"] = results["predicted_slot_ms"] * self.cost_per_slot_ms

            # Generate recommendations
            recommendations = self._generate_recommendations(results, features)
            results["recommendations"] = recommendations

            # Calculate prediction time
            results["prediction_time_ms"] = int((time.time() - start_time) * 1000)

            return results

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            # Return a basic fallback prediction if error occurs
            return {
                "error": str(e),
                "predicted_slot_ms": 1000000,
                "predicted_execution_ms": 10000,
                "estimated_cost_usd": 0.05,
                "complexity": {"query_length": len(query_text)},
                "query_category": "UNKNOWN",
                "recommendations": [
                    {
                        "title": "Error in Prediction",
                        "description": f"An error occurred during prediction: {str(e)}",
                        "impact": "HIGH"
                    }
                ]
            }

    def _generate_recommendations(self, results, features):
        """
        Generate optimization recommendations based on prediction results

        Args:
            results: Dictionary with prediction results
            features: Extracted query features

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # 1. Slot usage recommendations
        if results["predicted_slot_ms"] > 1000000:  # Over 1M slot ms
            if features["join_count"] > 2:
                recommendations.append({
                    "title": "Optimize JOINs",
                    "description": f"Query has {features['join_count']} JOINs which may be causing high slot usage. Consider adding filters before JOINs or optimizing JOIN order.",
                    "impact": "HIGH"
                })

            if features["window_function_count"] > 1:
                recommendations.append({
                    "title": "Optimize Window Functions",
                    "description": f"Query has {features['window_function_count']} window functions. Consider materializing intermediate results with window functions.",
                    "impact": "MEDIUM"
                })

            if features["subquery_count"] > 1:
                recommendations.append({
                    "title": "Review Subqueries",
                    "description": f"Query has {features['subquery_count']} subqueries which can increase resource usage. Consider using CTEs (WITH clauses) for better readability and potentially better performance.",
                    "impact": "MEDIUM"
                })

        # 2. Execution time recommendations
        if results["predicted_execution_ms"] > 60000:  # Over 1 minute
            recommendations.append({
                "title": "Long-Running Query",
                "description": f"Query is predicted to run for {results['predicted_execution_ms'] / 1000:.1f} seconds. Consider using partitioned tables to reduce data scanned.",
                "impact": "HIGH" if results["predicted_execution_ms"] > 300000 else "MEDIUM"  # High if over 5 min
            })

            if features["group_by_count"] > 0 and features["query_length"] > 500:
                recommendations.append({
                    "title": "Complex Aggregation",
                    "description": "Query contains GROUP BY with large query size. Consider creating summary tables for frequently used aggregations.",
                    "impact": "MEDIUM"
                })

        # 3. Cost recommendations
        cost = results["estimated_cost_usd"]
        if cost > 1.0:
            recommendations.append({
                "title": "High-Cost Query",
                "description": f"This query is estimated to cost ${cost:.2f}. Consider optimizing data scanned or refactoring.",
                "impact": "HIGH"
            })
        elif cost > 0.25:
            recommendations.append({
                "title": "Consider Cost Optimization",
                "description": f"Query cost of ${cost:.2f} could be reduced by limiting data scanned or optimizing JOINs.",
                "impact": "MEDIUM"
            })

        # 4. Specific optimizations based on query features
        if features["distinct_count"] > 0 and features["group_by_count"] > 0:
            recommendations.append({
                "title": "DISTINCT with GROUP BY",
                "description": "Using DISTINCT with GROUP BY may be redundant. GROUP BY already ensures unique rows for grouped columns.",
                "impact": "LOW"
            })

        if features["query_length"] > 1000:
            recommendations.append({
                "title": "Large Query",
                "description": "Consider refactoring this large query into smaller, more manageable parts using views or stored procedures.",
                "impact": "LOW"
            })

        # Add cache recommendation if appropriate
        if (features.get("statement_SELECT", 0) == 1 and
                results["predicted_execution_ms"] > 10000 and
                results["predicted_slot_ms"] > 100000):
            recommendations.append({
                "title": "Enable Query Cache",
                "description": "This query could benefit from BigQuery's cache. Ensure caching is enabled and avoid using non-deterministic functions if query results won't change.",
                "impact": "MEDIUM"
            })

        return recommendations


def main():
    """Example usage of the QueryPredictor"""
    predictor = QueryPredictor()

    # Example queries
    example_queries = [
        # Simple query
        """
        SELECT * FROM my_dataset.my_table
        WHERE date = '2023-01-01'
        LIMIT 1000
        """,

        # Medium complexity
        """
        SELECT 
          user_id,
          COUNT(*) as order_count,
          SUM(amount) as total_spent,
          AVG(amount) as avg_purchase
        FROM `my_project.dataset.transactions`
        WHERE transaction_date > '2023-01-01'
        GROUP BY user_id
        HAVING order_count > 5
        ORDER BY total_spent DESC
        LIMIT 1000
        """,

        # Complex query
        """
        WITH user_orders AS (
          SELECT
            user_id,
            COUNT(*) as order_count,
            SUM(amount) as total_spent
          FROM `my_project.dataset.orders`
          WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31'
          GROUP BY user_id
        ),
        user_activities AS (
          SELECT
            user_id,
            COUNT(*) as activity_count,
            COUNT(DISTINCT activity_type) as unique_activities
          FROM `my_project.dataset.activities`
          WHERE activity_date > '2023-01-01'
          GROUP BY user_id
        )
        SELECT
          u.name,
          u.email,
          u.signup_date,
          uo.order_count,
          uo.total_spent,
          ua.activity_count,
          ua.unique_activities,
          RANK() OVER (PARTITION BY u.country ORDER BY uo.total_spent DESC) as country_rank
        FROM `my_project.dataset.users` u
        LEFT JOIN user_orders uo ON u.id = uo.user_id
        LEFT JOIN user_activities ua ON u.id = ua.user_id
        WHERE u.status = 'active'
        ORDER BY uo.total_spent DESC NULLS LAST
        LIMIT 100
        """
    ]

    # Make predictions for each example
    for i, query in enumerate(example_queries):
        print(f"\n===== Example Query {i + 1} =====")
        prediction = predictor.predict(query)

        # Print results
        print("QUERY RESOURCE PREDICTION:")
        print(f"Slot Time: {prediction['predicted_slot_ms']:,.2f} ms")
        print(f"Execution Time: {prediction['predicted_execution_ms'] / 1000:,.2f} seconds")
        print(f"Estimated Cost: ${prediction['estimated_cost_usd']:.6f}")
        print(f"Query Category: {prediction['query_category']}")

        print("\nCOMPLEXITY METRICS:")
        for metric, value in prediction['complexity'].items():
            print(f"{metric.replace('_', ' ').title()}: {value}")

        print("\nRECOMMENDATIONS:")
        if prediction['recommendations']:
            for rec in prediction['recommendations']:
                print(f"[{rec['impact']}] {rec['title']}: {rec['description']}")
        else:
            print("No specific recommendations - query looks efficient!")

        print(f"\nPrediction completed in {prediction['prediction_time_ms']} ms")


if __name__ == "__main__":
    main()
