# preprocess_new_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import logging
from datetime import datetime

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_and_preprocess_data(data_path):
    """
    Load and preprocess the new BigQuery job data

    Args:
        data_path: Path to the CSV file with the job data

    Returns:
        Preprocessed DataFrame
    """
    logging.info(f"Loading data from {data_path}")

    try:
        # Load data
        df = pd.read_csv(data_path)
        logging.info(f"Loaded {len(df)} rows of data")

        # Initial data exploration
        logging.info(f"Data columns: {df.columns.tolist()}")
        logging.info(f"Missing values:\n{df.isnull().sum().sort_values(ascending=False)}")

        # Fix column names with tab characters
        df.columns = [col.replace('\t', '') for col in df.columns]
        logging.info(f"Fixed column names: {df.columns.tolist()}")

        # Convert timestamp columns to datetime
        time_columns = ['utc_start_time', 'utc_end_time', 'est_start_time', 'est_end_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Handle missing values for numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logging.info(f"Filling {missing_count} missing values in {col} with 0")
                df[col] = df[col].fillna(0)

        # Fill missing categorical columns with "unknown"
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logging.info(f"Filling {missing_count} missing values in {col} with 'unknown'")
                df[col] = df[col].fillna("unknown")

        # Clean up column names
        if "le_id" in df.columns:
            df.rename(columns={"le_id": "table_id"}, inplace=True)

        if "referenced_les" in df.columns:
            df.rename(columns={"referenced_les": "referenced_tables"}, inplace=True)

        if "referenced_le_count" in df.columns:
            df.rename(columns={"referenced_le_count": "referenced_table_count"}, inplace=True)

        # Filter out rows with actual errors (not just where error_message is non-null)
        # Modified from your original code that was filtering out all non-null error_messages
        if 'error_message' in df.columns:
            # Only filter out rows where error_message is not 'unknown' (meaning it had an actual error)
            error_rows = (df['error_message'] != 'unknown').sum()
            if error_rows > 0:
                logging.info(f"Removing {error_rows} rows with actual errors")
                df = df[df['error_message'] == 'unknown']

        # Filter out zero or negative values for target variables
        if 'total_slot_ms' in df.columns:
            invalid_count = (df['total_slot_ms'] <= 0).sum()
            if invalid_count > 0:
                logging.info(f"Removing {invalid_count} rows with invalid total_slot_ms")
                df = df[df['total_slot_ms'] > 0]

        if 'elapsed_time_ms' in df.columns:
            invalid_count = (df['elapsed_time_ms'] <= 0).sum()
            if invalid_count > 0:
                logging.info(f"Removing {invalid_count} rows with invalid elapsed_time_ms")
                df = df[df['elapsed_time_ms'] > 0]

        # Verification of data integrity
        duplicate_count = df.duplicated(subset=['job_id']).sum()
        if duplicate_count > 0:
            logging.warning(f"Found {duplicate_count} duplicate job_ids")
            df = df.drop_duplicates(subset=['job_id'])

        # Log the number of rows remaining
        logging.info(f"After preprocessing, {len(df)} rows remain")

        # Save the processed data
        output_path = os.path.join(os.path.dirname(data_path), "processed_bq_jobs.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path} with {len(df)} rows")

        # Generate some basic statistics
        logging.info("Basic statistics for key metrics:")
        if len(df) > 0:  # Only generate stats if we have data
            logging.info(
                f"total_slot_ms: min={df['total_slot_ms'].min()}, max={df['total_slot_ms'].max()}, mean={df['total_slot_ms'].mean()}")
            if 'elapsed_time_ms' in df.columns:
                logging.info(
                    f"elapsed_time_ms: min={df['elapsed_time_ms'].min()}, max={df['elapsed_time_ms'].max()}, mean={df['elapsed_time_ms'].mean()}")
            if 'total_bytes_processed' in df.columns:
                logging.info(
                    f"total_bytes_processed: min={df['total_bytes_processed'].min()}, max={df['total_bytes_processed'].max()}, mean={df['total_bytes_processed'].mean()}")
        else:
            logging.warning("No data remaining after preprocessing to generate statistics")

        return df

    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        raise


def plot_data_insights(df):
    """Generate exploratory visualizations of the data"""
    if len(df) == 0:
        logging.warning("No data to plot - skipping visualizations")
        return

    output_dir = "data_insights"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Distribution of slot usage
    plt.figure(figsize=(10, 6))
    sns.histplot(np.log1p(df['total_slot_ms']), kde=True)
    plt.title('Distribution of Log(Total Slot MS)')
    plt.xlabel('Log(Total Slot MS)')
    plt.savefig(os.path.join(output_dir, "slot_ms_distribution.png"))
    plt.close()

    # 2. Distribution of execution time
    if 'elapsed_time_ms' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(np.log1p(df['elapsed_time_ms']), kde=True)
        plt.title('Distribution of Log(Execution Time MS)')
        plt.xlabel('Log(Execution Time MS)')
        plt.savefig(os.path.join(output_dir, "execution_time_distribution.png"))
        plt.close()

    # 3. Slot usage by day of week
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='day_of_week', y='total_slot_ms', data=df)
    plt.title('Slot Usage by Day of Week')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, "slot_usage_by_day.png"))
    plt.close()

    # 4. Slot usage by hour of day
    plt.figure(figsize=(16, 6))
    sns.boxplot(x='hour_of_day', y='total_slot_ms', data=df)
    plt.title('Slot Usage by Hour of Day')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, "slot_usage_by_hour.png"))
    plt.close()

    # 5. Bytes processed vs. Slot ms
    if 'total_bytes_processed' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            np.log1p(df['total_bytes_processed']),
            np.log1p(df['total_slot_ms']),
            alpha=0.3
        )
        plt.title('Bytes Processed vs. Slot MS (Log Scale)')
        plt.xlabel('Log(Bytes Processed)')
        plt.ylabel('Log(Slot MS)')
        plt.savefig(os.path.join(output_dir, "bytes_vs_slots.png"))
        plt.close()

    # 6. Statement type distribution
    if 'statement_type' in df.columns and len(df['statement_type'].unique()) > 0:
        plt.figure(figsize=(12, 6))
        statement_counts = df['statement_type'].value_counts()
        statement_counts.plot(kind='bar')
        plt.title('Distribution of Statement Types')
        plt.savefig(os.path.join(output_dir, "statement_types.png"))
        plt.close()

    logging.info(f"Saved exploratory visualizations to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = config.TRAINING_DATA_PATH
    df = load_and_preprocess_data(data_path)

    if len(df) > 0:
        plot_data_insights(df)
    else:
        logging.warning("No data to analyze after preprocessing")
