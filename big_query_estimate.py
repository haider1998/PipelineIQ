# big_query_estimate.py
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

import config
from config import BQ_CONFIG, QUERY_HISTORY_TABLE
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache

# Set up logging with a more configurable format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("queryiq.log"),
        logging.StreamHandler()
    ]
)

# Set up logger
logger = logging.getLogger("queryiq.bigquery")

# Initialize BigQuery client
try:
    client = bigquery.Client()
    logger.info("Successfully initialized BigQuery client")
except Exception as e:
    logger.error(f"Failed to initialize BigQuery client: {str(e)}")
    client = None


@lru_cache(maxsize=16)
def get_recent_queries(n: int = 20) -> pd.DataFrame:
    """
    Fetch recent query data from BigQuery

    Args:
        n: Number of recent queries to fetch

    Returns:
        DataFrame with recent query history
    """
    if not client:
        raise RuntimeError("BigQuery client not available")

    logger.info(f"Fetching {n} recent queries from BigQuery")
    start_time = time.time()

    try:
        query = f"""
        SELECT 
            utc_start_time as timestamp,
            SUBSTR(query, 1, 100) as query_snippet,
            total_slot_ms as slot_ms,
            elapsed_time_ms as execution_ms,
            total_bytes_processed as bytes_processed,
            (total_slot_ms * 0.00000001) as cost_usd,
            statement_type as query_type
        FROM `{QUERY_HISTORY_TABLE}`
        WHERE query IS NOT NULL
          AND total_slot_ms > 0
          AND elapsed_time_ms > 0
        ORDER BY utc_start_time DESC
        LIMIT {n}
        """

        df = client.query(query).to_dataframe()

        # Add additional classification based on complexity
        def classify_query(row):
            if row['execution_ms'] > 10000 or row['slot_ms'] > 100000:
                return 'Complex'
            elif row['execution_ms'] > 1000 or row['slot_ms'] > 10000:
                return 'Moderate'
            else:
                return 'Simple'

        df['query_category'] = df.apply(classify_query, axis=1)

        exec_time = time.time() - start_time
        logger.info(f"Successfully retrieved {len(df)} recent queries in {exec_time:.2f}s")
        return df

    except Exception as e:
        logger.error(f"Error fetching recent queries: {str(e)}")
        raise


@lru_cache(maxsize=8)
def get_historical_metrics(days: int = 90) -> pd.DataFrame:
    """
    Generate historical usage metrics from BigQuery data

    Args:
        days: Number of days of historical data to fetch

    Returns:
        DataFrame with daily aggregated metrics
    """
    if not client:
        raise RuntimeError("BigQuery client not available")

    logger.info(f"Fetching historical metrics for the past {days} days")
    start_time = time.time()

    try:
        query = f"""
        SELECT 
            EXTRACT(DATE FROM utc_start_time) as date,
            COUNT(*) as total_queries,
            AVG(total_slot_ms) as avg_slot_ms,
            AVG(elapsed_time_ms) as avg_execution_ms,
            SUM(total_bytes_processed) as total_bytes_processed,
            EXTRACT(DAYOFWEEK FROM utc_start_time) - 1 as day_of_week
        FROM `{QUERY_HISTORY_TABLE}`
        WHERE utc_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
          AND total_slot_ms > 0
        GROUP BY date, day_of_week
        ORDER BY date
        """

        df = client.query(query).to_dataframe()

        # Ensure date is datetime type for easier filtering
        df['date'] = pd.to_datetime(df['date']).dt.date  # Convert to date only (no time)

        # Create date range with date objects only (no time component)
        date_range = pd.date_range(end=datetime.now(), periods=days).date  # Get date part only
        date_df = pd.DataFrame({'date': date_range})
        df = pd.merge(date_df, df, on='date', how='left')

        # Fill missing data with reasonable defaults
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek  # Create from date objects
        df['total_queries'] = df['total_queries'].fillna(0)
        df['avg_slot_ms'] = df['avg_slot_ms'].fillna(0)
        df['avg_execution_ms'] = df['avg_execution_ms'].fillna(0)
        df['total_bytes_processed'] = df['total_bytes_processed'].fillna(0)

        exec_time = time.time() - start_time
        logger.info(f"Successfully retrieved historical metrics for {len(df)} days in {exec_time:.2f}s")
        return df

    except Exception as e:
        logger.error(f"Error fetching historical metrics: {str(e)}")
        raise


@lru_cache(maxsize=4)
def get_query_cost_distribution(n: int = 1000) -> np.ndarray:
    """
    Get distribution of query costs based on slot usage

    Args:
        n: Number of queries to include in distribution

    Returns:
        NumPy array of query costs
    """
    if not client:
        raise RuntimeError("BigQuery client not available")

    logger.info(f"Fetching cost distribution for {n} queries")
    start_time = time.time()

    try:
        query = f"""
        SELECT
          total_slot_ms * 0.00000001 as query_cost
        FROM `{QUERY_HISTORY_TABLE}`
        WHERE total_slot_ms > 0
          AND total_slot_ms IS NOT NULL
          AND utc_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY utc_start_time DESC
        LIMIT {n}
        """

        df = client.query(query).to_dataframe()

        exec_time = time.time() - start_time
        logger.info(f"Successfully retrieved cost distribution for {len(df)} queries in {exec_time:.2f}s")
        return df.query_cost.values

    except Exception as e:
        logger.error(f"Error fetching query cost distribution: {str(e)}")
        raise


@lru_cache(maxsize=4)
def get_slot_utilization_stats() -> Dict[str, float]:
    """
    Get slot utilization metrics from actual data

    Returns:
        Dictionary of utilization statistics
    """
    if not client:
        raise RuntimeError("BigQuery client not available")

    logger.info("Fetching slot utilization statistics")
    start_time = time.time()

    try:
        query = f"""
        SELECT
          AVG(utilization_percentage) as avg_utilization,
          MAX(utilization_percentage) as peak_utilization,
          AVG(utilization_percentage_5K) as avg_utilization_5k,
          AVG(utilization_percentage_15K) as avg_utilization_15k,
          AVG(utilization_percentage_30K) as avg_utilization_30k
        FROM `{QUERY_HISTORY_TABLE}`
        WHERE utilization_percentage IS NOT NULL
          AND utc_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        """

        df = client.query(query).to_dataframe()

        # Default values if no data
        if df.empty:
            return {
                'avg_utilization': 0.5,
                'peak_utilization': 0.8,
                'avg_utilization_5k': 0.3,
                'avg_utilization_15k': 0.4,
                'avg_utilization_30k': 0.6
            }

        result = df.iloc[0].to_dict()

        exec_time = time.time() - start_time
        logger.info(f"Successfully retrieved slot utilization stats in {exec_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error fetching slot utilization stats: {str(e)}")

        # Return reasonable defaults on error
        return {
            'avg_utilization': 0.5,
            'peak_utilization': 0.8,
            'avg_utilization_5k': 0.3,
            'avg_utilization_15k': 0.4,
            'avg_utilization_30k': 0.6
        }


def get_query_size(query_text: str) -> Dict[str, Any]:
    """
    Estimate query size/bytes processed using BigQuery's dry run

    Args:
        query_text: SQL query to estimate

    Returns:
        Dictionary with processed_bytes, error, and formatted_size
    """
    if not client:
        return {
            "processed_bytes": None,
            "error": "BigQuery client not available",
            "formatted_size": "Unknown"
        }

    logger.info("Estimating query size using dry run")
    start_time = time.time()

    try:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(query_text, job_config=job_config)

        processed_bytes = query_job.total_bytes_processed

        # Format the size
        if processed_bytes < 1024:
            formatted_size = f"{processed_bytes} B"
        elif processed_bytes < 1024 ** 2:
            formatted_size = f"{processed_bytes / 1024:.2f} KB"
        elif processed_bytes < 1024 ** 3:
            formatted_size = f"{processed_bytes / 1024 ** 2:.2f} MB"
        elif processed_bytes < 1024 ** 4:
            formatted_size = f"{processed_bytes / 1024 ** 3:.2f} GB"
        else:
            formatted_size = f"{processed_bytes / 1024 ** 4:.2f} TB"

        exec_time = time.time() - start_time
        logger.info(f"Successfully estimated query size: {formatted_size} in {exec_time:.2f}s")

        return {
            "processed_bytes": processed_bytes,
            "error": None,
            "formatted_size": formatted_size
        }

    except Exception as e:
        exec_time = time.time() - start_time
        logger.error(f"Error estimating query size: {str(e)}")

        return {
            "processed_bytes": None,
            "error": str(e),
            "formatted_size": "Unknown"
        }


@lru_cache(maxsize=8)
def get_dataset_tables_info(project_id: str, dataset_id: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Get detailed information about tables in a dataset

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID

    Returns:
        Tuple of (table_info_list, size_in_bytes_list)
    """
    if not client:
        raise RuntimeError("BigQuery client not available")

    logger.info(f"Fetching table information for {project_id}.{dataset_id}")
    start_time = time.time()

    try:
        # Get tables list
        tables = list(client.list_tables(f"{project_id}.{dataset_id}"))
        logger.info(f"Found {len(tables)} tables in dataset")

        table_info = []
        sizes_in_bytes = []

        for table in tables:
            # Get full table information
            table_ref = client.get_table(table.reference)

            # Determine partition type
            partition_type = "None"
            if table_ref.time_partitioning:
                if table_ref.time_partitioning.type_ == 'DAY':
                    partition_type = "Daily"
                elif table_ref.time_partitioning.type_ == 'HOUR':
                    partition_type = "Hourly"
                elif table_ref.time_partitioning.type_ == 'MONTH':
                    partition_type = "Monthly"
                elif table_ref.time_partitioning.type_ == 'YEAR':
                    partition_type = "Yearly"

            # Gather table information
            table_data = {
                "table_name": table.table_id,
                "row_count": table_ref.num_rows,
                "size_bytes": table_ref.num_bytes,
                "partition_type": partition_type,
                "creation_time": table_ref.created.isoformat(),
                "last_modified": table_ref.modified.isoformat(),
                "schema_fields": len(table_ref.schema),
                "clustering_fields": table_ref.clustering_fields
            }

            table_info.append(table_data)
            sizes_in_bytes.append(table_ref.num_bytes)

        exec_time = time.time() - start_time
        logger.info(f"Successfully retrieved info for {len(table_info)} tables in {exec_time:.2f}s")

        return table_info, sizes_in_bytes

    except Exception as e:
        logger.error(f"Error fetching dataset tables info: {str(e)}")
        raise


@lru_cache(maxsize=16)
def get_common_query_patterns(n: int = 10) -> List[Dict[str, Any]]:
    """
    Extract common query patterns and their performance metrics

    Args:
        n: Number of patterns to retrieve

    Returns:
        List of query pattern dictionaries with performance metrics
    """
    if not client:
        raise RuntimeError("BigQuery client not available")

    logger.info(f"Fetching {n} common query patterns")
    start_time = time.time()

    try:
        query = f"""
        WITH query_hashes AS (
          SELECT 
            MD5(REGEXP_REPLACE(LOWER(query), r'\\s+', ' ')) as query_hash,
            AVG(total_slot_ms) as avg_slot_ms,
            AVG(elapsed_time_ms) as avg_execution_ms,
            AVG(total_bytes_processed) as avg_bytes_processed,
            COUNT(*) as execution_count,
            MIN(query) as sample_query
          FROM `{QUERY_HISTORY_TABLE}`
          WHERE query IS NOT NULL
            AND utc_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
          GROUP BY query_hash
          HAVING execution_count > 2
          ORDER BY execution_count DESC
          LIMIT {n}
        )
        SELECT * FROM query_hashes
        """

        df = client.query(query).to_dataframe()

        # Convert to list of dictionaries
        patterns = df.to_dict('records')

        exec_time = time.time() - start_time
        logger.info(f"Successfully retrieved {len(patterns)} query patterns in {exec_time:.2f}s")
        return patterns

    except Exception as e:
        logger.error(f"Error fetching query patterns: {str(e)}")
        # Return empty list on error
        return []


@lru_cache(maxsize=8)
def get_table_query_patterns(project_id: str, dataset_id: str, table_id: str, days: int = 30) -> Dict[str, Any]:
    """
    Analyze query patterns for a specific table

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        days: Number of days to analyze

    Returns:
        Dictionary with query pattern analysis
    """
    if not client:
        raise RuntimeError("BigQuery client not available")

    fully_qualified_table = f"{project_id}.{dataset_id}.{table_id}"
    logger.info(f"Analyzing query patterns for table: {fully_qualified_table}")
    start_time = time.time()

    try:
        query = f"""
        WITH table_queries AS (
          SELECT 
            query,
            total_slot_ms,
            elapsed_time_ms,
            total_bytes_processed
          FROM `{QUERY_HISTORY_TABLE}`
          WHERE query IS NOT NULL
            AND REGEXP_CONTAINS(LOWER(query), LOWER(r'\\b{table_id}\\b'))
            AND utc_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        )
        SELECT
          COUNT(*) as query_count,
          AVG(total_slot_ms) as avg_slot_ms,
          AVG(elapsed_time_ms) as avg_execution_ms,
          AVG(total_bytes_processed) as avg_bytes_processed,
          COUNT(DISTINCT REGEXP_EXTRACT(LOWER(query), r'where[\\s\\S]+?(?:group by|order by|limit|$)')) as filter_count,
          COUNT(DISTINCT REGEXP_EXTRACT(LOWER(query), r'group by[\\s\\S]+?(?:having|order by|limit|$)')) as group_by_count,
          COUNT(DISTINCT REGEXP_EXTRACT(LOWER(query), r'join[\\s\\S]+?(?:where|group by|order by|limit|$)')) as join_count
        FROM table_queries
        """

        result = client.query(query).to_dataframe().iloc[0].to_dict() if client.query(query).to_dataframe().shape[
                                                                             0] > 0 else {
            'query_count': 0,
            'avg_slot_ms': 0,
            'avg_execution_ms': 0,
            'avg_bytes_processed': 0,
            'filter_count': 0,
            'group_by_count': 0,
            'join_count': 0
        }

        # Analyze common columns in WHERE clauses
        if result['query_count'] > 0:
            columns_query = f"""
            WITH table_queries AS (
              SELECT 
                query
              FROM `{QUERY_HISTORY_TABLE}`
              WHERE query IS NOT NULL
                AND REGEXP_CONTAINS(LOWER(query), LOWER(r'\\b{table_id}\\b'))
                AND REGEXP_CONTAINS(LOWER(query), r'where')
                AND utc_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
              LIMIT 100
            )
            SELECT
              REGEXP_EXTRACT_ALL(LOWER(query), r'where[\\s\\S]+?(?:group by|order by|limit|$)') as where_clauses
            FROM table_queries
            """

            where_clauses = client.query(columns_query).to_dataframe()

            # Extract column names from WHERE clauses
            columns = []
            if not where_clauses.empty and 'where_clauses' in where_clauses.columns:
                for clause_list in where_clauses['where_clauses']:
                    if clause_list:
                        for clause in clause_list:
                            # Simple extraction of potential column names
                            words = re.findall(r'([a-zA-Z0-9_]+)\s*(?:[=><!]|is|like|in|between)', clause.lower())
                            columns.extend(words)

            # Count occurrences of each column
            if columns:
                column_counts = {}
                for col in columns:
                    if col not in ['where', 'and', 'or', 'not', 'null', 'is', 'in']:
                        column_counts[col] = column_counts.get(col, 0) + 1

                # Sort by frequency
                result['common_filter_columns'] = [
                    {"column": col, "count": count}
                    for col, count in sorted(column_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                ]
            else:
                result['common_filter_columns'] = []
        else:
            result['common_filter_columns'] = []

        exec_time = time.time() - start_time
        logger.info(f"Successfully analyzed query patterns for {fully_qualified_table} in {exec_time:.2f}s")
        return result

    except Exception as e:
        logger.error(f"Error analyzing table query patterns: {str(e)}")
        # Return empty results on error
        return {
            'query_count': 0,
            'avg_slot_ms': 0,
            'avg_execution_ms': 0,
            'avg_bytes_processed': 0,
            'filter_count': 0,
            'group_by_count': 0,
            'join_count': 0,
            'common_filter_columns': []
        }


def get_cost_optimization_recommendations() -> List[Dict[str, Any]]:
    """
    Generate cost optimization recommendations based on actual query patterns

    Returns:
        List of recommendation dictionaries
    """
    logger.info("Generating data-driven cost optimization recommendations")

    try:
        # Get common query patterns to analyze
        patterns = get_common_query_patterns(20)

        # Get slot utilization stats
        utilization = get_slot_utilization_stats()

        # Initialize recommendations list
        recommendations = []

        # Check if we got data to analyze
        if patterns:
            # Look for expensive repeated queries (candidates for materialization)
            expensive_repeated = [p for p in patterns if p['execution_count'] > 5 and p['avg_slot_ms'] > 10000]
            if expensive_repeated:
                recommendations.append({
                    "title": "Materialize Frequently Run Expensive Queries",
                    "description": f"Found {len(expensive_repeated)} expensive queries that run frequently. Consider creating materialized views or scheduled query results.",
                    "impact": "HIGH",
                    "savings_estimate": "15-30% reduction in slot usage for these queries",
                    "implementation_effort": "MEDIUM"
                })

            # Look for queries reading excessive data
            data_heavy = [p for p in patterns if p['avg_bytes_processed'] > 10 * 1024 * 1024 * 1024]  # > 10GB
            if data_heavy:
                recommendations.append({
                    "title": "Optimize Data-Heavy Queries",
                    "description": f"Found {len(data_heavy)} queries processing more than 10GB of data. Review these queries for opportunities to reduce data scanned.",
                    "impact": "HIGH",
                    "savings_estimate": "20-40% reduction in bytes processed",
                    "implementation_effort": "MEDIUM"
                })

        # Check slot utilization
        if utilization.get('peak_utilization', 0) > 0.9:
            recommendations.append({
                "title": "Optimize Slot Usage During Peak Times",
                "description": f"Slot utilization exceeds 90% during peak times. Consider spreading workloads or adding slot capacity.",
                "impact": "MEDIUM",
                "savings_estimate": "Improved query performance and reduced wait times",
                "implementation_effort": "MEDIUM"
            })

        # If we didn't get enough data-driven recommendations, add some general ones
        if len(recommendations) < 3:
            general_recs = [
                {
                    "title": "Partition Large Tables",
                    "description": "Tables over 1TB should be partitioned to reduce data scanned. Consider partitioning by date for time-series data.",
                    "impact": "HIGH",
                    "savings_estimate": "20-40% reduction in data processed",
                    "implementation_effort": "MEDIUM"
                },
                {
                    "title": "Optimize JOIN Operations",
                    "description": "Add filters before JOIN operations to reduce data shuffling. Join smaller tables to larger ones, not vice versa.",
                    "impact": "HIGH",
                    "savings_estimate": "15-25% reduction in slot usage",
                    "implementation_effort": "LOW"
                },
                {
                    "title": "Use Materialized Views",
                    "description": "Create materialized views for frequently executed complex queries to avoid redundant computation.",
                    "impact": "MEDIUM",
                    "savings_estimate": "10-20% reduction in execution time and cost",
                    "implementation_effort": "LOW"
                },
                {
                    "title": "Implement Query Caching",
                    "description": "Enable query caching for read-heavy workloads. Avoid non-deterministic functions to maximize cache hits.",
                    "impact": "MEDIUM",
                    "savings_estimate": "5-15% reduction in total cost",
                    "implementation_effort": "LOW"
                },
                {
                    "title": "Schedule Batch Jobs Off-Peak",
                    "description": "Schedule large analytical jobs during off-peak hours to better utilize slot capacity and avoid contention.",
                    "impact": "LOW",
                    "savings_estimate": "Better resource utilization, no direct cost savings",
                    "implementation_effort": "LOW"
                }
            ]

            # Add general recommendations until we have at least 5
            for rec in general_recs:
                if len(recommendations) < 5 and rec not in recommendations:
                    recommendations.append(rec)

        logger.info(f"Generated {len(recommendations)} cost optimization recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating cost optimization recommendations: {str(e)}")

        # Return default recommendations on error
        return [
            {
                "title": "Partition Large Tables",
                "description": "Tables over 1TB should be partitioned to reduce data scanned. Consider partitioning by date for time-series data.",
                "impact": "HIGH",
                "savings_estimate": "20-40% reduction in data processed",
                "implementation_effort": "MEDIUM"
            },
            {
                "title": "Optimize JOIN Operations",
                "description": "Add filters before JOIN operations to reduce data shuffling. Join smaller tables to larger ones, not vice versa.",
                "impact": "HIGH",
                "savings_estimate": "15-25% reduction in slot usage",
                "implementation_effort": "LOW"
            },
            {
                "title": "Use Materialized Views",
                "description": "Create materialized views for frequently executed complex queries to avoid redundant computation.",
                "impact": "MEDIUM",
                "savings_estimate": "10-20% reduction in execution time and cost",
                "implementation_effort": "LOW"
            }
        ]


def get_query_performance_history(query_text: str,
                                  match_threshold: float = 0.8) -> Optional[Dict[str, Any]]:
    """
    Find historical performance data for similar queries

    Args:
        query_text: Query to find history for
        match_threshold: Similarity threshold for matching

    Returns:
        Dictionary with historical performance data or None
    """
    if not client:
        return None

    # For this example, we'll use a simplified approach
    # In a production system, use more sophisticated matching and ML

    # Normalize the query by removing whitespace and making lowercase
    normalized_query = re.sub(r'\s+', ' ', query_text.lower()).strip()

    try:
        # Get recent queries to compare against
        recent_queries = get_recent_queries(100)

        # Find the best match
        best_match = None
        best_score = 0

        for _, row in recent_queries.iterrows():
            if 'query_snippet' in row:
                # Very simple similarity - in production use more sophisticated algorithms
                # Count matching words as a simple similarity metric
                full_query = row['query_snippet']
                full_normalized = re.sub(r'\s+', ' ', full_query.lower()).strip()

                words1 = set(normalized_query.split())
                words2 = set(full_normalized.split())
                common_words = words1.intersection(words2)

                if len(words1) > 0 and len(words2) > 0:
                    # Jaccard similarity
                    score = len(common_words) / len(words1.union(words2))

                    if score > best_score and score >= match_threshold:
                        best_score = score
                        best_match = row

        if best_match is not None:
            return {
                'similar_query': best_match['query_snippet'],
                'similarity_score': best_score,
                'historical_slot_ms': best_match['slot_ms'],
                'historical_execution_ms': best_match['execution_ms'],
                'historical_bytes_processed': best_match['bytes_processed'] if 'bytes_processed' in best_match else None
            }

        return None

    except Exception as e:
        logger.error(f"Error finding similar query: {str(e)}")
        return None


def get_query_size(query_string):
    """
    Returns the estimated bytes processed for a BigQuery query without running it.

    Args:
        query_string (str): The SQL query to estimate

    Returns:
        dict: Contains processed_bytes and formatted size
    """

    # Configure the query job to be a dry run
    job_config = bigquery.QueryJobConfig(dry_run=True)

    try:
        # Start the query job as a dry run
        query_job = client.query(query_string, job_config=job_config)

        # Get the bytes that would be processed
        bytes_processed = query_job.total_bytes_processed

        # Format the size in a readable way
        if bytes_processed < 1024:
            formatted_size = f"{bytes_processed} B"
        elif bytes_processed < 1024 ** 2:
            formatted_size = f"{bytes_processed / 1024:.2f} KB"
        elif bytes_processed < 1024 ** 3:
            formatted_size = f"{bytes_processed / (1024 ** 2):.2f} MB"
        else:
            formatted_size = f"{bytes_processed / (1024 ** 3):.2f} GB"

        return {
            "processed_bytes": bytes_processed,
            "formatted_size": formatted_size
        }

    except GoogleCloudError as e:
        return {"error": str(e)}

def get_dataset_tables_info(project_id, dataset_id):
    """
    Returns information about all tables in a specified BigQuery dataset.

    Args:
        project_id (str): Google Cloud Project ID
        dataset_id (str): BigQuery Dataset ID

    Returns:
        list: List of dictionaries containing table information
    """
    client = bigquery.Client(project=project_id)

    # Query to get table metadata - using the correct column names
    metadata_query = f"""
    SELECT 
        table_id,  -- Changed from table_name to table_id
        size_bytes,
        row_count
    FROM 
        `{project_id}.{dataset_id}.__TABLES__`
    """

    metadata_job = client.query(metadata_query)
    metadata_results = {row['table_id']: row for row in metadata_job}  # Updated to use table_id

    # Get list of tables to check for partitioning info
    dataset_ref = client.dataset(dataset_id, project=project_id)
    tables = list(client.list_tables(dataset_ref))

    tables_info = []

    for table in tables:
        table_id = table.table_id

        if table_id in metadata_results:
            metadata = metadata_results[table_id]

            # Format size for readability
            size_bytes = metadata['size_bytes']
            if size_bytes < 1024 ** 2:
                formatted_size = f"{size_bytes / 1024:.2f} KB"
            elif size_bytes < 1024 ** 3:
                formatted_size = f"{size_bytes / (1024 ** 2):.2f} MB"
            elif size_bytes < 1024 ** 4:
                formatted_size = f"{size_bytes / (1024 ** 3):.2f} GB"
            else:
                formatted_size = f"{size_bytes / (1024 ** 4):.2f} TB"

            # Get partition information - requires fetching full table metadata
            table_ref = client.get_table(f"{project_id}.{dataset_id}.{table_id}")

            # Determine partition type
            partition_type = "Not partitioned"
            if hasattr(table_ref, 'time_partitioning') and table_ref.time_partitioning:
                partition_type_map = {
                    'DAY': 'Daily',
                    'HOUR': 'Hourly',
                    'MONTH': 'Monthly',
                    'YEAR': 'Yearly'
                }
                partition_type = partition_type_map.get(
                    table_ref.time_partitioning.type_, 'Custom')

            tables_info.append({
                "table_name": table_id,
                "data_size": formatted_size,
                "size_gb": f"{size_bytes / (1024 ** 3):.2f} GB",
                "num_rows": metadata['row_count'],
                "partition_type": partition_type
            })

    return tables_info

# Example usage
if __name__ == "__main__":
    query = """
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
    result = get_query_size(query)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"This query will process {result['formatted_size']} when run")
        print(f"Raw bytes: {result['processed_bytes']}")

    print(get_dataset_tables_info(config.PRPJECT_ID, 'user_activities'))