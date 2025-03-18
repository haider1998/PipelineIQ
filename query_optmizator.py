import asyncio
import pandas as pd
import re
import time
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, Forbidden, NotFound
from typing import Dict, List, Any, Set, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

import config


class BigQueryMultiProjectOptimizer:
    def __init__(self, default_project_id: str):
        """
        Initialize with your default Google Cloud project ID.

        Args:
            default_project_id: The default project to use when project isn't specified
        """
        self.client = bigquery.Client(project=default_project_id)
        self.default_project_id = default_project_id
        self.executor = ThreadPoolExecutor(max_workers=10)  # For parallel processing

    async def optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point - analyze query and return optimization insights.
        Handles tables from multiple projects.

        Args:
            query: The SQL query to analyze

        Returns:
            Dict containing analysis results and recommendations
        """
        # Start timing for performance tracking
        start_time = time.time()

        # Extract all tables from the query with their respective projects
        tables = self._extract_tables_from_query(query)

        # Run analyses in parallel
        query_analysis, table_info, query_history = await asyncio.gather(
            self._analyze_query(query),
            self._gather_table_info(tables),
            self._get_query_history_stats(tables)
        )

        # Generate optimization recommendations
        recommendations = self._generate_recommendations(query, query_analysis, table_info)

        # Calculate total execution time
        execution_time = time.time() - start_time

        return {
            "query_analysis": query_analysis,
            "table_info": table_info,
            "historical_performance": query_history,
            "recommendations": recommendations,
            "optimization_time_seconds": round(execution_time, 2),
            "tables_analyzed": len(tables),
            "inaccessible_tables": sum(1 for t in table_info.values() if 'error' in t)
        }

    def _extract_tables_from_query(self, query: str) -> Set[str]:
        """
        Extract fully qualified table names from query using regex.
        Handles tables from multiple projects.

        Args:
            query: SQL query to extract table references from

        Returns:
            Set of fully qualified table names
        """
        # Remove comments and string literals to avoid false positives
        clean_query = self._remove_comments_and_literals(query)

        # Find table references - handles various formats
        # This regex handles:
        # - project.dataset.table
        # - `project.dataset.table`
        # - dataset.table (will be prefixed with default project)
        patterns = [
            # Standard FROM/JOIN pattern
            r'(?:FROM|JOIN)\s+`?([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){1,2})`?',
            # UNNEST pattern
            r'UNNEST\(\(SELECT.*?FROM\s+`?([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){1,2})`?',
            # WITH clause subquery references
            r'WITH\s+[^()]*\s+AS\s+\([^()]*?FROM\s+`?([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){1,2})`?',
            # IN/EXISTS subqueries
            r'(?:IN|EXISTS)\s*\(\s*SELECT.*?FROM\s+`?([a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){1,2})`?'
        ]

        tables = set()
        for pattern in patterns:
            matches = re.finditer(pattern, clean_query, re.IGNORECASE | re.DOTALL)
            for match in matches:
                table_ref = match.group(1)
                # Ensure fully qualified name with project
                parts = table_ref.split('.')
                if len(parts) == 2:  # Only dataset.table specified
                    table_ref = f"{self.default_project_id}.{table_ref}"
                tables.add(table_ref)

        return tables

    def _remove_comments_and_literals(self, query: str) -> str:
        """
        Remove SQL comments and string literals to avoid false positives in regex matching.

        Args:
            query: Original SQL query

        Returns:
            Clean query with comments and literals removed
        """
        # Remove single line comments (-- style)
        query = re.sub(r'--.*?$', ' ', query, flags=re.MULTILINE)

        # Remove multi-line comments (/* ... */ style)
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)

        # Replace string literals with placeholders
        # This is a simplified approach - complex nested quotes might need more sophisticated handling
        query = re.sub(r"'.*?'", "''", query)
        query = re.sub(r'".*?"', '""', query)

        return query

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query for cost, performance, and structure.

        Args:
            query: SQL query to analyze

        Returns:
            Dict containing query analysis results
        """
        loop = asyncio.get_event_loop()

        try:
            # Run a dry run to get metadata without executing
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

            # Execute in thread pool to avoid blocking
            dry_run_job = await loop.run_in_executor(
                self.executor,
                lambda: self.client.query(query, job_config=job_config)
            )

            # Get execution plan with EXPLAIN
            explain_query = f"EXPLAIN {query}"
            explain_config = bigquery.QueryJobConfig(use_query_cache=False)

            explain_job = await loop.run_in_executor(
                self.executor,
                lambda: self.client.query(explain_query, job_config=explain_config)
            )

            explain_results = await loop.run_in_executor(
                self.executor,
                lambda: list(explain_job.result())
            )

            # Extract key performance indicators
            bytes_processed = dry_run_job.total_bytes_processed or 0
            slot_ms = getattr(dry_run_job, 'total_slot_ms', None)
            estimated_cost = bytes_processed / (1024 ** 3) * 5 / 1000  # $5 per TB

            # Extract operation types from the plan
            operations_count = {}
            operations_list = []

            for row in explain_results:
                plan_node = row.get('plan_node', {})
                if isinstance(plan_node, dict) and 'kind' in plan_node:
                    kind = plan_node.get('kind', '')
                    operations_count[kind] = operations_count.get(kind, 0) + 1
                    operations_list.append({
                        "kind": kind,
                        "name": plan_node.get('display_name', ''),
                        "details": {k: v for k, v in plan_node.items()
                                    if k not in ['kind', 'display_name']}
                    })

            return {
                "estimated_bytes_processed": bytes_processed,
                "estimated_gb_processed": round(bytes_processed / (1024 ** 3), 2),
                "estimated_cost_usd": round(estimated_cost, 4),
                "slot_ms": slot_ms,
                "operations_summary": operations_count,
                "operations": operations_list,
                "referenced_tables_count": len(self._extract_tables_from_query(query))
            }

        except GoogleAPIError as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def _gather_table_info(self, tables: Set[str]) -> Dict[str, Dict]:
        """
        Gather information about tables in parallel across multiple projects.

        Args:
            tables: Set of fully qualified table names

        Returns:
            Dict mapping table names to their metadata
        """
        tasks = []
        for table in tables:
            tasks.append(self._analyze_table(table))

        # Gather results from all tasks
        results = await asyncio.gather(*tasks)

        # Combine into a single dictionary
        table_info = {}
        for table, info in zip(tables, results):
            table_info[table] = info

        return table_info

    async def _analyze_table(self, table: str) -> Dict[str, Any]:
        """
        Analyze a single table using INFORMATION_SCHEMA views.
        Handles tables from different projects.

        Args:
            table: Fully qualified table name

        Returns:
            Dict containing table metadata and statistics
        """
        loop = asyncio.get_event_loop()

        # Split the fully qualified table name
        parts = table.split('.')
        if len(parts) != 3:
            return {"error": f"Invalid table reference: {table}"}

        project, dataset, table_name = parts

        try:
            # First check if we can access the table
            table_ref = f"{project}.{dataset}.{table_name}"
            try:
                bq_table = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.get_table(table_ref)
                )
            except (Forbidden, NotFound) as e:
                return {
                    "error": f"Cannot access table: {str(e)}",
                    "error_type": type(e).__name__,
                    "project": project,
                    "dataset": dataset,
                    "table": table_name
                }

            # If we can access the table, get detailed information
            metadata_query = f"""
            SELECT
                t.table_name,
                t.table_type,
                t.creation_time,
                t.last_modified_time,
                t.row_count,
                t.size_bytes,
                CASE WHEN EXISTS (
                    SELECT 1 FROM `{project}.{dataset}.INFORMATION_SCHEMA.PARTITIONS` 
                    WHERE table_name = '{table_name}' LIMIT 1
                ) THEN TRUE ELSE FALSE END as is_partitioned
            FROM
                `{project}.{dataset}.INFORMATION_SCHEMA.TABLES` t
            WHERE
                t.table_name = '{table_name}'
            """

            # Execute query in thread pool
            query_job = await loop.run_in_executor(
                self.executor,
                lambda: self.client.query(metadata_query)
            )

            metadata_df = await loop.run_in_executor(
                self.executor,
                lambda: query_job.result().to_dataframe()
            )

            if metadata_df.empty:
                return {"error": f"Table exists but no metadata available: {table}"}

            # Get column information
            columns_query = f"""
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM
                `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE
                table_name = '{table_name}'
            ORDER BY
                ordinal_position
            """

            columns_job = await loop.run_in_executor(
                self.executor,
                lambda: self.client.query(columns_query)
            )

            columns_df = await loop.run_in_executor(
                self.executor,
                lambda: columns_job.result().to_dataframe()
            )

            # Get partition information if table is partitioned
            partitions_df = pd.DataFrame()
            if metadata_df['is_partitioned'].iloc[0]:
                partitions_query = f"""
                SELECT
                    partition_id,
                    total_rows,
                    total_logical_bytes
                FROM
                    `{project}.{dataset}.INFORMATION_SCHEMA.PARTITIONS`
                WHERE
                    table_name = '{table_name}'
                ORDER BY
                    partition_id
                """

                try:
                    partitions_job = await loop.run_in_executor(
                        self.executor,
                        lambda: self.client.query(partitions_query)
                    )

                    partitions_df = await loop.run_in_executor(
                        self.executor,
                        lambda: partitions_job.result().to_dataframe()
                    )
                except GoogleAPIError:
                    # Some tables might be partitioned but we can't access partition info
                    pass

            # Get column statistics for optimization insights (only for moderately sized tables)
            column_stats = {}
            if bq_table.num_rows and bq_table.num_rows < 100000000:  # Less than 100M rows
                # Select key columns for statistics based on name patterns and data types
                potential_key_columns = []

                # Look for ID columns, date columns, and other valuable columns for optimization
                id_pattern = re.compile(r'.*_id$|^id$|.*key$|.*code$', re.IGNORECASE)
                date_pattern = re.compile(r'.*date$|.*time$|.*day$|.*month$|.*year$', re.IGNORECASE)

                for _, row in columns_df.iterrows():
                    column = row['column_name']
                    data_type = row['data_type']

                    # Prioritize columns that are likely join or filter columns
                    if (id_pattern.match(column) or
                            date_pattern.match(column) or
                            data_type in ['DATE', 'TIMESTAMP', 'DATETIME']):
                        potential_key_columns.append((column, data_type))

                # Limit to first 10 key columns to keep it fast
                for column, data_type in potential_key_columns[:10]:
                    column_stats[column] = await self._get_column_quick_stats(
                        project, dataset, table_name, column, data_type, bq_table.num_rows
                    )

            # Process metadata
            basic_info = metadata_df.iloc[0].to_dict()

            # Convert DataFrames to dicts
            columns = columns_df.to_dict('records') if not columns_df.empty else []
            partitions = partitions_df.to_dict('records') if not partitions_df.empty else []

            # Get partitioning information
            partition_info = {
                "type": bq_table.time_partitioning.type_ if bq_table.time_partitioning else None,
                "field": bq_table.time_partitioning.field if bq_table.time_partitioning else None,
                "expiration_ms": bq_table.time_partitioning.expiration_ms if bq_table.time_partitioning else None
            } if hasattr(bq_table, 'time_partitioning') and bq_table.time_partitioning else None

            # Get clustering information
            clustering_info = {
                "fields": bq_table.clustering_fields
            } if hasattr(bq_table, 'clustering_fields') and bq_table.clustering_fields else None

            return {
                "basic_info": {
                    **{k: v for k, v in basic_info.items() if k != 'is_partitioned'},
                    "size_gb": round(basic_info.get('size_bytes', 0) / (1024 ** 3), 2)
                },
                "partitioning": partition_info,
                "clustering": clustering_info,
                "columns": columns,
                "partitions_summary": {
                    "count": len(partitions),
                    "sample": partitions[:5] if len(partitions) > 5 else partitions
                } if partitions else None,
                "column_statistics": column_stats
            }

        except GoogleAPIError as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "project": project,
                "dataset": dataset,
                "table": table_name
            }

    async def _get_column_quick_stats(self, project: str, dataset: str, table_name: str,
                                      column: str, data_type: str, row_count: int) -> Dict[str, Any]:
        """
        Get fast statistics for a single column, using sampling for large tables.

        Args:
            project: GCP project ID
            dataset: BigQuery dataset
            table_name: BigQuery table name
            column: Column name to analyze
            data_type: Column data type
            row_count: Approximate row count of the table

        Returns:
            Dict containing column statistics
        """
        loop = asyncio.get_event_loop()

        # Determine if we need sampling based on table size
        sample_clause = ""
        if row_count > 10000000:  # 10M+ rows
            sample_ratio = min(1000000 / row_count * 100, 10)  # Cap at 10%
            sample_clause = f" TABLESAMPLE SYSTEM ({sample_ratio} PERCENT)"

        # Base query for all column types
        query = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT `{column}`) AS distinct_count,
            COUNTIF(`{column}` IS NULL) AS null_count,
            {100.0} * COUNT(DISTINCT `{column}`) / COUNT(*) AS distinct_percentage
        """

        # Add type-specific metrics
        if data_type in ['INT64', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC']:
            query += f"""
            , MIN(`{column}`) AS min_value
            , MAX(`{column}`) AS max_value
            , AVG(`{column}`) AS avg_value
            """
        elif data_type in ['DATE', 'TIMESTAMP', 'DATETIME']:
            query += f"""
            , MIN(`{column}`) AS min_value
            , MAX(`{column}`) AS max_value
            , TIMESTAMP_DIFF(MAX(`{column}`), MIN(`{column}`), DAY) AS day_range
            """

        query += f"\nFROM `{project}.{dataset}.{table_name}`{sample_clause}"

        try:
            query_job = await loop.run_in_executor(
                self.executor,
                lambda: self.client.query(query)
            )

            results = await loop.run_in_executor(
                self.executor,
                lambda: next(query_job.result())
            )

            stats = {
                "total_rows": results.total_rows,
                "distinct_count": results.distinct_count,
                "null_count": results.null_count,
                "cardinality_ratio": round(results.distinct_percentage / 100, 4),
                "is_sampled": bool(sample_clause),
            }

            # Add numeric stats if available
            if hasattr(results, 'min_value'):
                stats.update({
                    "min_value": results.min_value,
                    "max_value": results.max_value
                })

            if hasattr(results, 'avg_value'):
                stats["avg_value"] = results.avg_value

            if hasattr(results, 'day_range'):
                stats["day_range"] = results.day_range

            return stats

        except GoogleAPIError as e:
            return {"error": str(e), "error_type": type(e).__name__}

    async def _get_query_history_stats(self, tables: Set[str]) -> Dict[str, Any]:
        """
        Get historical performance data for queries touching these tables.

        Args:
            tables: Set of fully qualified table names

        Returns:
            Dict containing query history statistics
        """
        loop = asyncio.get_event_loop()

        # Extract table references for matching
        table_patterns = []
        for table in tables:
            parts = table.split('.')
            if len(parts) == 3:
                # Match project.dataset.table
                table_patterns.append(f"%{parts[0]}.{parts[1]}.{parts[2]}%")
                # Also match dataset.table without project
                table_patterns.append(f"%{parts[1]}.{parts[2]}%")

        # No tables to analyze
        if not table_patterns:
            return {"message": "No tables found for historical analysis"}

        # Build OR conditions for each table
        table_conditions = " OR ".join([f"query LIKE '{pattern}'" for pattern in table_patterns])

        # Query job history from the last 7 days
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        # Try to get history from main project first
        try:
            query = f"""
            SELECT
                creation_time,
                user_email,
                job_id,
                total_bytes_processed,
                total_bytes_billed,
                total_slot_ms,
                cache_hit,
                error_result,
                TIMESTAMP_DIFF(end_time, start_time, SECOND) AS duration_seconds,
                statement_type
            FROM
                `{self.default_project_id}.region-us`.INFORMATION_SCHEMA.JOBS
            WHERE
                creation_time >= '{seven_days_ago}'
                AND job_type = 'QUERY'
                AND ({table_conditions})
                AND state = 'DONE'
            ORDER BY
                creation_time DESC
            LIMIT 100
            """

            query_job = await loop.run_in_executor(
                self.executor,
                lambda: self.client.query(query)
            )

            history_df = await loop.run_in_executor(
                self.executor,
                lambda: query_job.result().to_dataframe()
            )

            if history_df.empty:
                return {"message": "No similar queries found in history"}

            # Calculate summary metrics
            avg_duration = history_df['duration_seconds'].mean()
            median_duration = history_df['duration_seconds'].median()
            p95_duration = history_df['duration_seconds'].quantile(0.95)
            avg_bytes = history_df['total_bytes_processed'].mean()
            cache_hit_rate = history_df['cache_hit'].mean() * 100 if 'cache_hit' in history_df else 0
            error_rate = (history_df['error_result'].notna().sum() / len(history_df)) * 100

            # Get trends by day
            history_df['date'] = pd.to_datetime(history_df['creation_time']).dt.date
            daily_stats = history_df.groupby('date').agg({
                'duration_seconds': 'mean',
                'total_bytes_processed': 'mean',
                'total_slot_ms': 'mean',
                'cache_hit': 'mean'
            }).reset_index()

            return {
                "query_count": len(history_df),
                "avg_duration_seconds": round(avg_duration, 2),
                "median_duration_seconds": round(median_duration, 2),
                "p95_duration_seconds": round(p95_duration, 2),
                "avg_bytes_processed": round(avg_bytes, 2),
                "avg_gb_processed": round(avg_bytes / (1024 ** 3), 4),
                "cache_hit_rate_pct": round(cache_hit_rate, 2),
                "error_rate_pct": round(error_rate, 2),
                "daily_trends": daily_stats.to_dict('records') if not daily_stats.empty else [],
                "by_statement_type": history_df.groupby('statement_type').size().to_dict()
            }

        except GoogleAPIError as e:
            return {"error": str(e), "error_type": type(e).__name__}

    def _generate_recommendations(self, query: str, query_analysis: Dict,
                                  table_info: Dict) -> List[Dict]:
        """
        Generate actionable optimization recommendations.

        Args:
            query: SQL query
            query_analysis: Dict with query analysis results
            table_info: Dict with table metadata

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Skip if query analysis failed
        if 'error' in query_analysis:
            recommendations.append({
                "type": "query_analysis_error",
                "severity": "high",
                "message": f"Query analysis failed: {query_analysis['error']}",
                "recommendation": "Check query syntax and permissions."
            })
            return recommendations

        # Check for high bytes processed
        if query_analysis.get('estimated_gb_processed', 0) > 5:
            recommendations.append({
                "type": "query_cost",
                "severity": "high" if query_analysis.get('estimated_gb_processed', 0) > 50 else "medium",
                "message": f"Query processes {query_analysis.get('estimated_gb_processed')} GB of data, which costs approximately ${query_analysis.get('estimated_cost_usd')}.",
                "recommendation": "Consider adding more specific filters or using partitioned tables."
            })

        # Check for inefficient operations in the plan
        operation_counts = query_analysis.get('operations_summary', {})
        if operation_counts.get('TABLE_SCAN', 0) > 2:
            recommendations.append({
                "type": "table_scans",
                "severity": "medium",
                "message": f"Query performs {operation_counts.get('TABLE_SCAN')} table scans.",
                "recommendation": "Consider materializing intermediate results or refactoring query."
            })

        # Check JOIN operations
        if operation_counts.get('JOIN', 0) > 3:
            recommendations.append({
                "type": "join_complexity",
                "severity": "medium",
                "message": f"Query has {operation_counts.get('JOIN')} join operations.",
                "recommendation": "Consider denormalizing data or creating materialized views."
            })

        # Check for cross-join patterns or nested loops
        for op in query_analysis.get('operations', []):
            if 'NESTED_LOOP_JOIN' in str(op):
                recommendations.append({
                    "type": "nested_loop_join",
                    "severity": "high",
                    "message": "Query uses nested loop join which can be expensive.",
                    "recommendation": "Ensure join conditions are on columns with appropriate indexes or clustering."
                })
                break

        # Check if query contains SELECT *
        if re.search(r'SELECT\s+\*', query, re.IGNORECASE):
            recommendations.append({
                "type": "column_pruning",
                "severity": "medium",
                "message": "Query uses SELECT * which processes all columns.",
                "recommendation": "Select only needed columns to reduce bytes processed."
            })

        # Check if query has proper WHERE clauses
        if not re.search(r'WHERE', query, re.IGNORECASE) and query_analysis.get('estimated_gb_processed', 0) > 1:
            recommendations.append({
                "type": "missing_filter",
                "severity": "medium",
                "message": "Query doesn't have a WHERE clause but processes significant data.",
                "recommendation": "Add filters to reduce the amount of data scanned."
            })

        # Table-specific recommendations
        for table_name, info in table_info.items():
            # Skip tables we couldn't access
            if 'error' in info:
                recommendations.append({
                    "type": "table_access_error",
                    "severity": "medium",
                    "message": f"Couldn't analyze table {table_name}: {info['error']}",
                    "recommendation": "Check permissions or verify table exists."
                })
                continue

            # Check for partitioning opportunities
            basic_info = info.get('basic_info', {})
            partitioning = info.get('partitioning', {})

            table_size_gb = basic_info.get('size_gb', 0)
            if table_size_gb > 10 and not partitioning:
                # Look for good partitioning candidates
                partition_candidates = []

                for col_name, stats in info.get('column_statistics', {}).items():
                    if isinstance(stats, dict) and 'error' not in stats:
                        if col_name.lower().endswith(('date', 'time', 'day', 'month', 'year')):
                            if stats.get('cardinality_ratio', 0) > 0.001 and stats.get('cardinality_ratio', 0) < 0.7:
                                partition_candidates.append(col_name)

                if partition_candidates:
                    recommendations.append({
                        "type": "missing_partitioning",
                        "severity": "high",
                        "message": f"Table {table_name} ({table_size_gb} GB) is not partitioned.",
                        "recommendation": f"Consider partitioning on columns: {', '.join(partition_candidates[:2])}"
                    })

            # Check for clustering opportunities
            clustering = info.get('clustering', {})
            if table_size_gb > 5 and not clustering:
                # Find good clustering candidates based on cardinality
                cluster_candidates = []

                for col_name, stats in info.get('column_statistics', {}).items():
                    if isinstance(stats, dict) and 'error' not in stats:
                        # Good clustering candidates have medium-high cardinality
                        if stats.get('cardinality_ratio', 0) > 0.001 and stats.get('cardinality_ratio', 0) < 0.9:
                            cluster_candidates.append((col_name, stats.get('cardinality_ratio', 0)))

                # Sort by cardinality and take top 3
                cluster_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = [c[0] for c in cluster_candidates[:3]]

                if top_candidates:
                    recommendations.append({
                        "type": "missing_clustering",
                        "severity": "medium",
                        "message": f"Table {table_name} ({table_size_gb} GB) has no clustering keys.",
                        "recommendation": f"Consider clustering on columns: {', '.join(top_candidates)}"
                    })

        # Check if query would benefit from materialization
        query_uses_aggregations = any(
            op in query.upper() for op in ['GROUP BY', 'COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX('])
        query_has_complex_joins = operation_counts.get('JOIN', 0) >= 2
        query_is_large = query_analysis.get('estimated_gb_processed', 0) > 20

        if query_uses_aggregations and query_has_complex_joins and query_is_large:
            recommendations.append({
                "type": "materialization",
                "severity": "medium",
                "message": "Complex query with joins and aggregations processing large data.",
                "recommendation": "Consider creating a materialized view or scheduling this as a materialization job."
            })

        # Add recommendations about query parameters if applicable
        if '?' in query or '@' in query:
            # Good - query is already using parameters
            pass
        elif re.search(r"'[0-9]{4}-[0-9]{2}-[0-9]{2}'", query):
            # Date literals could be parameterized
            recommendations.append({
                "type": "query_parameters",
                "severity": "low",
                "message": "Query contains date literals that could be parameterized.",
                "recommendation": "Use query parameters for dates to improve caching."
            })

        return recommendations


# Helper function to run the optimizer
async def run_optimization(project_id: str, query: str):
    optimizer = BigQueryMultiProjectOptimizer(project_id)
    result = await optimizer.optimize_query(query)
    return result


# Example usage
def optimize_query(project_id: str, query: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for the async optimization process.

    Args:
        project_id: Default GCP project ID
        query: SQL query to analyze

    Returns:
        Dict with optimization results
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_optimization(project_id, query))
    finally:
        loop.close()


# Example usage
if __name__ == "__main__":
    # Example query
    query = """
        SELECT 
  customers.name,
  COUNT(orders.id) as order_count
FROM customers
JOIN orders ON customers.id = orders.customer_id
WHERE orders.order_date > '2023-01-01'
GROUP BY customers.name
"""

    # Run the analysis
    try:
        results = optimize_query(config.PRPJECT_ID, query)

        # Check if there was an error in query analysis
        if 'error' not in results['query_analysis']:
            print("\nQUERY ANALYSIS:")
            print(f"Estimated cost: ${results['query_analysis']['estimated_cost_usd']}")
            print(f"Data processed: {results['query_analysis']['estimated_gb_processed']} GB")
        else:
            print("\nQUERY ANALYSIS ERROR:")
            print(f"Error: {results['query_analysis'].get('error')}")

        # Print recommendations
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. [{rec['severity'].upper()}] {rec['message']}")
            print(f"   Recommendation: {rec['recommendation']}\n")

        # Print table info summary
        print("\nTABLES ANALYZED:")
        for table, info in results['table_info'].items():
            if 'error' not in info:
                size = info.get('basic_info', {}).get('size_gb', 'Unknown')
                print(f"- {table}: {size} GB")
            else:
                print(f"- {table}: Error - {info.get('error')}")

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
