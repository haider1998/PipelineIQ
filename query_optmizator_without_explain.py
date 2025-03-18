import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from concurrent.futures import ThreadPoolExecutor
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, Forbidden, NotFound

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
        """
        # Start timing for performance tracking
        start_time = datetime.now()

        # Extract all tables from the query with their respective projects
        tables = self._extract_tables_from_query(query)

        # Run analyses in parallel
        query_analysis, table_info = await asyncio.gather(
            self._analyze_query(query),
            self._gather_table_info(tables)
        )

        # Generate optimization recommendations
        recommendations = self._generate_recommendations(query, query_analysis, table_info)

        # Calculate total execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "query_analysis": query_analysis,
            "table_info": table_info,
            "recommendations": recommendations,
            "optimization_time_seconds": round(execution_time, 2),
            "tables_analyzed": len(tables),
            "inaccessible_tables": sum(1 for t in table_info.values() if 'error' in t)
        }

    def _extract_tables_from_query(self, query: str) -> Set[str]:
        """
        Extract fully qualified table names from query using regex.
        Handles tables from multiple projects.
        """
        # Remove comments and string literals to avoid false positives
        clean_query = self._remove_comments_and_literals(query)

        # Find table references - handles various formats
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
        """
        # Remove single line comments (-- style)
        query = re.sub(r'--.*?$', ' ', query, flags=re.MULTILINE)

        # Remove multi-line comments (/* ... */ style)
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)

        # Replace string literals with placeholders
        query = re.sub(r"'.*?'", "''", query)
        query = re.sub(r'".*?"', '""', query)

        return query

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query for cost, performance, and structure.
        No longer uses EXPLAIN statement which may not be supported.
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

            # Extract key performance indicators
            bytes_processed = dry_run_job.total_bytes_processed or 0
            slot_ms = getattr(dry_run_job, 'total_slot_ms', None)
            estimated_cost = bytes_processed / (1024 ** 3) * 5 / 1000  # $5 per TB

            # Get query patterns (without EXPLAIN)
            has_join = bool(re.search(r'\bJOIN\b', query, re.IGNORECASE))
            has_where = bool(re.search(r'\bWHERE\b', query, re.IGNORECASE))
            has_group_by = bool(re.search(r'\bGROUP\s+BY\b', query, re.IGNORECASE))
            has_order_by = bool(re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE))
            has_star = bool(re.search(r'SELECT\s+\*', query, re.IGNORECASE))

            return {
                "estimated_bytes_processed": bytes_processed,
                "estimated_gb_processed": round(bytes_processed / (1024 ** 3), 2),
                "estimated_cost_usd": round(estimated_cost, 4),
                "slot_ms": slot_ms,
                "query_patterns": {
                    "has_join": has_join,
                    "has_where": has_where,
                    "has_group_by": has_group_by,
                    "has_order_by": has_order_by,
                    "has_star": has_star
                },
                "referenced_tables_count": len(self._extract_tables_from_query(query))
            }

        except GoogleAPIError as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "estimated_bytes_processed": 0,
                "estimated_gb_processed": 0,
                "estimated_cost_usd": 0,
                "query_patterns": {}
            }

    async def _gather_table_info(self, tables: Set[str]) -> Dict[str, Dict]:
        """
        Gather information about tables in parallel across multiple projects.
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
        Analyze a single table using API calls rather than INFORMATION_SCHEMA
        which may have compatibility issues.
        """
        loop = asyncio.get_event_loop()

        # Split the fully qualified table name
        parts = table.split('.')
        if len(parts) != 3:
            return {"error": f"Invalid table reference: {table}"}

        project, dataset, table_name = parts

        try:
            # Use API method to get table instead of INFORMATION_SCHEMA
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

            # Use API methods to get basic table info
            table_info = {
                "basic_info": {
                    "table_name": table_name,
                    "dataset": dataset,
                    "project": project,
                    "num_rows": bq_table.num_rows,
                    "num_bytes": bq_table.num_bytes,
                    "size_gb": round(bq_table.num_bytes / (1024 ** 3), 2),
                    "created": bq_table.created.isoformat() if bq_table.created else None,
                    "table_type": bq_table.table_type
                },
                "partitioning": {
                    "type": bq_table.time_partitioning.type_ if bq_table.time_partitioning else None,
                    "field": bq_table.time_partitioning.field if bq_table.time_partitioning else None,
                    "expiration_ms": bq_table.time_partitioning.expiration_ms if bq_table.time_partitioning else None
                } if hasattr(bq_table, 'time_partitioning') and bq_table.time_partitioning else None,
                "clustering": {
                    "fields": bq_table.clustering_fields
                } if hasattr(bq_table, 'clustering_fields') and bq_table.clustering_fields else None,
            }

            # Get schema information
            table_info["columns"] = [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode
                }
                for field in bq_table.schema
            ] if bq_table.schema else []

            # Analyze columns for optimization candidates
            date_columns = []
            id_columns = []

            for col in table_info["columns"]:
                col_name = col["name"].lower()
                col_type = col["type"]

                # Identify potential partition columns
                if col_type in ['DATE', 'TIMESTAMP', 'DATETIME'] or any(x in col_name for x in ['date', 'time', 'day']):
                    date_columns.append(col["name"])

                # Identify potential clustering columns
                if 'id' in col_name or 'key' in col_name or 'code' in col_name:
                    id_columns.append(col["name"])

            table_info["optimization_candidates"] = {
                "partition_columns": date_columns[:3],  # Top 3 candidates
                "clustering_columns": id_columns[:4]  # Top 4 candidates
            }

            return table_info

        except GoogleAPIError as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "project": project,
                "dataset": dataset,
                "table": table_name
            }

    def _generate_recommendations(self, query: str, query_analysis: Dict,
                                  table_info: Dict) -> List[Dict]:
        """
        Generate actionable optimization recommendations.
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

        # Check query patterns
        patterns = query_analysis.get('query_patterns', {})

        # Check for SELECT *
        if patterns.get('has_star', False):
            recommendations.append({
                "type": "column_pruning",
                "severity": "medium",
                "message": "Query uses SELECT * which processes all columns.",
                "recommendation": "Select only needed columns to reduce bytes processed."
            })

        # Check for missing WHERE clause
        if not patterns.get('has_where', True) and query_analysis.get('estimated_gb_processed', 0) > 1:
            recommendations.append({
                "type": "missing_filter",
                "severity": "medium",
                "message": "Query doesn't have a WHERE clause but processes significant data.",
                "recommendation": "Add filters to reduce the amount of data scanned."
            })

        # Check for complex query patterns
        if patterns.get('has_join', False) and patterns.get('has_group_by', False) and query_analysis.get(
                'estimated_gb_processed', 0) > 10:
            recommendations.append({
                "type": "complex_query",
                "severity": "medium",
                "message": f"Query has JOINs and GROUP BY on {query_analysis.get('estimated_gb_processed')} GB of data.",
                "recommendation": "Consider materializing intermediate results or creating a view."
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
                # Get partition candidates
                candidates = info.get('optimization_candidates', {}).get('partition_columns', [])

                if candidates:
                    recommendations.append({
                        "type": "missing_partitioning",
                        "severity": "high",
                        "message": f"Table {table_name} ({table_size_gb} GB) is not partitioned.",
                        "recommendation": f"Consider partitioning on columns: {', '.join(candidates[:2])}"
                    })

            # Check for clustering opportunities
            clustering = info.get('clustering', {})
            if table_size_gb > 5 and not clustering:
                # Get clustering candidates
                candidates = info.get('optimization_candidates', {}).get('clustering_columns', [])

                if candidates:
                    recommendations.append({
                        "type": "missing_clustering",
                        "severity": "medium",
                        "message": f"Table {table_name} ({table_size_gb} GB) has no clustering keys.",
                        "recommendation": f"Consider clustering on columns: {', '.join(candidates[:3])}"
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
GROUP BY customers.name"""

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
