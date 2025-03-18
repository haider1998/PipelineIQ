import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set, Optional
from concurrent.futures import ThreadPoolExecutor
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, Forbidden, NotFound


class BigQueryLLMOptimizer:
    def __init__(self, default_project_id: str):
        """
        Initialize with your default Google Cloud project ID.
        """
        self.client = bigquery.Client(project=default_project_id)
        self.default_project_id = default_project_id
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def optimize_for_llm(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and format the results specifically for LLM consumption.
        Returns structured information that an LLM can use to optimize the query.
        """
        # Get basic optimization information
        optimizer = BigQueryMultiProjectOptimizer(self.default_project_id)
        results = await optimizer.optimize_query(query)

        # Extract and format the information for LLM consumption
        llm_context = self._format_for_llm(query, results)

        return llm_context

    def _format_for_llm(self, original_query: str, optimization_results: Dict) -> Dict[str, Any]:
        """
        Format optimization results into a structure ideal for LLM consumption.
        """
        query_analysis = optimization_results['query_analysis']
        table_info = optimization_results['table_info']
        recommendations = optimization_results['recommendations']

        # 1. Build comprehensive table schemas
        table_schemas = {}
        for table_name, info in table_info.items():
            if 'error' in info:
                table_schemas[table_name] = {"error": info['error']}
                continue

            # Format table metadata in a clear, structured way
            schema = {
                "full_name": table_name,
                "row_count": info.get('basic_info', {}).get('num_rows', 'unknown'),
                "size_gb": info.get('basic_info', {}).get('size_gb', 'unknown'),
                "columns": self._format_columns(info.get('columns', [])),
                "partitioning": self._format_partitioning(info.get('partitioning')),
                "clustering": self._format_clustering(info.get('clustering')),
                "optimization_candidates": info.get('optimization_candidates', {})
            }
            table_schemas[table_name] = schema

        # 2. Analyze query patterns more deeply for LLM understanding
        query_patterns = self._analyze_query_patterns(original_query)

        # 3. Extract join conditions for LLM to understand table relationships
        join_conditions = self._extract_join_conditions(original_query)

        # 4. Analyze WHERE clause to understand filtering logic
        where_conditions = self._extract_where_conditions(original_query)

        # 5. Format recommendations with clear reasoning
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append({
                "issue": rec['message'],
                "recommendation": rec['recommendation'],
                "importance": rec['severity'],
                "type": rec['type']
            })

        # Combine everything into a comprehensive context for the LLM
        return {
            "original_query": original_query,
            "performance_metrics": {
                "estimated_gb_processed": query_analysis.get('estimated_gb_processed', 0),
                "estimated_cost_usd": query_analysis.get('estimated_cost_usd', 0)
            },
            "table_schemas": table_schemas,
            "query_structure": {
                "patterns": query_patterns,
                "join_conditions": join_conditions,
                "where_conditions": where_conditions,
                "group_by": self._extract_group_by(original_query),
                "order_by": self._extract_order_by(original_query),
                "select_columns": self._extract_select_columns(original_query)
            },
            "optimization_recommendations": formatted_recommendations,
            "query_intent": self._infer_query_intent(original_query, query_patterns)
        }

    def _format_columns(self, columns: List[Dict]) -> List[Dict]:
        """Format column information for better LLM understanding."""
        formatted = []
        for col in columns:
            formatted.append({
                "name": col.get('name'),
                "data_type": col.get('type'),
                "mode": col.get('mode', 'NULLABLE'),
                "description": self._generate_column_description(col)
            })
        return formatted

    def _generate_column_description(self, col: Dict) -> str:
        """Generate a human-readable description of a column for the LLM."""
        name = col.get('name', '').lower()
        col_type = col.get('type', '')

        # Identify likely column roles based on naming patterns
        if 'id' in name or name.endswith('key'):
            return f"Identifier column of type {col_type}"
        elif 'date' in name or col_type in ['DATE', 'TIMESTAMP', 'DATETIME']:
            return f"Temporal column of type {col_type}"
        elif 'amount' in name or 'price' in name or 'cost' in name:
            return f"Numeric measure column of type {col_type}"
        elif name in ['created_at', 'updated_at', 'modified_at']:
            return f"Record timestamp of type {col_type}"
        elif 'name' in name or 'description' in name:
            return f"Descriptive text column of type {col_type}"
        else:
            return f"Column of type {col_type}"

    def _format_partitioning(self, partitioning: Optional[Dict]) -> Dict:
        """Format partitioning information for LLM."""
        if not partitioning:
            return {"exists": False}

        return {
            "exists": True,
            "type": partitioning.get('type'),
            "field": partitioning.get('field'),
            "expiration_days": partitioning.get('expiration_ms') / (1000 * 60 * 60 * 24) if partitioning.get(
                'expiration_ms') else None
        }

    def _format_clustering(self, clustering: Optional[Dict]) -> Dict:
        """Format clustering information for LLM."""
        if not clustering or not clustering.get('fields'):
            return {"exists": False}

        return {
            "exists": True,
            "fields": clustering.get('fields')
        }

    def _analyze_query_patterns(self, query: str) -> Dict[str, Any]:
        """Perform deeper analysis of query patterns for LLM understanding."""
        # Clean the query for analysis
        query = re.sub(r'--.*?$', ' ', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)
        query = query.upper()

        # Identify common patterns
        patterns = {
            "has_join": bool(re.search(r'\bJOIN\b', query)),
            "join_types": self._identify_join_types(query),
            "has_where": bool(re.search(r'\bWHERE\b', query)),
            "has_group_by": bool(re.search(r'\bGROUP\s+BY\b', query)),
            "has_having": bool(re.search(r'\bHAVING\b', query)),
            "has_order_by": bool(re.search(r'\bORDER\s+BY\b', query)),
            "has_limit": bool(re.search(r'\bLIMIT\b', query)),
            "has_union": bool(re.search(r'\bUNION\b', query)),
            "has_with": bool(re.search(r'\bWITH\b', query)),
            "has_subquery": bool(re.search(r'\(\s*SELECT\b', query)),
            "has_window_function": bool(re.search(r'\bOVER\s*\(', query)),
            "has_star": bool(re.search(r'SELECT\s+\*', query)),
            "has_distinct": bool(re.search(r'\bDISTINCT\b', query)),
            "aggregation_functions": self._identify_aggregation_functions(query)
        }

        return patterns

    def _identify_join_types(self, query: str) -> List[str]:
        """Identify the types of joins used in the query."""
        join_types = []

        if re.search(r'\bINNER\s+JOIN\b', query) or re.search(r'\bJOIN\b', query):
            join_types.append("INNER JOIN")
        if re.search(r'\bLEFT\s+(?:OUTER\s+)?JOIN\b', query):
            join_types.append("LEFT JOIN")
        if re.search(r'\bRIGHT\s+(?:OUTER\s+)?JOIN\b', query):
            join_types.append("RIGHT JOIN")
        if re.search(r'\bFULL\s+(?:OUTER\s+)?JOIN\b', query):
            join_types.append("FULL JOIN")
        if re.search(r'\bCROSS\s+JOIN\b', query):
            join_types.append("CROSS JOIN")

        return join_types

    def _identify_aggregation_functions(self, query: str) -> List[str]:
        """Identify aggregation functions used in the query."""
        agg_functions = []

        if re.search(r'\bCOUNT\s*\(', query):
            agg_functions.append("COUNT")
        if re.search(r'\bSUM\s*\(', query):
            agg_functions.append("SUM")
        if re.search(r'\bAVG\s*\(', query):
            agg_functions.append("AVG")
        if re.search(r'\bMIN\s*\(', query):
            agg_functions.append("MIN")
        if re.search(r'\bMAX\s*\(', query):
            agg_functions.append("MAX")

        return agg_functions

    def _extract_join_conditions(self, query: str) -> List[Dict[str, str]]:
        """Extract join conditions to understand table relationships."""
        # This is a simplified implementation - a proper parser would be better
        # for complex queries, but this works for common patterns
        join_conditions = []

        # Find JOIN ... ON patterns
        join_pattern = r'\b(\w+(?:\.\w+)*)\s+(?:AS\s+)?(\w+)?\s+(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN\s+(\w+(?:\.\w+)*)\s+(?:AS\s+)?(\w+)?\s+ON\s+(.+?)(?:\s+(?:WHERE|GROUP|ORDER|LIMIT|HAVING|QUALIFY|WINDOW)\b|$)'

        matches = re.finditer(join_pattern, query, re.IGNORECASE | re.DOTALL)
        for match in matches:
            left_table = match.group(1)
            left_alias = match.group(2)
            right_table = match.group(3)
            right_alias = match.group(4)
            condition = match.group(5).strip()

            join_conditions.append({
                "left_table": left_table,
                "left_alias": left_alias if left_alias else left_table,
                "right_table": right_table,
                "right_alias": right_alias if right_alias else right_table,
                "condition": condition
            })

        return join_conditions

    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions to understand filtering logic."""
        where_clause_match = re.search(r'\bWHERE\b\s+(.+?)(?:\s+(?:GROUP|ORDER|LIMIT|HAVING|QUALIFY|WINDOW)\b|$)',
                                       query, re.IGNORECASE | re.DOTALL)

        if not where_clause_match:
            return []

        where_clause = where_clause_match.group(1).strip()

        # Split on AND, but be careful with nested conditions
        # This is simplified and won't handle all complex cases
        conditions = []

        # Handle basic AND conditions
        parts = re.split(r'\bAND\b', where_clause, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if part:
                conditions.append(part)

        return conditions

    def _extract_group_by(self, query: str) -> List[str]:
        """Extract GROUP BY columns."""
        group_by_match = re.search(r'\bGROUP\s+BY\b\s+(.+?)(?:\s+(?:ORDER|LIMIT|HAVING|QUALIFY|WINDOW)\b|$)',
                                   query, re.IGNORECASE | re.DOTALL)

        if not group_by_match:
            return []

        group_by = group_by_match.group(1).strip()

        # Split by commas, handling nested functions
        # This is a simplified approach
        columns = []

        # Simple split by comma outside of parentheses
        parts = []
        paren_level = 0
        current = ""

        for char in group_by:
            if char == ',' and paren_level == 0:
                parts.append(current.strip())
                current = ""
            else:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                current += char

        if current.strip():
            parts.append(current.strip())

        return parts

    def _extract_order_by(self, query: str) -> List[str]:
        """Extract ORDER BY columns."""
        order_by_match = re.search(r'\bORDER\s+BY\b\s+(.+?)(?:\s+(?:LIMIT|QUALIFY|WINDOW)\b|$)',
                                   query, re.IGNORECASE | re.DOTALL)

        if not order_by_match:
            return []

        order_by = order_by_match.group(1).strip()

        # Split using the same technique as GROUP BY
        parts = []
        paren_level = 0
        current = ""

        for char in order_by:
            if char == ',' and paren_level == 0:
                parts.append(current.strip())
                current = ""
            else:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                current += char

        if current.strip():
            parts.append(current.strip())

        return parts

    def _extract_select_columns(self, query: str) -> List[str]:
        """Extract columns in the SELECT clause."""
        # Find the SELECT clause
        select_match = re.search(r'\bSELECT\b\s+(.+?)\s+\bFROM\b',
                                 query, re.IGNORECASE | re.DOTALL)

        if not select_match:
            return []

        select_clause = select_match.group(1).strip()

        # If it's SELECT *, return a special indicator
        if select_clause == '*':
            return ["*"]

        # Split columns, handling nested functions
        columns = []
        paren_level = 0
        current = ""

        for char in select_clause:
            if char == ',' and paren_level == 0:
                columns.append(current.strip())
                current = ""
            else:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                current += char

        if current.strip():
            columns.append(current.strip())

        return columns

    def _infer_query_intent(self, query: str, patterns: Dict[str, Any]) -> str:
        """Infer the high-level intent of the query for the LLM."""
        intent_parts = []

        # Determine if it's an aggregation query
        if patterns.get('has_group_by') or patterns.get('aggregation_functions'):
            intent_parts.append("aggregating data")

        # Determine if it's a filtering query
        if patterns.get('has_where'):
            intent_parts.append("filtering records")

        # Determine if it's joining data
        if patterns.get('has_join'):
            intent_parts.append("joining multiple tables")

        # Determine if it's a sorted query
        if patterns.get('has_order_by'):
            intent_parts.append("sorting results")

        # Determine if it's a limited result set
        if patterns.get('has_limit'):
            intent_parts.append("returning a limited number of rows")

        if not intent_parts:
            return "retrieving data from tables"

        return "This query is " + ", ".join(intent_parts)


class BigQueryMultiProjectOptimizer:
    """The core optimizer class (same implementation as before)"""

    # ... (Keep the entire implementation from before)
    def __init__(self, default_project_id: str):
        self.client = bigquery.Client(project=default_project_id)
        self.default_project_id = default_project_id
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def optimize_query(self, query: str) -> Dict[str, Any]:
        # Implementation as before
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

    # Include all the methods from the previous implementation
    def _extract_tables_from_query(self, query: str) -> Set[str]:
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
        # Remove single line comments (-- style)
        query = re.sub(r'--.*?$', ' ', query, flags=re.MULTILINE)

        # Remove multi-line comments (/* ... */ style)
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)

        # Replace string literals with placeholders
        query = re.sub(r"'.*?'", "''", query)
        query = re.sub(r'".*?"', '""', query)

        return query

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
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


# Helper function to run the LLM Optimizer
async def run_llm_optimization(project_id: str, query: str):
    optimizer = BigQueryLLMOptimizer(project_id)
    result = await optimizer.optimize_for_llm(query)
    return result


def prepare_query_for_llm(project_id: str, query: str) -> Dict[str, Any]:
    """
    Analyze a BigQuery SQL and prepare it for an LLM to optimize.
    Returns structured information in a format ideal for LLM consumption.

    Args:
        project_id: Default GCP project ID
        query: SQL query to analyze

    Returns:
        Dict with comprehensive query information for LLM optimization
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_llm_optimization(project_id, query))
    finally:
        loop.close()


def generate_llm_prompt(query_info: Dict[str, Any]) -> str:
    """
    Generate a formatted prompt for an LLM based on the query information.

    Args:
        query_info: The structured query information from prepare_query_for_llm

    Returns:
        Formatted prompt string for the LLM
    """
    prompt = f"""# BigQuery Optimization Task

## Original Query
```sql
{query_info['original_query']}
```

## Query Intent
{query_info['query_intent']}

## Performance Metrics
- Estimated data processed: {query_info['performance_metrics']['estimated_gb_processed']} GB
- Estimated cost: ${query_info['performance_metrics']['estimated_cost_usd']}

## Table Information
"""

    # Add table schemas
    for table_name, schema in query_info['table_schemas'].items():
        if 'error' in schema:
            prompt += f"\n### Table: {table_name}\nError: {schema['error']}\n"
            continue

        prompt += f"""
### Table: {table_name}
- Rows: {schema['row_count']}
- Size: {schema['size_gb']} GB
- Partitioning: {schema['partitioning']['field'] if schema['partitioning']['exists'] else 'None'}
- Clustering: {', '.join(schema['clustering']['fields']) if schema['clustering']['exists'] else 'None'}

#### Columns:
"""
        for col in schema['columns']:
            prompt += f"- {col['name']} ({col['data_type']}): {col['description']}\n"

    # Add query structure information
    prompt += "\n## Query Structure\n"

    # Add SELECT columns
    select_cols = query_info['query_structure']['select_columns']
    prompt += "\n### SELECT Columns\n"
    if select_cols == ["*"]:
        prompt += "- All columns (*)\n"
    else:
        for col in select_cols:
            prompt += f"- {col}\n"

    # Add JOIN conditions
    if query_info['query_structure']['join_conditions']:
        prompt += "\n### JOIN Conditions\n"
        for join in query_info['query_structure']['join_conditions']:
            prompt += f"- {join['left_alias']} JOIN {join['right_alias']} ON {join['condition']}\n"

    # Add WHERE conditions
    if query_info['query_structure']['where_conditions']:
        prompt += "\n### WHERE Conditions\n"
        for condition in query_info['query_structure']['where_conditions']:
            prompt += f"- {condition}\n"

    # Add GROUP BY and ORDER BY
    if query_info['query_structure']['group_by']:
        prompt += "\n### GROUP BY\n"
        for col in query_info['query_structure']['group_by']:
            prompt += f"- {col}\n"

    if query_info['query_structure']['order_by']:
        prompt += "\n### ORDER BY\n"
        for col in query_info['query_structure']['order_by']:
            prompt += f"- {col}\n"

    # Add optimization recommendations
    prompt += "\n## Optimization Recommendations\n"
    for rec in query_info['optimization_recommendations']:
        prompt += f"- [{rec['importance'].upper()}] {rec['issue']}\n  → {rec['recommendation']}\n"

    # Final instructions for the LLM
    prompt += """
## Your Task
Please rewrite this query to improve its performance and reduce costs while maintaining the exact same output results. Focus on:

1. Using partitioning and clustering effectively
2. Reducing the amount of data processed
3. Optimizing join operations
4. Improving filter conditions
5. Selecting only necessary columns

Provide the optimized query and explain your optimization strategies.
"""

    return prompt


# Example usage
if __name__ == "__main__":
    # Example query to analyze
    query = """
    SELECT 
        *
    FROM 
        `dfa50884-d.dfa_campaign_service_op675_usa_sxm_used_fz_db.usasxmm01_owner_used_ob_pfl`
    """

    # Prepare the query information for an LLM
    query_info = prepare_query_for_llm('dfa50884-d', query)

    # Generate a formatted prompt for the LLM
    prompt = generate_llm_prompt(query_info)

    # Print or save the result
    print("\nLLM-READY OPTIMIZATION CONTEXT:")
    print("---------------------------------")
    print(prompt)
