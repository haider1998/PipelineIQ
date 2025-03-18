# streamlit_app.py - Enhanced version with improved data integration and advanced features
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import zipfile
from io import BytesIO
import os
import re
import logging
from datetime import datetime, timedelta
import altair as alt
import json
import uuid
import time
import traceback
from typing import Dict, List, Tuple, Any, Optional

# Import custom modules
import config
from big_query_estimate import (
    get_query_size,
    get_dataset_tables_info,
    get_recent_queries,
    get_historical_metrics,
    get_query_cost_distribution,
    get_slot_utilization_stats,
    get_cost_optimization_recommendations,
    get_table_query_patterns,
    get_query_performance_history
)
from config import BQ_CONFIG
from model_prediction import QueryPredictor
from llm_preprocessing import prepare_query_for_llm, generate_llm_prompt
from llm_generator import query_gemini_api

# Set up logging with a more configurable format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("PipelineIQ.log"),
        logging.StreamHandler()
    ]
)
# Create a logger for this module
logger = logging.getLogger("PipelineIQ")

# Constants
COST_PER_SLOT_MS = BQ_CONFIG['cost_per_slot_ms']
SESSION_ID = str(uuid.uuid4())  # Generate unique session ID
HISTORY_FILE = "query_history.csv"
ANALYTICS_FILE = "usage_analytics.json"

# Page configuration
st.set_page_config(
    page_title="PipelineIQ - Pipeline Resource Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info(f"Application started with session ID: {SESSION_ID}")


# Initialize QueryPredictor
@st.cache_resource
def get_predictor():
    try:
        logger.info("Initializing QueryPredictor")
        start_time = time.time()
        predictor = QueryPredictor()
        end_time = time.time()
        logger.info(f"QueryPredictor initialized successfully in {end_time - start_time:.2f} seconds")
        return predictor
    except Exception as e:
        logger.error(f"Failed to initialize QueryPredictor: {str(e)}")
        logger.debug(f"QueryPredictor initialization error details: {traceback.format_exc()}")
        raise


# Load recent queries for history
@st.cache_data(ttl=600)  # Reduced TTL to 10 minutes for more fresh data
def load_recent_queries(n=20):
    """Load recent queries from BigQuery history"""
    logger.debug(f"Attempting to load {n} recent queries from BigQuery")
    try:
        from big_query_estimate import get_recent_queries
        df = get_recent_queries(n)
        logger.info(f"Successfully loaded {len(df)} queries from BigQuery")
        return df
    except Exception as e:
        logger.warning(f"Could not load query history from BigQuery: {str(e)}")
        logger.debug(f"History loading error details: {traceback.format_exc()}")

        # Fall back to sample data if BigQuery fetch fails
        logger.info("Generating sample data due to BigQuery fetch failure")
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n)],
            'query_snippet': ["SELECT * FROM orders WHERE date > '2023-01-01'"] * n,
            'slot_ms': np.random.randint(1000, 1000000, n, dtype=np.int64),  # Specify dtype=np.int64
            'execution_ms': np.random.randint(100, 60000, n, dtype=np.int64),  # Specify dtype=np.int64
            'bytes_processed': np.random.randint(1000000, 10000000000, n, dtype=np.int64),  # Specify dtype=np.int64
        })
        return sample_data



# Load historical metrics
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_historical_metrics():
    """Load historical metrics from BigQuery data"""
    logger.debug("Attempting to load historical metrics from BigQuery")
    try:
        return get_historical_metrics(90)  # Get 90 days of data
    except Exception as e:
        logger.warning(f"Could not load historical metrics from BigQuery: {str(e)}")
        logger.debug(f"Historical metrics loading error details: {traceback.format_exc()}")

        # Generate sample data if BigQuery fetch fails
        return generate_sample_historical_metrics()


# Generate sample data as fallback
def generate_sample_historical_metrics():
    logger.info("Generating sample historical metrics data")
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'total_queries': np.random.randint(500, 2000, len(dates), dtype=np.int64),
        'avg_slot_ms': np.random.randint(50000, 500000, len(dates), dtype=np.int64),
        'avg_execution_ms': np.random.randint(2000, 30000, len(dates), dtype=np.int64),
        'total_bytes_processed': (np.random.random(len(dates)) * 9e10 + 1e9).astype(np.int64),
    })

    # Add weekly patterns
    logger.debug("Adding weekly patterns to sample historical metrics")
    day_multipliers = {0: 1.2, 1: 1.5, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.6, 6: 0.4}  # Mon-Sun
    df['day_of_week'] = df['date'].dt.dayofweek
    for col in ['total_queries', 'avg_slot_ms', 'avg_execution_ms', 'total_bytes_processed']:
        df[col] = df.apply(lambda row: row[col] * day_multipliers[row['day_of_week']], axis=1)

    logger.info(f"Generated sample historical metrics with {len(df)} records")
    return df


# Get cost optimization recommendations with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_cost_recommendations():
    """Get cost optimization recommendations with caching"""
    try:
        return get_cost_optimization_recommendations()
    except Exception as e:
        logger.warning(f"Error getting cost recommendations: {str(e)}")
        # Return default recommendations on error
        return [
            {
                "title": "Partition Large Tables",
                "description": "Tables over 1TB should be partitioned to reduce data scanned.",
                "impact": "HIGH",
                "savings_estimate": "20-40% reduction in data processed",
                "implementation_effort": "MEDIUM"
            },
            {
                "title": "Optimize JOIN Operations",
                "description": "Add filters before JOIN operations to reduce data shuffling.",
                "impact": "HIGH",
                "savings_estimate": "15-25% reduction in slot usage",
                "implementation_effort": "LOW"
            }
        ]


# Get slot utilization stats with caching
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_cached_slot_utilization():
    """Get slot utilization statistics with caching"""
    try:
        return get_slot_utilization_stats()
    except Exception as e:
        logger.warning(f"Error getting slot utilization: {str(e)}")
        # Return default stats on error
        return {
            'avg_utilization': 0.5,
            'peak_utilization': 0.8,
            'avg_utilization_5k': 0.3,
            'avg_utilization_15k': 0.4,
            'avg_utilization_30k': 0.6
        }


def format_time(milliseconds):
    """Format time values to appropriate units (ms, sec, min, hours)"""
    if milliseconds < 1000:  # Less than 1 second
        return f"{milliseconds:.2f} ms"
    elif milliseconds < 60000:  # Less than 1 minute
        return f"{milliseconds / 1000:.2f} sec"
    elif milliseconds < 3600000:  # Less than 1 hour
        minutes = milliseconds / 60000
        return f"{minutes:.2f} min"
    else:  # Hours or more
        hours = milliseconds / 3600000
        return f"{hours:.2f} hrs"


def format_bytes(bytes_value):
    """Format byte values to appropriate units (B, KB, MB, GB, TB)"""
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.2f} KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / 1024 ** 2:.2f} MB"
    elif bytes_value < 1024 ** 4:
        return f"{bytes_value / 1024 ** 3:.2f} GB"
    else:
        return f"{bytes_value / 1024 ** 4:.2f} TB"


def format_number(num, is_bytes=False, is_time=False):
    """Format numbers for display with appropriate units"""
    if is_time:
        return format_time(num)
    elif is_bytes:
        return format_bytes(num)
    elif num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"


def record_usage_analytics(query_type, prediction_result):
    """Record usage analytics for improving the tool"""
    logger.debug(f"Recording usage analytics for query type: {query_type}")
    try:
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'session_id': SESSION_ID,
            'query_type': query_type,
            'slot_ms': prediction_result.get('predicted_slot_ms', 0),
            'execution_ms': prediction_result.get('predicted_execution_ms', 0),
            'complexity': prediction_result.get('complexity', {}),
            'recommendation_count': len(prediction_result.get('recommendations', []))
        }

        # Load existing analytics
        if os.path.exists(ANALYTICS_FILE):
            logger.debug(f"Loading existing analytics from {ANALYTICS_FILE}")
            with open(ANALYTICS_FILE, 'r') as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {ANALYTICS_FILE}, resetting analytics")
                    existing = []
        else:
            logger.debug(f"No existing analytics file found, creating new")
            existing = []

        # Append new analytics
        existing.append(analytics)

        # Save back (limit to last 1000 entries)
        logger.debug(f"Saving updated analytics to {ANALYTICS_FILE}")
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(existing[-1000:], f)

        logger.info(f"Successfully recorded analytics for query type: {query_type}")

    except Exception as e:
        logger.warning(f"Could not record analytics: {str(e)}")
        logger.debug(f"Analytics recording error details: {traceback.format_exc()}")


def save_to_history(query_text, prediction):
    """Save query and prediction to history"""
    logger.debug("Saving query and prediction to history")
    try:
        # Create a new record
        new_record = pd.DataFrame({
            'timestamp': [datetime.now().isoformat()],
            'query_snippet': [query_text[:100] + "..." if len(query_text) > 100 else query_text],
            'slot_ms': [prediction['predicted_slot_ms']],
            'execution_ms': [prediction['predicted_execution_ms']],
            'bytes_processed': [prediction.get('total_bytes_processed', 0)],
            'cost_usd': [prediction['estimated_cost_usd']],
            'query_category': [prediction['query_category']]
        })

        # Append to existing history or create new file
        if os.path.exists(HISTORY_FILE):
            logger.debug(f"Appending to existing history file: {HISTORY_FILE}")
            history = pd.read_csv(HISTORY_FILE)
            history = pd.concat([new_record, history]).reset_index(drop=True)
        else:
            logger.debug(f"Creating new history file: {HISTORY_FILE}")
            history = new_record

        # Keep only last 100 entries
        if len(history) > 100:
            logger.debug("Trimming history to 100 entries")
            history = history.head(100)

        # Save back to file
        history.to_csv(HISTORY_FILE, index=False)
        logger.info(f"Successfully saved query to history, total entries: {len(history)}")

    except Exception as e:
        logger.warning(f"Could not save query to history: {str(e)}")
        logger.debug(f"History saving error details: {traceback.format_exc()}")


def analyze_query(query_text):
    """Analyze a single query and return prediction results"""
    query_length = len(query_text)
    query_snippet = query_text[:50] + "..." if query_length > 50 else query_text
    logger.info(f"Analyzing query: {query_snippet}")
    logger.debug(f"Query length: {query_length} characters")

    try:
        # Get predictor
        logger.debug("Getting QueryPredictor instance")
        predictor = get_predictor()

        # Estimate query size
        logger.debug("Estimating query size")
        size_start_time = time.time()
        size_result = get_query_size(query_text)
        size_estimation_time = (time.time() - size_start_time) * 1000  # ms
        logger.debug(f"Query size estimation completed in {size_estimation_time:.2f} ms")

        if size_result["processed_bytes"]:
            total_bytes_processed = size_result["processed_bytes"]
            bytes_message = f"Estimated data processed: {size_result['formatted_size']}"
            logger.info(f"Query size estimated successfully: {size_result['formatted_size']}")
        else:
            # Default to 100MB if estimation fails
            total_bytes_processed = 100_000_000
            bytes_message = f"Could not estimate query size: {size_result['error']}. Using default estimate (100MB)."
            logger.warning(f"Query size estimation failed: {size_result['error']}, using default 100MB")

        # Look for similar queries in history
        logger.debug("Looking for similar queries in history")
        similar_query = get_query_performance_history(query_text)

        # Make prediction
        logger.debug("Making prediction")
        start_time = time.time()
        prediction = predictor.predict(query_text, total_bytes_processed)
        prediction_time = (time.time() - start_time) * 1000  # ms
        logger.info(f"Prediction completed in {prediction_time:.2f} ms")

        # Add bytes processed to the prediction
        prediction["total_bytes_processed"] = total_bytes_processed

        # Add prediction time
        prediction["prediction_time_ms"] = prediction_time

        # Add similar query information if available
        if similar_query:
            logger.info(f"Found similar query with {similar_query['similarity_score']:.2f} similarity score")
            prediction["similar_query"] = similar_query

        # Log prediction results
        logger.info(f"Prediction results: slot_ms={prediction['predicted_slot_ms']}, " +
                    f"execution_ms={prediction['predicted_execution_ms']}, " +
                    f"cost=${prediction['estimated_cost_usd']:.6f}, " +
                    f"category={prediction['query_category']}")

        # Log recommendations count
        recommendation_count = len(prediction.get('recommendations', []))
        if recommendation_count > 0:
            logger.info(f"Generated {recommendation_count} optimization recommendations")
            # Log high impact recommendations
            high_impact = [r for r in prediction.get('recommendations', []) if r['impact'] == 'HIGH']
            if high_impact:
                logger.info(f"Found {len(high_impact)} HIGH impact recommendations")

        # Save to history
        logger.debug("Saving to history")
        save_to_history(query_text, prediction)

        # Record analytics
        logger.debug("Recording analytics")
        record_usage_analytics("single_query", prediction)

        return prediction, bytes_message, None
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        logger.debug(f"Query analysis error details: {traceback.format_exc()}")
        return None, None, f"Error analyzing query: {str(e)}"


def extract_sql_files_from_zip(zip_file):
    """Extract SQL files from a ZIP archive"""
    logger.info(f"Extracting SQL files from ZIP: {zip_file.name}")
    try:
        # Create a BytesIO object from the zip file
        zip_bytes = BytesIO(zip_file.read())

        sql_files = {}
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            # Get all SQL files in the zip
            file_list = [f for f in zip_ref.namelist() if f.endswith('.sql')]
            logger.info(f"Found {len(file_list)} SQL files in ZIP archive")

            # Extract each SQL file's content
            for file_name in file_list:
                logger.debug(f"Extracting file: {file_name}")
                sql_content = zip_ref.read(file_name).decode('utf-8')
                sql_files[file_name] = sql_content

        logger.info(f"Successfully extracted {len(sql_files)} SQL files from ZIP")
        return sql_files, None
    except zipfile.BadZipFile:
        error_msg = f"'{zip_file.name}' is not a valid ZIP file."
        logger.error(error_msg)
        return {}, error_msg
    except Exception as e:
        error_msg = f"Error processing '{zip_file.name}': {str(e)}"
        logger.error(error_msg)
        logger.debug(f"ZIP extraction error details: {traceback.format_exc()}")
        return {}, error_msg


def create_complexity_radar_chart(complexity):
    """Create a radar chart for query complexity metrics"""
    logger.debug("Creating complexity radar chart")
    categories = ['Joins', 'GROUP BYs', 'ORDER BYs', 'Window Functions', 'Subqueries', 'Query Length (÷100)']
    values = [
        complexity['join_count'],
        complexity['group_by_count'],
        complexity['order_by_count'],
        complexity['window_function_count'],
        complexity['subquery_count'],
        complexity['query_length'] / 100
    ]

    logger.debug(f"Complexity metrics: joins={complexity['join_count']}, " +
                 f"group_bys={complexity['group_by_count']}, " +
                 f"order_bys={complexity['order_by_count']}, " +
                 f"window_funcs={complexity['window_function_count']}, " +
                 f"subqueries={complexity['subquery_count']}, " +
                 f"query_length={complexity['query_length']}")

    # Create radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Query Complexity',
        line=dict(color='rgb(30, 136, 229)'),
        fillcolor='rgba(30, 136, 229, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2 or 1]
            )
        ),
        height=400,
        margin=dict(l=60, r=60, t=20, b=20)
    )
    logger.debug("Complexity radar chart created successfully")
    return fig


def display_prediction_results(prediction, bytes_message):
    """Display prediction results with enhanced visualizations"""
    logger.info("Displaying prediction results")
    # Display bytes message
    if "Estimated" in bytes_message:
        st.success(bytes_message)
        logger.debug(f"Displayed success message: {bytes_message}")
    else:
        st.warning(bytes_message)
        logger.debug(f"Displayed warning message: {bytes_message}")

    # Display prediction time
    prediction_time = prediction.get('prediction_time_ms', 0)
    st.caption(f"Prediction generated in {prediction_time:.1f} ms")
    logger.debug(f"Prediction time: {prediction_time:.1f} ms")

    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_number(prediction["predicted_slot_ms"], is_time=True)}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Predicted Slot Time</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{format_number(prediction["predicted_execution_ms"], is_time=True)}</div>',
            unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Execution Time</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${prediction["estimated_cost_usd"]:.6f}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Estimated Cost</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{prediction["query_category"]}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Query Category</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    logger.debug("Displayed main metrics cards")

    # Similar query comparison if available
    if "similar_query" in prediction:
        st.subheader("Similar Query Found in History")
        similar = prediction["similar_query"]

        st.markdown(f"""
        <div class="info-box">
            <strong>Found a similar query with {similar['similarity_score']:.2f} similarity score!</strong><br>
            This allows for more accurate predictions based on actual historical performance.
        </div>
        """, unsafe_allow_html=True)

        # Compare predicted vs historical
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Historical Performance")
            hist_metrics = {
                "Slot Time": format_number(similar['historical_slot_ms'], is_time=True),
                "Execution Time": format_number(similar['historical_execution_ms'], is_time=True),
                "Data Processed": format_number(similar['historical_bytes_processed'], is_bytes=True) if similar[
                    'historical_bytes_processed'] else "Unknown"
            }

            for metric, value in hist_metrics.items():
                st.metric(metric, value)

            with st.expander("View Similar Query"):
                st.code(similar['similar_query'], language="sql")

        with col2:
            # Compare values
            st.markdown("### Performance Comparison")

            # Calculate differences as percentages
            slot_diff = ((prediction["predicted_slot_ms"] - similar['historical_slot_ms']) / similar[
                'historical_slot_ms']) * 100 if similar['historical_slot_ms'] > 0 else 0
            exec_diff = ((prediction["predicted_execution_ms"] - similar['historical_execution_ms']) / similar[
                'historical_execution_ms']) * 100 if similar['historical_execution_ms'] > 0 else 0

            # Create comparison chart
            comp_fig = go.Figure()

            # Add bars for comparison
            comp_fig.add_trace(go.Bar(
                x=['Slot Time', 'Execution Time'],
                y=[similar['historical_slot_ms'], similar['historical_execution_ms']],
                name='Historical',
                marker_color='rgba(50, 171, 96, 0.7)'
            ))

            comp_fig.add_trace(go.Bar(
                x=['Slot Time', 'Execution Time'],
                y=[prediction["predicted_slot_ms"], prediction["predicted_execution_ms"]],
                name='Predicted',
                marker_color='rgba(55, 128, 191, 0.7)'
            ))

            # Layout
            comp_fig.update_layout(
                barmode='group',
                title='Historical vs Predicted Performance',
                yaxis=dict(
                    title='Milliseconds',
                    type='log'  # Use log scale to show both large and small values
                ),
                height=300
            )

            st.plotly_chart(comp_fig, use_container_width=True)

            # Display the differences
            st.markdown(f"""
            **Prediction Variance from History:**
            - Slot Time: {slot_diff:+.1f}%
            - Execution Time: {exec_diff:+.1f}%
            """)

    # Query complexity visualization
    st.subheader("Query Complexity Analysis")
    logger.debug("Displaying complexity analysis section")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Create radar chart for complexity metrics
        radar_fig = create_complexity_radar_chart(prediction["complexity"])
        st.plotly_chart(radar_fig)
        logger.debug("Displayed complexity radar chart")

    with col2:
        # Show complexity breakdown with colored metrics
        st.markdown("### Complexity Metrics")
        metrics = {
            "JOIN clauses": prediction["complexity"]['join_count'],
            "GROUP BY clauses": prediction["complexity"]['group_by_count'],
            "ORDER BY clauses": prediction["complexity"]['order_by_count'],
            "Window functions": prediction["complexity"]['window_function_count'],
            "Subqueries": prediction["complexity"]['subquery_count'],
            "Query length (chars)": prediction["complexity"]['query_length']
        }

        for label, value in metrics.items():
            # Color code metrics based on complexity
            if label == "JOIN clauses" and value > 3:
                delta_color = "inverse"
                logger.debug(f"High complexity detected: {label}={value}")
            elif label == "Window functions" and value > 2:
                delta_color = "inverse"
                logger.debug(f"High complexity detected: {label}={value}")
            elif label == "Subqueries" and value > 2:
                delta_color = "inverse"
                logger.debug(f"High complexity detected: {label}={value}")
            elif label == "Query length (chars)" and value > 1000:
                delta_color = "inverse"
                logger.debug(f"High complexity detected: {label}={value}")
            else:
                delta_color = "normal"

            st.metric(label, value, delta_color=delta_color)

        logger.debug("Displayed complexity metrics")

    # New performance visualization
    st.subheader("Performance Analysis")
    logger.debug("Displaying performance analysis section")

    # Create a gauge chart for slot utilization
    col1, col2 = st.columns(2)

    with col1:
        # Calculate a simple efficiency score (0-100) based on predictions
        bytes_per_slot = prediction["total_bytes_processed"] / max(prediction["predicted_slot_ms"], 1)

        # Normalize to 0-100 scale (higher is better)
        efficiency_score = min(100, bytes_per_slot / 5000 * 100)
        logger.debug(f"Calculated efficiency score: {efficiency_score:.2f}/100")

        # Create gauge chart for efficiency score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=efficiency_score,
            title={'text': "Query Efficiency Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightcoral"},
                    {'range': [33, 66], 'color': "khaki"},
                    {'range': [66, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig)
        logger.debug("Displayed efficiency score gauge chart")

    with col2:
        # Create a comparison to similar queries
        comparable_queries = {
            "Avg. Similar Queries": {
                "slot_ms": prediction["predicted_slot_ms"] * 1.2,
                "execution_ms": prediction["predicted_execution_ms"] * 1.3,
                "cost_usd": prediction["estimated_cost_usd"] * 1.2
            },
            "Best Practice": {
                "slot_ms": prediction["predicted_slot_ms"] * 0.6,
                "execution_ms": prediction["predicted_execution_ms"] * 0.7,
                "cost_usd": prediction["estimated_cost_usd"] * 0.6
            },
            "Your Query": {
                "slot_ms": prediction["predicted_slot_ms"],
                "execution_ms": prediction["predicted_execution_ms"],
                "cost_usd": prediction["estimated_cost_usd"]
            }
        }

        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparable_queries).T
        comparison_df["slot_ms"] = comparison_df["slot_ms"].apply(lambda x: format_number(x, is_time=True))
        comparison_df["execution_ms"] = comparison_df["execution_ms"].apply(lambda x: format_number(x, is_time=True))
        comparison_df["cost_usd"] = comparison_df["cost_usd"].apply(lambda x: f"${x:.6f}")

        # Display comparison table
        st.markdown("### Comparison with Similar Queries")
        st.table(comparison_df)
        st.caption("Note: Comparison is based on estimated similar query patterns")
        logger.debug("Displayed query comparison table")

    # Recommendations
    st.subheader("Optimization Recommendations")
    logger.debug(f"Displaying {len(prediction['recommendations'])} optimization recommendations")

    if prediction["recommendations"]:
        for rec in prediction["recommendations"]:
            impact_class = f"recommendation recommendation-{rec['impact'].lower()}"
            st.markdown(f"""
            <div class="{impact_class}">
                <strong>{rec['title']}</strong> ({rec['impact']} Impact)<br>
                {rec['description']}
            </div>
            """, unsafe_allow_html=True)
            logger.debug(f"Displayed recommendation: {rec['title']} ({rec['impact']} impact)")
    else:
        st.info("No specific optimization recommendations for this query. It appears to be efficient!")
        logger.debug("No recommendations to display")

    # Add a detailed cost breakdown section
    with st.expander("Detailed Cost Analysis"):
        # Calculate components of cost
        slot_cost = prediction["predicted_slot_ms"] * COST_PER_SLOT_MS
        bytes_cost_estimate = prediction["total_bytes_processed"] / 1e12 * 5  # $5 per TB
        logger.debug(f"Cost breakdown: slot_cost=${slot_cost:.6f}, bytes_cost=${bytes_cost_estimate:.6f}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cost Breakdown")
            st.markdown(f"**Slot Usage Cost:** ${slot_cost:.6f}")
            st.markdown(f"**Data Processed Cost (est.):** ${bytes_cost_estimate:.6f}")
            st.markdown(f"**Total Estimated Cost:** ${prediction['estimated_cost_usd']:.6f}")

            # Monthly projection
            monthly_queries = 100  # Assume 100 similar queries per month
            monthly_cost = prediction["estimated_cost_usd"] * monthly_queries
            st.markdown(f"**Monthly Projection (x{monthly_queries}):** ${monthly_cost:.2f}")
            logger.debug(f"Monthly cost projection (x{monthly_queries}): ${monthly_cost:.2f}")

        with col2:
            # Create pie chart for cost components
            fig = px.pie(
                values=[slot_cost, bytes_cost_estimate],
                names=['Slot Usage', 'Data Processing'],
                title='Cost Components'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig)

    logger.info("Completed displaying prediction results")


# App Initialization
logger.debug("Setting up app custom CSS")

# Custom CSS with improved styling
st.markdown("""
<style>
    .main-header {color:#1E88E5; font-size:40px; font-weight:bold; margin-bottom:0px;}
    .sub-header {color:#424242; font-size:20px; margin-top:0px;}
    .metric-card {background-color:#f5f5f5; border-radius:8px; padding:15px; text-align:center; 
                  box-shadow: 2px 2px 5px rgba(0,0,0,0.1); height:100px; margin-bottom:10px;}
    .metric-value {font-size:28px; font-weight:bold; color:#1E88E5; margin-bottom:5px;}
    .metric-label {font-size:16px; color:#616161;}
    .recommendation {background-color:#f5f7ff; border-left:4px solid #4285F4; padding:15px; margin:15px 0; border-radius:4px;}
    .recommendation-high {border-left:4px solid #EA4335; background-color:#ffefef;}
    .recommendation-medium {border-left:4px solid #FBBC05; background-color:#fff9e6;}
    .recommendation-low {border-left:4px solid #34A853; background-color:#f0fff0;}
    .code-box {background-color:#f5f5f5; padding:15px; border-radius:5px; font-family:monospace; margin:10px 0; overflow:auto;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0px 0px;}
    .sql-file-container {border:1px solid #e0e0e0; border-radius:5px; padding:10px; margin:10px 0;}
    .sql-file-header {background-color:#f0f0f0; padding:8px; border-radius:4px; margin-bottom:8px; font-weight:bold;}
    .batch-results {margin-top:20px; padding-top:15px;}
    .batch-item {margin-bottom:20px; padding:15px; border-radius:8px; background-color:#f9f9f9;}
    .comparison-table {width:100%; border-collapse:collapse;}
    .comparison-table th {background-color:#f0f0f0; padding:8px; text-align:left;}
    .comparison-table td {padding:8px; border-bottom:1px solid #ddd;}
    .stButton button {padding:10px 20px; font-size:16px;}

    /* Enhanced styles for UI */
    .feature-pill {background-color:#e0f7fa; padding:5px 10px; border-radius:15px; font-size:12px; margin:3px;}
    .warning-box {background-color:#fff3cd; border-left:4px solid #ffc107; padding:10px; margin:10px 0; border-radius:4px;}
    .success-box {background-color:#d4edda; border-left:4px solid #28a745; padding:10px; margin:10px 0; border-radius:4px;}
    .info-box {background-color:#cce5ff; border-left:4px solid #007bff; padding:10px; margin:10px 0; border-radius:4px;}
    .footer {text-align:center; margin-top:30px; padding-top:20px; border-top:1px solid #eee; font-size:12px; color:#666;}
    .metric-trend-up {color: #28a745;}
    .metric-trend-down {color: #dc3545;}
    .table-stats {width: 100%; border-collapse: collapse; margin: 15px 0;}
    .table-stats th {background-color: #f8f9fa; text-align: left; padding: 8px; border: 1px solid #dee2e6;}
    .table-stats td {padding: 8px; border: 1px solid #dee2e6;}
    .highlight-row {background-color: #f2f9ff;}
    .ai-badge {background-color: #9c27b0; color: white; padding: 3px 8px; border-radius: 12px; font-size: 11px;}
</style>
""", unsafe_allow_html=True)

# App header with new design
logger.debug("Setting up app header")
st.markdown('<p class="main-header">PipelineIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BigQuery Resource Intelligence & Optimization</p>', unsafe_allow_html=True)

# Create enhanced tabs with icons
logger.debug("Creating main application tabs")
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Query Analyzer",
    "📊 Resource Insights",
    "💰 Cost Optimization",
    "🔍 Advanced Tools"
])

# Tab 1: Query Analyzer
with tab1:
    logger.info("User navigated to Query Analyzer tab")
    st.header("BigQuery Resource Analyzer")
    st.write("Predict resources, execution time, and cost before running your queries")

    # Initialize session state variables
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ""
        logger.debug("Initialized empty query_text in session state")
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "Manual"
        logger.debug("Initialized input_method in session state to 'Manual'")
    if 'selected_zip' not in st.session_state:
        st.session_state.selected_zip = None
        logger.debug("Initialized selected_zip in session state to None")
    if 'selected_sql_file' not in st.session_state:
        st.session_state.selected_sql_file = None
        logger.debug("Initialized selected_sql_file in session state to None")
    if 'sql_files' not in st.session_state:
        st.session_state.sql_files = {}
        logger.debug("Initialized empty sql_files dict in session state")
    if 'batch_mode' not in st.session_state:
        st.session_state.batch_mode = False
        logger.debug("Initialized batch_mode in session state to False")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "Upload SQL File(s)"],
        horizontal=True,
        key="input_method_radio"
    )
    st.session_state.input_method = input_method
    logger.debug(f"User selected input method: {input_method}")

    # Based on selected input method, show appropriate UI
    if st.session_state.input_method == "Manual Entry":
        logger.debug("Showing Manual Entry UI")
        # Sample queries in a more compact expander
        with st.expander("📋 Sample queries (click to use)"):
            sample_queries = {
                "Simple SELECT": """
                    SELECT * 
                    FROM `project.dataset.table` 
                    WHERE date = '2025-01-01'
                    LIMIT 1000
                """,
                "JOIN with GROUP BY": """
                    SELECT 
                        users.name,
                        COUNT(*) as order_count,
                        SUM(orders.amount) as total_spent
                    FROM `project.dataset.users` users
                    JOIN `project.dataset.orders` orders
                      ON users.id = orders.user_id
                    WHERE orders.status = 'completed'
                      AND orders.date BETWEEN '2025-01-01' AND '2025-03-31'
                    GROUP BY users.name
                    ORDER BY total_spent DESC
                    LIMIT 100
                """,
                "Complex Analytical": """
                    WITH revenue_data AS (
                        SELECT 
                            user_id,
                            EXTRACT(MONTH FROM transaction_date) as month,
                            SUM(amount) as monthly_revenue
                        FROM `project.dataset.transactions`
                        WHERE EXTRACT(YEAR FROM transaction_date) = 2025
                        GROUP BY user_id, month
                    ),
                    user_segments AS (
                        SELECT
                            user_id,
                            CASE 
                                WHEN MAX(monthly_revenue) > 1000 THEN 'high_value'
                                WHEN MAX(monthly_revenue) > 500 THEN 'medium_value'
                                ELSE 'low_value'
                            END as segment
                        FROM revenue_data
                        GROUP BY user_id
                    )
                    SELECT
                        us.segment,
                        COUNT(DISTINCT rd.user_id) as user_count,
                        AVG(rd.monthly_revenue) as avg_monthly_revenue,
                        SUM(rd.monthly_revenue) as total_revenue
                    FROM revenue_data rd
                    JOIN user_segments us ON rd.user_id = us.user_id
                    GROUP BY us.segment
                    ORDER BY total_revenue DESC
                """
            }

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Simple SELECT", key="sample1"):
                    logger.info("User selected 'Simple SELECT' sample query")
                    st.session_state.query_text = sample_queries["Simple SELECT"].strip()
            with col2:
                if st.button("JOIN with GROUP BY", key="sample2"):
                    logger.info("User selected 'JOIN with GROUP BY' sample query")
                    st.session_state.query_text = sample_queries["JOIN with GROUP BY"].strip()
            with col3:
                if st.button("Complex Analytical", key="sample3"):
                    logger.info("User selected 'Complex Analytical' sample query")
                    st.session_state.query_text = sample_queries["Complex Analytical"].strip()

        # Enhanced query input with syntax highlighting
        query_text = st.text_area(
            "Enter your SQL query:",
            value=st.session_state.query_text,
            height=200,
            key="query_input"
        )
        if query_text != st.session_state.query_text:
            logger.debug("User updated query text in text area")
        st.session_state.query_text = query_text

        # More prominent analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🚀 Analyze Query", type="primary", use_container_width=True)

        # Show prediction results for manual entry
        if analyze_button and st.session_state.query_text:
            logger.info("User clicked Analyze Query button")
            with st.spinner("⚙️ Analyzing query and predicting resources..."):
                logger.debug("Starting query analysis")
                start_time = time.time()
                prediction, bytes_message, error = analyze_query(st.session_state.query_text)
                analysis_time = time.time() - start_time
                logger.debug(f"Query analysis completed in {analysis_time:.2f} seconds")

                if error:
                    logger.error(f"Analysis error: {error}")
                    st.error(f"🔴 {error}")
                else:
                    logger.info("Successfully analyzed query, displaying results")
                    display_prediction_results(prediction, bytes_message)

                    # Query highlighting
                    st.subheader("Query Analysis")
                    st.code(query_text, language="sql")
                    logger.debug("Displayed query syntax highlighting")

    else:  # File Upload method
        logger.debug("Showing File Upload UI")
        uploaded_files = st.file_uploader(
            "Upload SQL files",
            type=["sql", "zip"],
            accept_multiple_files=True,
            help="Upload .sql files directly or zip archives containing SQL files"
        )

        if uploaded_files:
            logger.info(f"User uploaded {len(uploaded_files)} files")
            # Process uploaded files
            sql_files = {}

            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.sql'):
                    # Directly process SQL file
                    logger.info(f"Processing SQL file: {uploaded_file.name}")
                    sql_content = uploaded_file.read().decode('utf-8')
                    sql_files[uploaded_file.name] = sql_content
                    logger.debug(f"Added SQL file {uploaded_file.name} ({len(sql_content)} chars)")
                elif uploaded_file.name.endswith('.zip'):
                    # Extract SQL files from ZIP
                    logger.info(f"Processing ZIP file: {uploaded_file.name}")
                    zip_sql_files, error = extract_sql_files_from_zip(uploaded_file)

                    if error:
                        logger.error(f"ZIP processing error: {error}")
                        st.error(error)
                    else:
                        logger.info(f"Extracted {len(zip_sql_files)} SQL files from {uploaded_file.name}")
                        sql_files.update(zip_sql_files)

            if not sql_files:
                logger.warning("No SQL files found in the uploaded files")
                st.warning("No SQL files found in the uploaded files.")
            else:
                logger.info(f"Successfully processed {len(sql_files)} SQL files")
                st.session_state.sql_files = sql_files

                # Offer batch or single analysis
                st.markdown("### Analysis Mode")
                batch_mode = st.checkbox("Batch analyze all SQL files", value=st.session_state.batch_mode)
                if batch_mode != st.session_state.batch_mode:
                    logger.debug(f"User changed batch mode to: {batch_mode}")
                st.session_state.batch_mode = batch_mode

                if batch_mode:
                    # Batch analysis mode
                    logger.debug("Showing batch analysis UI")
                    st.success(f"Found {len(sql_files)} SQL files")

                    if st.button("📊 Analyze All SQL Files", type="primary"):
                        logger.info(f"User initiated batch analysis of {len(sql_files)} files")
                        # Show progress bar
                        progress_bar = st.progress(0)

                        # Container for batch results
                        batch_results_container = st.container()

                        # Summary metrics
                        total_slot_ms = 0
                        total_execution_ms = 0
                        total_cost = 0
                        high_impact_count = 0
                        most_expensive_file = {'name': '', 'cost': 0}
                        slowest_file = {'name': '', 'time': 0}

                        with batch_results_container:
                            st.markdown("## Batch Analysis Results")

                            # Process each SQL file
                            batch_start_time = time.time()
                            batch_results = {}

                            for i, (file_name, query_text) in enumerate(sql_files.items()):
                                with st.spinner(f"Analyzing {file_name}... ({i + 1}/{len(sql_files)})"):
                                    # Update progress
                                    progress = (i + 1) / len(sql_files)
                                    progress_bar.progress(progress)
                                    logger.info(f"Batch progress: {progress * 100:.1f}% - Analyzing {file_name}")

                                    # Analyze the query
                                    file_start_time = time.time()
                                    prediction, bytes_message, error = analyze_query(query_text)
                                    file_analysis_time = time.time() - file_start_time
                                    logger.debug(f"Analyzed file {file_name} in {file_analysis_time:.2f} seconds")

                                    # Store result
                                    if not error:
                                        batch_results[file_name] = prediction

                                    # Display batch item result
                                    with st.expander(f"📄 {file_name}"):
                                        if error:
                                            logger.error(f"Error analyzing {file_name}: {error}")
                                            st.error(error)
                                        else:
                                            logger.debug(f"Successfully analyzed {file_name}")
                                            # Update summary metrics
                                            total_slot_ms += prediction['predicted_slot_ms']
                                            total_execution_ms += prediction['predicted_execution_ms']
                                            total_cost += prediction['estimated_cost_usd']

                                            # Track most expensive and slowest
                                            if prediction['estimated_cost_usd'] > most_expensive_file['cost']:
                                                most_expensive_file = {'name': file_name,
                                                                       'cost': prediction['estimated_cost_usd']}

                                            if prediction['predicted_execution_ms'] > slowest_file['time']:
                                                slowest_file = {'name': file_name,
                                                                'time': prediction['predicted_execution_ms']}

                                            # Count high impact recommendations
                                            high_impact_recs = [r for r in prediction.get('recommendations', []) if
                                                                r['impact'] == 'HIGH']
                                            high_impact_count += len(high_impact_recs)
                                            if high_impact_recs:
                                                logger.info(
                                                    f"Found {len(high_impact_recs)} high impact recommendations in {file_name}")

                                            # Show metrics
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Slot Time",
                                                          format_number(prediction['predicted_slot_ms'], is_time=True))
                                            with col2:
                                                st.metric("Execution Time",
                                                          format_number(prediction['predicted_execution_ms'],
                                                                        is_time=True))
                                            with col3:
                                                st.metric("Cost", f"${prediction['estimated_cost_usd']:.6f}")

                                            # Show recommendations if any
                                            if prediction["recommendations"]:
                                                st.markdown("**Recommendations:**")
                                                for rec in prediction["recommendations"]:
                                                    impact_color = {
                                                        'HIGH': '🔴',
                                                        'MEDIUM': '🟠',
                                                        'LOW': '🟢'
                                                    }.get(rec['impact'], '⚪')
                                                    st.markdown(
                                                        f"{impact_color} **{rec['title']}** ({rec['impact']} Impact)")

                                            # Show query snippet
                                            if st.checkbox("View Query", key=f"view_query_{file_name}"):
                                                st.code(query_text, language="sql")

                            total_batch_time = time.time() - batch_start_time
                            logger.info(
                                f"Completed batch analysis of {len(sql_files)} files in {total_batch_time:.2f} seconds")
                            logger.info(
                                f"Batch summary: total_slot_ms={total_slot_ms}, total_execution_ms={total_execution_ms}, " +
                                f"total_cost=${total_cost:.4f}, high_impact_recommendations={high_impact_count}")

                            # Show batch summary
                            st.markdown("## Batch Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Queries", len(sql_files))
                            with col2:
                                st.metric("Total Slot Time", format_number(total_slot_ms, is_time=True))
                            with col3:
                                st.metric("Total Execution Time", format_number(total_execution_ms, is_time=True))
                            with col4:
                                st.metric("Total Cost", f"${total_cost:.4f}")

                            # Key insights
                            st.markdown("### Key Insights")

                            insights_col1, insights_col2 = st.columns(2)

                            with insights_col1:
                                st.markdown(f"""
                                - Most expensive query: **{most_expensive_file['name']}** (${most_expensive_file['cost']:.6f})
                                - Slowest query: **{slowest_file['name']}** ({format_number(slowest_file['time'], is_time=True)})
                                - Found {high_impact_count} high-impact optimization opportunities
                                """)

                            with insights_col2:
                                # Create a pie chart of cost distribution
                                if len(batch_results) > 0:
                                    cost_data = {
                                        'query': [k for k in batch_results.keys()],
                                        'cost': [v['estimated_cost_usd'] for v in batch_results.values()]
                                    }
                                    cost_df = pd.DataFrame(cost_data)

                                    # Only show top 5 and aggregate the rest
                                    if len(cost_df) > 5:
                                        top5 = cost_df.nlargest(5, 'cost')
                                        other_cost = cost_df.iloc[5:]['cost'].sum()
                                        top5 = pd.concat([
                                            top5,
                                            pd.DataFrame({'query': ['Other queries'], 'cost': [other_cost]})
                                        ]).reset_index(drop=True)
                                        cost_df = top5

                                    cost_fig = px.pie(
                                        cost_df,
                                        values='cost',
                                        names='query',
                                        title='Cost Distribution'
                                    )
                                    cost_fig.update_layout(height=300)
                                    st.plotly_chart(cost_fig, use_container_width=True)

                            # Add summary recommendations
                            if high_impact_count > 0:
                                warning_msg = f"Found {high_impact_count} high-impact optimization opportunities across your queries!"
                                st.warning(warning_msg)
                                logger.info(warning_msg)

                                # Show all high impact recommendations
                                st.markdown("### Critical Optimization Opportunities")

                                all_high_impact = []
                                for file_name, result in batch_results.items():
                                    for rec in result.get('recommendations', []):
                                        if rec['impact'] == 'HIGH':
                                            all_high_impact.append({
                                                'file': file_name,
                                                'title': rec['title'],
                                                'description': rec['description']
                                            })

                                for idx, rec in enumerate(all_high_impact):
                                    st.markdown(f"""
                                    <div class="recommendation recommendation-high">
                                        <strong>{rec['title']}</strong> (in {rec['file']})<br>
                                        {rec['description']}
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Add download option for batch results
                            st.markdown("### 💾 Download Batch Results")

                            # Create a download button for the batch results
                            logger.debug("Preparing batch results for download")
                            batch_results_list = []
                            for file_name, prediction in batch_results.items():
                                recommendations = "; ".join(
                                    [f"{r['title']} ({r['impact']})" for r in prediction.get('recommendations', [])])
                                batch_results_list.append({
                                    'file_name': file_name,
                                    'slot_ms': prediction.get('predicted_slot_ms', 0),
                                    'execution_ms': prediction.get('predicted_execution_ms', 0),
                                    'execution_sec': prediction.get('predicted_execution_ms', 0) / 1000,
                                    'cost_usd': prediction.get('estimated_cost_usd', 0),
                                    'bytes_processed': prediction.get('total_bytes_processed', 0),
                                    'query_category': prediction.get('query_category', ''),
                                    'recommendations': recommendations
                                })

                            results_df = pd.DataFrame(batch_results_list)
                            logger.debug(f"Created results dataframe with {len(results_df)} rows")

                            # Create download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results CSV",
                                data=csv,
                                file_name="query_analysis_results.csv",
                                mime="text/csv"
                            )
                            logger.info("Added download button for batch results")

                else:
                    # Single query analysis mode
                    logger.debug("Showing single file analysis UI")
                    # Step 2: Let the user select which SQL file to analyze
                    selected_sql_file = st.selectbox(
                        "Select SQL file to analyze:",
                        list(sql_files.keys()),
                        key="sql_file_selector"
                    )
                    if selected_sql_file != st.session_state.selected_sql_file:
                        logger.info(f"User selected SQL file: {selected_sql_file}")
                    st.session_state.selected_sql_file = selected_sql_file

                    # Display selected SQL query
                    if st.session_state.selected_sql_file:
                        selected_query = sql_files[st.session_state.selected_sql_file]
                        logger.debug(
                            f"Displaying selected SQL file content: {st.session_state.selected_sql_file} ({len(selected_query)} chars)")

                        st.markdown("### SQL Query Content:")
                        st.code(selected_query, language="sql")

                        # Analyze button
                        if st.button("🚀 Analyze Selected SQL Query", type="primary"):
                            logger.info(
                                f"User clicked Analyze Selected SQL Query button for {st.session_state.selected_sql_file}")
                            with st.spinner(f"Analyzing {st.session_state.selected_sql_file}..."):
                                # Analyze the query
                                start_time = time.time()
                                prediction, bytes_message, error = analyze_query(selected_query)
                                analysis_time = time.time() - start_time
                                logger.debug(f"Analysis completed in {analysis_time:.2f} seconds")

                                if error:
                                    logger.error(f"Error analyzing {st.session_state.selected_sql_file}: {error}")
                                    st.error(error)
                                else:
                                    logger.info(f"Successfully analyzed {st.session_state.selected_sql_file}")
                                    display_prediction_results(prediction, bytes_message)

# Tab 2: Resource Insights
with tab2:
    logger.info("User navigated to Resource Insights tab")
    st.header("Resource Usage Insights")
    st.write("Analyze historical query patterns and resource usage trends")

    # Load historical data
    logger.debug("Loading historical metrics and recent queries data")
    historical_data = load_historical_metrics()
    recent_queries = load_recent_queries(20)

    # Date filter
    col1, col2 = st.columns([1, 3])
    with col1:
        period = st.selectbox("Time period:", ["Last 7 days", "Last 30 days", "Last 90 days"])
        logger.debug(f"User selected time period: {period}")

    # Convert period to days
    days = 7
    if period == "Last 30 days":
        days = 30
    elif period == "Last 90 days":
        days = 90

    logger.info(historical_data)

    if not pd.api.types.is_datetime64_any_dtype(historical_data['date']):
        historical_data['date'] = pd.to_datetime(historical_data['date'])

    # Then filter using datetime64 objects
    comparison_date = pd.Timestamp(datetime.now() - timedelta(days=days))
    filtered_data = historical_data[historical_data['date'] > comparison_date]
    logger.debug(f"Filtered historical data to {len(filtered_data)} records over {days} days")

    logger.info(filtered_data)

    # Executive summary with KPIs
    st.subheader("Resource Usage Summary")
    logger.debug("Calculating period-over-period changes")

    # Calculate period-over-period changes
    if days > 7:
        midpoint = pd.Timestamp(datetime.now() - timedelta(days=days // 2))
        current_period = filtered_data[filtered_data['date'] > midpoint]
        previous_period = filtered_data[filtered_data['date'] <= midpoint]

        # Calculate deltas
        query_delta = ((current_period['total_queries'].mean() / max(previous_period['total_queries'].mean(),
                                                                     1)) - 1) * 100
        slot_delta = ((current_period['avg_slot_ms'].mean() / max(previous_period['avg_slot_ms'].mean(), 1)) - 1) * 100
        exec_delta = ((current_period['avg_execution_ms'].mean() / max(previous_period['avg_execution_ms'].mean(),
                                                                       1)) - 1) * 100
        data_delta = ((current_period['total_bytes_processed'].mean() / max(
            previous_period['total_bytes_processed'].mean(), 1)) - 1) * 100

        logger.debug(f"Period-over-period changes: query_delta={query_delta:.1f}%, " +
                     f"slot_delta={slot_delta:.1f}%, exec_delta={exec_delta:.1f}%, data_delta={data_delta:.1f}%")
    else:
        query_delta = None
        slot_delta = None
        exec_delta = None
        data_delta = None
        logger.debug("Period too short for period-over-period comparison")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_queries = int(filtered_data['total_queries'].mean())
        st.metric("Avg Queries/Day", format_number(avg_queries),
                  f"{query_delta:.1f}%" if query_delta is not None else None)

    with col2:
        avg_slots = filtered_data['avg_slot_ms'].mean()
        st.metric("Avg Slot Time", format_number(avg_slots, is_time=True),
                  f"{slot_delta:.1f}%" if slot_delta is not None else None)

    with col3:
        avg_execution = filtered_data['avg_execution_ms'].mean()
        st.metric("Avg Execution Time", format_number(avg_execution, is_time=True),
                  f"{exec_delta:.1f}%" if exec_delta is not None else None)

    with col4:
        total_data = filtered_data['total_bytes_processed'].sum()
        st.metric("Data Processed", format_number(total_data, is_bytes=True),
                  f"{data_delta:.1f}%" if data_delta is not None else None)

    logger.debug("Displayed resource usage summary metrics")

    # Data usage forecast
    st.subheader("Data Usage Forecast")

    # Attempt to predict next month's usage based on trend
    if len(filtered_data) > 14:  # Need enough data for trend
        try:
            # Simple linear regression for forecasting
            data_for_forecast = filtered_data.copy()
            data_for_forecast['day_number'] = range(len(data_for_forecast))

            # Calculate the trend line
            x = data_for_forecast['day_number'].values
            y = data_for_forecast['total_bytes_processed'].values

            # Add a small constant to avoid zero values
            y = y + 1

            # Calculate slope and intercept
            n = len(x)
            slope = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x ** 2) - sum(x) ** 2)
            intercept = (sum(y) - slope * sum(x)) / n

            # Create forecast dataframe
            last_day = data_for_forecast['day_number'].max()
            forecast_days = 30  # Forecast 30 days
            forecast_days_range = range(last_day + 1, last_day + forecast_days + 1)

            forecast_df = pd.DataFrame({
                'day_number': forecast_days_range,
                'forecast': [intercept + slope * x for x in forecast_days_range]
            })

            # Ensure forecast values are positive
            forecast_df['forecast'] = forecast_df['forecast'].apply(lambda x: max(x, 0))

            # Calculate the forecast for next month
            next_month_bytes = forecast_df['forecast'].sum()
            next_month_tb = next_month_bytes / 1e12

            # Calculate the current month's usage for comparison
            current_month_bytes = filtered_data['total_bytes_processed'].sum()
            current_month_tb = current_month_bytes / 1e12

            # Calculate the change as a percentage
            if current_month_tb > 0:
                change_pct = ((next_month_tb / current_month_tb) - 1) * 100
            else:
                change_pct = 0

            # Display the forecast
            forecast_col1, forecast_col2 = st.columns([3, 2])

            with forecast_col1:
                # Create a line chart with actual and forecasted data
                combined_df = pd.concat([
                    pd.DataFrame({
                        'day_number': data_for_forecast['day_number'],
                        'value': data_for_forecast['total_bytes_processed'],
                        'type': 'Actual'
                    }),
                    pd.DataFrame({
                        'day_number': forecast_df['day_number'],
                        'value': forecast_df['forecast'],
                        'type': 'Forecast'
                    })
                ])

                forecast_fig = px.line(
                    combined_df,
                    x='day_number',
                    y='value',
                    color='type',
                    title='Data Usage Forecast (Bytes Processed)',
                    labels={'value': 'Bytes Processed', 'day_number': 'Day'}
                )

                # Add a vertical line at the current day
                forecast_fig.add_vline(
                    x=last_day,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Today"
                )

                forecast_fig.update_layout(height=300)
                st.plotly_chart(forecast_fig, use_container_width=True)

            with forecast_col2:
                st.markdown("### Projected Monthly Usage")

                # Projected TB usage
                st.markdown(f"""
                **Current Period:** {format_bytes(current_month_bytes)}

                **Next 30 Days (Forecast):** {format_bytes(next_month_bytes)}

                **Change:** {change_pct:+.1f}%

                **Estimated Monthly Cost:** ${next_month_tb * 5:.2f}
                """)

                # Add a warning if growth is high
                if change_pct > 20:
                    st.warning(
                        f"⚠️ Projected data usage growth of {change_pct:.1f}% is significant. Consider implementing data reduction strategies.")
                elif change_pct < -20:
                    st.success(
                        f"✅ Projected data usage reduction of {abs(change_pct):.1f}% shows good optimization progress.")

        except Exception as e:
            logger.error(f"Error creating forecast: {str(e)}")
            st.error("Could not create forecast due to insufficient data or calculation error.")

    # Enhanced time series charts
    st.subheader("Resource Usage Trends")

    chart_type = st.radio("Metric:", ["Slot Usage", "Query Volume", "Execution Time", "Data Processed"],
                          horizontal=True)
    logger.debug(f"User selected chart type: {chart_type}")

    if chart_type == "Slot Usage":
        y_col = 'avg_slot_ms'
        title = 'Average Slot Time (ms) per Day'
        color = '#1E88E5'
    elif chart_type == "Query Volume":
        y_col = 'total_queries'
        title = 'Total Queries per Day'
        color = '#43A047'
    elif chart_type == "Execution Time":
        y_col = 'avg_execution_ms'
        title = 'Average Execution Time (ms) per Day'
        color = '#FB8C00'
    else:  # Data Processed
        y_col = 'total_bytes_processed'
        title = 'Total Bytes Processed per Day'
        color = '#E53935'

    # Add a 7-day moving average to the chart
    filtered_data = filtered_data.copy()  # Create an explicit copy
    filtered_data['moving_avg'] = filtered_data[y_col].rolling(7, min_periods=1).mean()
    logger.debug(f"Calculated 7-day moving average for {y_col}")

    # Create Altair chart with moving average
    logger.debug("Creating time series chart with Altair")
    base = alt.Chart(filtered_data).encode(x=alt.X('date:T', title='Date'))

    line = base.mark_line(color=color).encode(
        y=alt.Y(f'{y_col}:Q', title=title),
        tooltip=['date:T', f'{y_col}:Q']
    )

    moving_avg = base.mark_line(color='red', strokeDash=[5, 5]).encode(
        y=alt.Y('moving_avg:Q', title=f'7-day Moving Avg ({title})'),
        tooltip=['date:T', 'moving_avg:Q']
    )

    points = base.mark_point(color=color, size=60).encode(
        y=alt.Y(f'{y_col}:Q'),
        opacity=alt.value(0.5),
        tooltip=['date:T', f'{y_col}:Q']
    )

    chart = (line + moving_avg + points).properties(height=400)

    st.altair_chart(chart, use_container_width=True)
    logger.debug("Displayed time series chart")

    # Add visualization options
    show_weekly = st.checkbox("Show Weekly Patterns")
    logger.debug(f"Show weekly patterns checkbox: {show_weekly}")

    if show_weekly:
        # Weekly patterns visualization
        st.subheader("Weekly Usage Patterns")
        logger.debug("Creating weekly usage patterns visualization")

        # Convert day of week to string names
        day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                       6: 'Sunday'}
        historical_data['day_name'] = historical_data['day_of_week'].map(day_mapping)

        # Group by day of week and calculate averages
        weekly_pattern = historical_data.groupby('day_name').agg({
            'avg_slot_ms': 'mean',
            'total_queries': 'mean',
            'avg_execution_ms': 'mean',
            'total_bytes_processed': 'mean'
        }).reset_index()
        logger.debug("Calculated weekly pattern aggregations")

        # Reorder days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern['day_name'] = pd.Categorical(weekly_pattern['day_name'], categories=day_order, ordered=True)
        weekly_pattern = weekly_pattern.sort_values('day_name')

        # Create bar chart for weekly patterns
        fig = px.bar(
            weekly_pattern,
            x='day_name',
            y=y_col,
            title=f'Average {title} by Day of Week',
            labels={'day_name': 'Day of Week', y_col: title},
            color_discrete_sequence=[color]
        )

        st.plotly_chart(fig, use_container_width=True)
        logger.debug("Displayed weekly patterns bar chart")

        # Add weekly recommendations
        st.markdown("### Workload Distribution Recommendations")

        # Find the busiest and least busy days
        busiest_day = weekly_pattern.loc[weekly_pattern[y_col].idxmax()]
        least_busy_day = weekly_pattern.loc[weekly_pattern[y_col].idxmin()]

        busiest_to_least_ratio = busiest_day[y_col] / max(least_busy_day[y_col], 1)

        if busiest_to_least_ratio > 2:  # Significant imbalance
            st.warning(f"""
            **Workload Imbalance Detected**: {busiest_day['day_name']} has {busiest_to_least_ratio:.1f}x higher usage than {least_busy_day['day_name']}.

            **Recommendation**: Consider rescheduling batch jobs from {busiest_day['day_name']} to {least_busy_day['day_name']} to balance resource usage 
            and improve overall query performance.
            """)
        else:
            st.success("Your workload is well-distributed across the week. Good job!")

    # Recent queries with improved display
    st.subheader("Recent Queries")
    logger.debug(f"Displaying {len(recent_queries)} recent queries")

    # Add performance insights derived from recent queries
    perf_col1, perf_col2 = st.columns([2, 1])

    with perf_col1:
        # Format the dataframe for display
        if len(recent_queries) > 0:
            recent_queries['execution_time'] = recent_queries['execution_ms'].apply(
                lambda x: format_number(x, is_time=True))
            recent_queries['slot_time'] = recent_queries['slot_ms'].apply(lambda x: format_number(x, is_time=True))
            recent_queries['data_processed'] = recent_queries['bytes_processed'].apply(
                lambda x: format_number(x, is_bytes=True))

            if 'cost_usd' in recent_queries.columns:
                recent_queries['cost'] = recent_queries['cost_usd'].apply(lambda x: f"${x:.6f}")
                display_cols = ['timestamp', 'query_snippet', 'execution_time', 'slot_time', 'data_processed', 'cost']
                display_names = {
                    'timestamp': 'Time',
                    'query_snippet': 'Query',
                    'execution_time': 'Execution Time',
                    'slot_time': 'Slot Time',
                    'data_processed': 'Data Processed',
                    'cost': 'Cost'
                }
            else:
                display_cols = ['timestamp', 'query_snippet', 'execution_time', 'slot_time', 'data_processed']
                display_names = {
                    'timestamp': 'Time',
                    'query_snippet': 'Query',
                    'execution_time': 'Execution Time',
                    'slot_time': 'Slot Time',
                    'data_processed': 'Data Processed'
                }

            st.dataframe(
                recent_queries[display_cols].rename(columns=display_names),
                use_container_width=True
            )
            logger.debug("Displayed recent queries table")
        else:
            st.info("No recent queries found in history.")
            logger.debug("No recent queries to display")

    with perf_col2:
        if len(recent_queries) > 0:
            st.markdown("### Recent Query Insights")

            # Calculate statistics
            avg_slot_ms = recent_queries['slot_ms'].mean()
            avg_exec_ms = recent_queries['execution_ms'].mean()
            avg_bytes = recent_queries['bytes_processed'].mean() if 'bytes_processed' in recent_queries.columns else 0

            # Identify query types by patterns
            recent_queries['query_type'] = recent_queries['query_snippet'].apply(lambda q:
                                                                                 'JOIN Query' if 'join' in q.lower() else
                                                                                 'Aggregation' if 'group by' in q.lower() else
                                                                                 'Complex' if 'with ' in q.lower() else
                                                                                 'Simple'
                                                                                 )

            # Count query types
            type_counts = recent_queries['query_type'].value_counts()

            # Display insights
            st.markdown(f"""
            **Recent Activity Overview:**
            - Average slot time: {format_number(avg_slot_ms, is_time=True)}
            - Average execution time: {format_number(avg_exec_ms, is_time=True)}
            - Average data processed: {format_number(avg_bytes, is_bytes=True)}

            **Query Type Distribution:**
            """)

            # Create small chart of query types
            if not type_counts.empty:
                type_fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title='Query Types'
                )
                type_fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=200)
                st.plotly_chart(type_fig, use_container_width=True)

# Tab 3: Cost Optimization
with tab3:
    logger.info("User navigated to Cost Optimization tab")
    st.header("Cost Optimization")
    st.write("Analyze and optimize your BigQuery costs")

    # Load historical data for optimization analysis
    logger.debug("Loading historical data for cost optimization analysis")
    historical_data = load_historical_metrics()

    # Slot utilization and commitment analysis
    st.subheader("Slot Utilization Analysis")

    # Get slot utilization stats from BigQuery
    slot_utilization = get_cached_slot_utilization()

    col1, col2 = st.columns(2)

    with col1:
        # Current slot usage analysis
        slot_usage = historical_data.tail(30)['avg_slot_ms'].mean() / 3600000  # Convert to slot hours
        peak_usage = historical_data.tail(30)['avg_slot_ms'].max() / 3600000
        logger.debug(
            f"Calculated slot usage metrics: avg={slot_usage:.2f} slot hours, peak={peak_usage:.2f} slot hours")

        # Create a slot usage gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=slot_utilization.get('avg_utilization', 0.5) * 100,  # Use real data
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Slot Utilization (%)"},
            delta={'reference': 70},  # Reference level for good utilization
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},  # Under-utilized
                    {'range': [30, 70], 'color': "lightgreen"},  # Good utilization
                    {'range': [70, 90], 'color': "yellow"},  # High utilization
                    {'range': [90, 100], 'color': "orange"}  # Over-utilized
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': slot_utilization.get('peak_utilization', 0.8) * 100  # Use real peak data
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        logger.debug("Displayed slot utilization gauge chart")

        # Display detailed metrics
        st.markdown(f"""
        **Utilization Metrics:**
        - Peak Utilization: {slot_utilization.get('peak_utilization', 0.8) * 100:.1f}%
        - Average Utilization: {slot_utilization.get('avg_utilization', 0.5) * 100:.1f}%
        - Average at 5K slots: {slot_utilization.get('avg_utilization_5k', 0.3) * 100:.1f}%
        - Average at 15K slots: {slot_utilization.get('avg_utilization_15k', 0.4) * 100:.1f}%
        - Average at 30K slots: {slot_utilization.get('avg_utilization_30k', 0.6) * 100:.1f}%
        """)

    with col2:
        # Slot commitment recommendations with interactive slider
        st.markdown("### Slot Commitment Recommendations")

        daily_queries = st.slider("Estimated daily queries:", min_value=10, max_value=1000, value=100, step=10)
        logger.debug(f"User set daily queries estimate to: {daily_queries}")

        # More intelligent slot calculation based on real utilization patterns
        avg_utilization = slot_utilization.get('avg_utilization', 0.5)
        peak_utilization = slot_utilization.get('peak_utilization', 0.8)

        # Calculate recommended slot commitments based on actual utilization data
        avg_slots_used = slot_usage * daily_queries
        peak_slots_used = peak_usage * daily_queries

        # Base commitment (60-70% of average usage)
        base_slots = int(avg_slots_used * 0.65)

        # Flex commitment (50-60% of peak minus base)
        flex_slots = int((peak_slots_used - base_slots) * 0.55)

        logger.debug(f"Calculated data-driven slot recommendations: base={base_slots}, flex={flex_slots}")

        # Ensure minimum values and round to nearest 100
        base_slots = max(100, round(base_slots / 100) * 100)
        flex_slots = max(100, round(flex_slots / 100) * 100)
        logger.debug(f"Rounded slot recommendations: base={base_slots}, flex={flex_slots}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Commitment", f"{base_slots} slots")
        with col2:
            st.metric("Flex Commitment", f"{flex_slots} slots")

        # Cost comparison with actual pricing
        on_demand_cost = (base_slots + flex_slots) * 24 * 30 * (COST_PER_SLOT_MS * 3600 * 1000)
        commitment_cost = (base_slots * 24 * 30 * (COST_PER_SLOT_MS * 3600 * 1000) * 0.6) + \
                          (flex_slots * 24 * 30 * (COST_PER_SLOT_MS * 3600 * 1000) * 0.8)
        savings = on_demand_cost - commitment_cost
        savings_pct = (savings / on_demand_cost) * 100
        logger.info(f"Calculated cost savings with commitments: ${savings:,.2f} ({savings_pct:.1f}%)")

        st.success(f"Estimated monthly savings with commitments: **${savings:,.2f}** ({savings_pct:.1f}%)")

        # Add ROI calculator
        with st.expander("📊 ROI Calculator"):
            st.markdown("### Return on Investment Analysis")
            logger.debug("Displaying ROI calculator")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Current (On-Demand)")
                st.markdown(f"Monthly Cost: **${on_demand_cost:,.2f}**")
                st.markdown(f"Annual Cost: **${on_demand_cost * 12:,.2f}**")

            with col2:
                st.markdown("#### With Commitments")
                st.markdown(f"Monthly Cost: **${commitment_cost:,.2f}**")
                st.markdown(f"Annual Cost: **${commitment_cost * 12:,.2f}**")
                st.markdown(f"Annual Savings: **${savings * 12:,.2f}**")

            # Calculate payback period for any upfront costs
            st.markdown("#### Payback Period Analysis")

            # Allow user to enter potential upfront costs
            upfront_cost = st.number_input("Upfront cost for implementation ($):", min_value=0, value=0)

            if upfront_cost > 0:
                payback_months = upfront_cost / savings if savings > 0 else float('inf')
                st.markdown(f"**Payback Period:** {payback_months:.1f} months")

                if payback_months < 3:
                    st.success("Very fast return on investment! Highly recommended.")
                elif payback_months < 6:
                    st.info("Good return on investment within 6 months.")
                elif payback_months < 12:
                    st.warning("Moderate return on investment within a year.")
                else:
                    st.error("Long payback period. Consider alternatives.")

    # Cost distribution analysis
    st.subheader("Query Cost Distribution")
    logger.debug("Creating query cost distribution visualization")

    # Get real cost distribution from BigQuery
    try:
        query_costs = get_query_cost_distribution()
        logger.debug(f"Retrieved {len(query_costs)} query costs for distribution")

        if len(query_costs) > 0:
            # Create cost statistics
            cost_stats = {
                'min': np.min(query_costs),
                'max': np.max(query_costs),
                'median': np.median(query_costs),
                'p90': np.percentile(query_costs, 90),
                'p95': np.percentile(query_costs, 95),
                'p99': np.percentile(query_costs, 99),
                'total': np.sum(query_costs)
            }

            # Create distribution visualization
            cost_col1, cost_col2 = st.columns([3, 2])

            with cost_col1:
                fig = px.histogram(
                    query_costs,
                    nbins=50,
                    labels={'value': 'Cost per Query ($)', 'count': 'Number of Queries'},
                    title='Distribution of Query Costs',
                    marginal='box'
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                logger.debug("Displayed query cost distribution histogram")

            with cost_col2:
                st.markdown("### Cost Analysis Insights")

                # Display cost statistics
                st.markdown(f"""
                **Query Cost Statistics:**
                - Median cost: ${cost_stats['median']:.6f}
                - 90th percentile: ${cost_stats['p90']:.6f}
                - 95th percentile: ${cost_stats['p95']:.6f}
                - 99th percentile: ${cost_stats['p99']:.6f}
                - Maximum cost: ${cost_stats['max']:.6f}
                - Total cost (sample): ${cost_stats['total']:.2f}
                """)

                # Cost-saving opportunity based on high-cost queries
                high_cost_threshold = cost_stats['p95']
                high_cost_queries = len([c for c in query_costs if c > high_cost_threshold])
                high_cost_total = sum([c for c in query_costs if c > high_cost_threshold])
                high_cost_pct = (high_cost_total / cost_stats['total']) * 100 if cost_stats['total'] > 0 else 0

                if high_cost_pct > 30:  # If 5% of queries account for >30% of cost
                    st.warning(f"""
                    **Cost Optimization Opportunity:**
                    The top 5% of queries account for **{high_cost_pct:.1f}%** of your total cost. 

                    Optimizing just these {high_cost_queries} queries could significantly reduce your overall costs.
                    """)

                    # Create a simple pie chart to show this
                    high_vs_rest = pd.DataFrame([
                        {'category': 'Top 5% queries', 'cost': high_cost_total},
                        {'category': 'All other queries', 'cost': cost_stats['total'] - high_cost_total}
                    ])

                    pie_fig = px.pie(
                        high_vs_rest,
                        values='cost',
                        names='category',
                        title='Cost Distribution by Query Percentile'
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No cost data available for distribution analysis.")

    except Exception as e:
        logger.error(f"Error creating cost distribution visualization: {str(e)}")
        st.error("Could not create cost distribution visualization due to data access issues.")

        # Fallback to sample data
        query_costs = np.random.exponential(scale=0.1, size=1000)
        query_costs = np.clip(query_costs, 0, 2)  # Cap at $2 per query
        logger.debug(f"Generated {len(query_costs)} sample query costs for distribution visualization")

        fig = px.histogram(
            query_costs,
            nbins=50,
            labels={'value': 'Cost per Query ($)', 'count': 'Number of Queries'},
            title='Distribution of Query Costs (SAMPLE DATA)',
            marginal='box'
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        logger.debug("Displayed sample query cost distribution histogram")

    # Cost-saving recommendations - Get data-driven recommendations
    st.subheader("Cost Optimization Recommendations")
    logger.debug("Displaying cost optimization recommendations")

    # Get data-driven recommendations
    recommendations = get_cached_cost_recommendations()
    logger.debug(f"Retrieved {len(recommendations)} cost optimization recommendations")

    # Display recommendations with enhanced UI
    for rec in recommendations:
        impact_class = f"recommendation recommendation-{rec['impact'].lower()}"

        # Calculate effort-to-value score (1-10)
        if rec['impact'] == 'HIGH' and rec['implementation_effort'] == 'LOW':
            score = 10
        elif rec['impact'] == 'HIGH' and rec['implementation_effort'] == 'MEDIUM':
            score = 8
        elif rec['impact'] == 'MEDIUM' and rec['implementation_effort'] == 'LOW':
            score = 7
        elif rec['impact'] == 'HIGH' and rec['implementation_effort'] == 'HIGH':
            score = 6
        elif rec['impact'] == 'MEDIUM' and rec['implementation_effort'] == 'MEDIUM':
            score = 5
        else:
            score = 3

        st.markdown(f"""
        <div class="{impact_class}">
            <strong>{rec['title']}</strong> ({rec['impact']} Impact / {rec['implementation_effort']} Effort)<br>
            {rec['description']}<br>
            <em>Estimated Benefit: {rec['savings_estimate']}</em><br>
            <strong>Implementation Priority Score: {score}/10</strong>
        </div>
        """, unsafe_allow_html=True)
        logger.debug(f"Displayed cost optimization recommendation: {rec['title']} (score: {score}/10)")

    # Add cost trends analysis
    st.subheader("Cost Trends Analysis")

    # Create cost trend visualization based on historical data
    if len(historical_data) > 14:  # Need enough data for trends
        # Calculate daily costs based on slot usage and bytes processed
        historical_data['daily_slot_cost'] = historical_data['avg_slot_ms'] * COST_PER_SLOT_MS * historical_data[
            'total_queries']
        historical_data['daily_bytes_cost'] = historical_data['total_bytes_processed'] / 1e12 * 5  # $5 per TB
        historical_data['daily_total_cost'] = historical_data['daily_slot_cost'] + historical_data['daily_bytes_cost']

        # Create rolling average for trend
        historical_data['cost_7day_avg'] = historical_data['daily_total_cost'].rolling(7, min_periods=1).mean()

        # Create the trend chart
        cost_trend_fig = px.line(
            historical_data,
            x='date',
            y=['daily_total_cost', 'cost_7day_avg'],
            title='Daily Cost Trend',
            labels={
                'daily_total_cost': 'Daily Cost ($)',
                'cost_7day_avg': '7-day Moving Average',
                'date': 'Date'
            }
        )

        # Update line colors and names
        cost_trend_fig.update_traces(
            line_color='#E53935',
            name='Daily Cost',
            selector=dict(name='daily_total_cost')
        )
        cost_trend_fig.update_traces(
            line_color='#1E88E5',
            line_width=3,
            name='7-day Moving Average',
            selector=dict(name='cost_7day_avg')
        )

        st.plotly_chart(cost_trend_fig, use_container_width=True)

        # Calculate month-to-date costs
        current_month = datetime.now().month
        current_year = datetime.now().year

        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(historical_data['date']):
            # Convert to datetime if it's not already
            historical_data['date'] = pd.to_datetime(historical_data['date'])

        mtd_data = historical_data[
            (historical_data['date'].dt.month == current_month) &
            (historical_data['date'].dt.year == current_year)
            ]

        if len(mtd_data) > 0:
            mtd_cost = mtd_data['daily_total_cost'].sum()

            # Estimate full month based on daily average
            daily_avg = mtd_data['daily_total_cost'].mean()
            days_in_month = pd.Timestamp(current_year, current_month, 1).days_in_month
            estimated_month_total = daily_avg * days_in_month

            # Display month-to-date stats
            st.markdown(f"""
            ### Month-to-Date Costs

            - MTD Cost: **${mtd_cost:.2f}**
            - Projected Monthly Total: **${estimated_month_total:.2f}**
            - Daily Average: **${daily_avg:.2f}**
            """)

# Tab 4: Advanced Tools
with tab4:
    logger.info("User navigated to Advanced Tools tab")
    st.header("Advanced Tools")
    st.write("Additional features to optimize your BigQuery experience")

    # Create subtabs
    logger.debug("Creating advanced tools subtabs")
    subtab1, subtab2, subtab3 = st.tabs([
        "Query Optimizer",
        "Schema Analyzer",
        "Benchmarking"
    ])

    # Query Optimizer tab
    with subtab1:
        logger.info("User navigated to Query Optimizer subtab")
        st.subheader("Query Optimizer")
        st.write("Get specific optimization suggestions for your queries")

        # Sample optimizer input
        query_to_optimize = st.text_area(
            "Enter a query to optimize:",
            height=150,
            value="""SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id WHERE orders.order_date > '2023-01-01'"""
        )

        optimize_button = st.button("🔧 Optimize Query", type="primary")

        if optimize_button and query_to_optimize:
            logger.info("User clicked Optimize Query button")
            with st.spinner("Analyzing query for optimization opportunities..."):
                try:
                    # Prepare the query information for an LLM
                    query_info = prepare_query_for_llm(config.PRPJECT_ID, query_to_optimize)

                    # Generate a formatted prompt for the LLM
                    prompt = generate_llm_prompt(query_info)

                    # Get AI-powered optimization
                    optimized_result = query_gemini_api(prompt)
                    logger.debug("Successfully generated query optimizations")

                    # Pattern to match content between ```sql and ``` markers
                    pattern = r"```sql\n(.*?)```"

                    # Find the SQL using re.DOTALL to match across multiple lines
                    match = re.search(pattern, optimized_result, re.DOTALL)

                    if match:
                        optimized_query = match.group(1).strip()

                        # Get performance estimates for both queries
                        original_perf = get_query_size(query_to_optimize)
                        optimized_perf = get_query_size(optimized_query)

                        # Calculate improvements
                        if original_perf['processed_bytes'] and optimized_perf['processed_bytes']:
                            bytes_improvement = (1 - (
                                        optimized_perf['processed_bytes'] / original_perf['processed_bytes'])) * 100
                        else:
                            bytes_improvement = None

                        # Display results with improved UI
                        st.success("✅ Query optimization complete!")

                        # Original vs Optimized queries side by side
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### Original Query")
                            st.code(query_to_optimize, language="sql")

                            if original_perf['processed_bytes']:
                                st.caption(f"Estimated data processed: {original_perf['formatted_size']}")

                        with col2:
                            st.markdown("### Optimized Query")
                            st.code(optimized_query, language="sql")

                            if optimized_perf['processed_bytes']:
                                st.caption(f"Estimated data processed: {optimized_perf['formatted_size']}")

                        # Explain the optimizations
                        st.markdown("### Optimization Explanations")

                        # Extract explanation text (everything except the code blocks)
                        explanation = re.sub(r'```sql.*?```', '', optimized_result, flags=re.DOTALL).strip()
                        st.markdown(explanation)

                        # Performance comparison
                        st.markdown("### Estimated Performance Improvement")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if bytes_improvement is not None:
                                st.metric("Data Processed", f"{bytes_improvement:.0f}% reduction",
                                          f"-{bytes_improvement:.0f}%")
                            else:
                                st.metric("Data Processed", "Unknown")

                        with col2:
                            # Use an estimated execution time improvement based on bytes processed
                            if bytes_improvement is not None:
                                exec_improvement = min(bytes_improvement * 0.8, 95)  # Conservative estimate
                                st.metric("Execution Time", f"{exec_improvement:.0f}% faster",
                                          f"-{exec_improvement:.0f}%")
                            else:
                                st.metric("Execution Time", "Unknown")

                        with col3:
                            if bytes_improvement is not None:
                                cost_improvement = min(bytes_improvement * 0.9, 95)  # Conservative estimate
                                st.metric("Cost", f"{cost_improvement:.0f}% savings", f"-{cost_improvement:.0f}%")
                            else:
                                st.metric("Cost", "Unknown")

                        # Add badge for AI-powered optimizations
                        st.markdown("""
                        <div style="text-align: right; margin-top: 20px;">
                            <span class="ai-badge">AI-POWERED OPTIMIZATION</span>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.error("Could not extract optimized query from the LLM response.")

                except Exception as e:
                    logger.error(f"Error optimizing query: {str(e)}")
                    st.error(f"Error optimizing query: {str(e)}")

    # Schema Analyzer tab
    with subtab2:
        logger.info("User navigated to Schema Analyzer subtab")
        st.subheader("Schema Analyzer")
        st.write("Analyze your table schemas for optimization opportunities")

        # Input form
        col1, col2 = st.columns(2)
        with col1:
            project_id = st.text_input("Google Cloud Project ID", value="my-project-id")
        with col2:
            dataset_id = st.text_input("BigQuery Dataset ID", value="my_dataset")

        logger.debug(f"User entered project_id: {project_id}, dataset_id: {dataset_id}")

        analyze_schema_button = st.button("🔍 Analyze Schema", type="primary")

        if analyze_schema_button:
            logger.info("User clicked Analyze Schema button")
            with st.spinner("Analyzing schema..."):
                try:
                    # Get real table information from BigQuery
                    tables_info = get_dataset_tables_info(project_id, dataset_id)

                    if not tables_info:
                        st.warning(f"No tables found in {project_id}.{dataset_id}")
                        logger.warning(f"No tables found in {project_id}.{dataset_id}")
                    else:
                        # Create tables DataFrame
                        tables_df = pd.DataFrame(tables_info)
                        logger.info(f"Retrieved information for {len(tables_df)} tables")

                        # Display available columns for debugging (can be removed in production)
                        logger.debug(f"Available columns in tables_df: {list(tables_df.columns)}")

                        # Show tables
                        st.markdown(f"### Tables in {project_id}.{dataset_id}")

                        # Determine which columns to display based on what's available
                        available_cols = []
                        desired_cols = ['table_name', 'row_count', 'num_rows', 'size_bytes', 'data_size',
                                        'partition_type', 'creation_time']

                        for col in desired_cols:
                            if col in tables_df.columns:
                                available_cols.append(col)

                        # Ensure we have at least the table name
                        if 'table_name' not in available_cols and len(tables_df.columns) > 0:
                            available_cols = [tables_df.columns[0]]  # Use first column as table identifier

                        # Display table information if columns are available
                        if available_cols:
                            st.dataframe(tables_df[available_cols], use_container_width=True)
                        else:
                            st.warning("Could not determine table column structure.")
                            st.write("Raw Data Sample:", tables_df.head(1))

                        # Calculate and add size in GB for visualization
                        try:
                            # Try different approaches to get size information
                            if 'size_bytes' in tables_df.columns:
                                tables_df['size_gb'] = pd.to_numeric(tables_df['size_bytes'], errors='coerce') / (
                                            1024 ** 3)
                            elif 'data_size' in tables_df.columns and isinstance(tables_df['data_size'].iloc[0], str):
                                # Try to extract numeric values from strings like "15.2 GB"
                                def extract_size_gb(size_str):
                                    try:
                                        if pd.isna(size_str) or not isinstance(size_str, str):
                                            return 0.001
                                        if 'TB' in size_str:
                                            return float(size_str.split()[0]) * 1024
                                        elif 'GB' in size_str:
                                            return float(size_str.split()[0])
                                        elif 'MB' in size_str:
                                            return float(size_str.split()[0]) / 1024
                                        elif 'KB' in size_str:
                                            return float(size_str.split()[0]) / (1024 ** 2)
                                        else:
                                            return 0.001
                                    except:
                                        return 0.001


                                tables_df['size_gb'] = tables_df['data_size'].apply(extract_size_gb)
                            else:
                                # Default small values if no size info available
                                tables_df['size_gb'] = 0.001

                            # Replace NaN values with a small default
                            tables_df['size_gb'] = tables_df['size_gb'].fillna(0.001)

                            # Display table size visualization
                            st.subheader("Table Size Visualization")

                            # Ensure size_gb is numeric
                            tables_df['size_gb'] = pd.to_numeric(tables_df['size_gb'], errors='coerce').fillna(0.001)

                            # Sort by size for better visualization
                            display_df = tables_df.sort_values('size_gb', ascending=True)

                            # Limit to top 30 tables if there are many
                            if len(display_df) > 30:
                                st.caption(f"Showing top 30 tables by size (out of {len(tables_df)} total)")
                                display_df = display_df.tail(30)  # Last 30 rows since we sorted ascending

                            # Create the bar chart visualization
                            fig = go.Figure()

                            # Add bars
                            fig.add_trace(go.Bar(
                                x=display_df['size_gb'],
                                y=display_df['table_name'],
                                orientation='h',
                                marker=dict(
                                    color='rgba(58, 71, 180, 0.6)',
                                    line=dict(color='rgba(58, 71, 180, 1.0)', width=1)
                                )
                            ))

                            # Update layout
                            fig.update_layout(
                                title='Table Sizes (GB)',
                                xaxis_title='Size (GB)',
                                yaxis_title='Table Name',
                                height=600,
                                margin=dict(l=200, r=20, t=40, b=40),
                                yaxis=dict(automargin=True),
                                xaxis_showgrid=True,
                                yaxis_showgrid=True,
                            )

                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)

                            # Calculate size statistics
                            total_size = tables_df['size_gb'].sum()
                            max_size = tables_df['size_gb'].max()

                            # Find the largest table
                            largest_table_name = "None"
                            if max_size > 0:
                                largest_idx = tables_df['size_gb'].idxmax()
                                largest_table_name = tables_df.loc[largest_idx, 'table_name']

                            # Display summary statistics
                            st.markdown(f"""
                            **Summary:**
                            - Total tables: {len(tables_df)}
                            - Total size: {total_size:.2f} GB
                            - Largest table: {largest_table_name} ({max_size:.2f} GB)
                            """)

                        except Exception as e:
                            st.error(f"Error creating table size visualization: {str(e)}")
                            logger.error(f"Error creating visualization: {str(e)}")
                            logger.debug(f"Visualization error details: {traceback.format_exc()}")

                        # Analyze each table in detail
                        st.subheader("Table Analysis")

                        # For each table, get actual query patterns
                        for idx, table in tables_df.iterrows():
                            # Get table name - handle different column names
                            table_name = table.get('table_name',
                                                   table.get('name',
                                                             f"Table {idx}"))  # Fallback if no name column found

                            with st.expander(f"📊 {table_name} Analysis"):
                                st.markdown(f"### {table_name}")


                                # Helper function to safely get values
                                def safe_get(obj, key, default="Not available"):
                                    """Safely get a value from a dict or object with a default"""
                                    if isinstance(obj, dict):
                                        return obj.get(key, default)
                                    try:
                                        return obj[key] if key in obj.index else default
                                    except:
                                        try:
                                            return getattr(obj, key, default)
                                        except:
                                            return default


                                # Get table properties with safe access
                                try:
                                    # Get size information
                                    if 'size_bytes' in table:
                                        size_bytes = pd.to_numeric(safe_get(table, 'size_bytes', 0))
                                        size_display = format_bytes(size_bytes)
                                    elif 'size_gb' in table:
                                        size_gb = pd.to_numeric(safe_get(table, 'size_gb', 0))
                                        size_display = f"{size_gb:.2f} GB"
                                    elif 'data_size' in table:
                                        size_display = safe_get(table, 'data_size', "Unknown")
                                    else:
                                        size_display = "Unknown"

                                    # Get row count
                                    if 'row_count' in table:
                                        row_count = safe_get(table, 'row_count', "Unknown")
                                    elif 'num_rows' in table:
                                        row_count = safe_get(table, 'num_rows', "Unknown")
                                    else:
                                        row_count = "Unknown"

                                    # Format row count if it's a number
                                    try:
                                        row_count = f"{int(row_count):,}"
                                    except:
                                        pass

                                    # Get other properties
                                    partition_type = safe_get(table, 'partition_type', "Unknown")
                                    schema_fields = safe_get(table, 'schema_fields', "N/A")
                                    creation_time = safe_get(table, 'creation_time', "Unknown")
                                    last_modified = safe_get(table, 'last_modified', "Unknown")

                                    # Display table information with safe values
                                    st.markdown(f"""
                                    <table class="table-stats">
                                        <tr>
                                            <th>Size</th>
                                            <td>{size_display}</td>
                                            <th>Row Count</th>
                                            <td>{row_count}</td>
                                        </tr>
                                        <tr>
                                            <th>Partition Type</th>
                                            <td>{partition_type}</td>
                                            <th>Schema Fields</th>
                                            <td>{schema_fields}</td>
                                        </tr>
                                        <tr>
                                            <th>Created</th>
                                            <td>{creation_time}</td>
                                            <th>Last Modified</th>
                                            <td>{last_modified}</td>
                                        </tr>
                                    </table>
                                    """, unsafe_allow_html=True)

                                except Exception as e:
                                    # Fallback to simple display
                                    st.error(f"Error showing table details: {str(e)}")

                                    # Display whatever properties we can
                                    st.write("**Table Properties:**")
                                    for col in table.index:
                                        if pd.notna(table[col]) and str(table[col]).strip() != "":
                                            st.write(f"**{col}:** {table[col]}")

                                # Get query patterns for this table
                                try:
                                    table_patterns = get_table_query_patterns(project_id, dataset_id, table_name)

                                    if table_patterns and table_patterns.get('query_count', 0) > 0:
                                        # Display query pattern insights
                                        st.markdown("#### Query Pattern Analysis")

                                        # Get query pattern metrics with safe defaults
                                        query_count = table_patterns.get('query_count', 0)
                                        avg_slot_ms = table_patterns.get('avg_slot_ms', 0)
                                        avg_execution_ms = table_patterns.get('avg_execution_ms', 0)
                                        avg_bytes_processed = table_patterns.get('avg_bytes_processed', 0)
                                        filter_count = table_patterns.get('filter_count', 0)
                                        group_by_count = table_patterns.get('group_by_count', 0)
                                        join_count = table_patterns.get('join_count', 0)

                                        st.markdown(f"""
                                        This table was queried **{query_count} times** in the past 30 days.

                                        **Usage Patterns:**
                                        - Average slot time: {format_number(avg_slot_ms, is_time=True)}
                                        - Average execution time: {format_number(avg_execution_ms, is_time=True)}
                                        - Average bytes processed: {format_number(avg_bytes_processed, is_bytes=True)}
                                        - Filters used: {filter_count}
                                        - GROUP BY operations: {group_by_count}
                                        - JOIN operations: {join_count}
                                        """)

                                        # Show common filter columns
                                        common_filter_columns = table_patterns.get('common_filter_columns', [])
                                        if common_filter_columns:
                                            st.markdown("#### Common Filter Columns")

                                            try:
                                                filter_df = pd.DataFrame(common_filter_columns)

                                                # Display as a horizontal bar chart
                                                if len(filter_df) > 0 and 'column' in filter_df.columns and 'count' in filter_df.columns:
                                                    filter_fig = px.bar(
                                                        filter_df,
                                                        y='column',
                                                        x='count',
                                                        orientation='h',
                                                        title='Most Commonly Filtered Columns',
                                                        labels={'count': 'Times used in WHERE clause',
                                                                'column': 'Column name'}
                                                    )
                                                    filter_fig.update_layout(height=250)
                                                    st.plotly_chart(filter_fig, use_container_width=True)
                                                else:
                                                    st.write("Common filters:", ", ".join(
                                                        [f"{f.get('column', 'Unknown')} ({f.get('count', 0)})" for f in
                                                         common_filter_columns]))
                                            except Exception as e:
                                                st.warning(f"Could not visualize filter columns: {str(e)}")
                                                st.write("Common filters:", common_filter_columns)

                                        # Generate table-specific recommendations
                                        st.markdown("#### Optimization Recommendations")

                                        # Helper to get size in GB safely
                                        table_size_gb = 0
                                        try:
                                            if 'size_gb' in table:
                                                table_size_gb = float(table['size_gb'])
                                            elif 'size_bytes' in table:
                                                table_size_gb = float(table['size_bytes']) / (1024 ** 3)
                                        except:
                                            table_size_gb = 0

                                        # Check if we need partitioning
                                        try:
                                            if table_size_gb > 1 and str(safe_get(table, 'partition_type')).lower() in [
                                                'none', 'not partitioned', 'unknown']:
                                                # Find potential partition column
                                                partition_column = None
                                                for col_info in common_filter_columns:
                                                    col_name = col_info.get('column', '').lower()
                                                    if 'date' in col_name or 'time' in col_name or 'day' in col_name:
                                                        partition_column = col_info.get('column')
                                                        break

                                                if partition_column:
                                                    st.markdown(f"""
                                                    <div class="recommendation recommendation-high">
                                                        <strong>Add Time-Based Partitioning</strong><br>
                                                        This table is {table_size_gb:.2f} GB and commonly filtered on '{partition_column}'. 
                                                        Consider partitioning by this column to reduce query costs.

                                                        Potential DDL:
                                                        <pre>
                                                        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_name}`
                                                        PARTITION BY DATE({partition_column})
                                                        AS SELECT * FROM `{project_id}.{dataset_id}.{table_name}`
                                                        </pre>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                else:
                                                    st.markdown(f"""
                                                    <div class="recommendation recommendation-medium">
                                                        <strong>Consider Partitioning</strong><br>
                                                        This table is {table_size_gb:.2f} GB with no partitioning.
                                                        Adding partitioning could reduce query costs if there's a suitable date/timestamp column.
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                        except Exception as e:
                                            logger.error(f"Error generating partitioning recommendation: {str(e)}")

                                        # Check if we need clustering
                                        try:
                                            if filter_count > 0 and common_filter_columns:
                                                top_filter = common_filter_columns[0].get(
                                                    'column') if common_filter_columns else None

                                                if top_filter and top_filter.lower() not in ['date', 'timestamp', 'day',
                                                                                             'month', 'year']:
                                                    st.markdown(f"""
                                                    <div class="recommendation recommendation-medium">
                                                        <strong>Add Clustering</strong><br>
                                                        This table is frequently filtered on '{top_filter}'. Consider adding clustering on this column.

                                                        Potential DDL:
                                                        <pre>
                                                        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_name}`
                                                        {f"PARTITION BY DATE({partition_column})" if partition_column else ""}
                                                        CLUSTER BY {top_filter}
                                                        AS SELECT * FROM `{project_id}.{dataset_id}.{table_name}`
                                                        </pre>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                        except Exception as e:
                                            logger.error(f"Error generating clustering recommendation: {str(e)}")

                                        # Check if materialized views would help
                                        try:
                                            if group_by_count > 1 and query_count > 5:
                                                st.markdown(f"""
                                                <div class="recommendation recommendation-medium">
                                                    <strong>Consider Materialized Views</strong><br>
                                                    This table is frequently queried with GROUP BY operations. 
                                                    Materialized views could improve performance for these common aggregation patterns.
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except Exception as e:
                                            logger.error(f"Error generating materialized view recommendation: {str(e)}")

                                    else:
                                        st.info("No query pattern data available for this table in the past 30 days.")

                                except Exception as e:
                                    st.warning(f"Could not analyze query patterns: {str(e)}")

                        # Overall schema recommendations
                        st.subheader("Overall Schema Recommendations")

                        # Identify large tables without partitioning
                        try:
                            # Make sure size_gb is numeric
                            tables_df['size_gb'] = pd.to_numeric(tables_df['size_gb'], errors='coerce').fillna(0)


                            # Get partition type as consistent format
                            def normalize_partition_type(p_type):
                                if pd.isna(p_type) or not p_type or str(p_type).lower() in ['none', 'not partitioned']:
                                    return 'None'
                                return str(p_type)


                            tables_df['norm_partition_type'] = tables_df.apply(
                                lambda row: normalize_partition_type(row.get('partition_type', 'None')),
                                axis=1
                            )

                            # Find large tables without partitioning
                            large_tables = tables_df[
                                (tables_df['size_gb'] > 5) & (tables_df['norm_partition_type'] == 'None')]
                            logger.debug(f"Found {len(large_tables)} large non-partitioned tables")

                            if len(large_tables) > 0:
                                for _, table in large_tables.iterrows():
                                    table_name = table.get('table_name', 'Unknown')
                                    table_size = float(table.get('size_gb', 0))
                                    row_count = table.get('row_count', table.get('num_rows', 'N/A'))

                                    # Format row count if it's a number
                                    try:
                                        row_count = f"{int(row_count):,}"
                                    except:
                                        pass

                                    st.markdown(f"""
                                    <div class="recommendation recommendation-high">
                                        <strong>Prioritize Partitioning for {table_name}</strong><br>
                                        This table is {table_size:.1f} GB with {row_count} rows and no partitioning.
                                        Adding partitioning could significantly reduce query costs and improve performance.
                                    </div>
                                    """, unsafe_allow_html=True)
                                    logger.info(
                                        f"Recommended partitioning for large table: {table_name} ({table_size:.1f} GB)")
                            else:
                                st.success("All large tables are appropriately partitioned. Good job!")
                                logger.debug("No large non-partitioned tables found")

                        except Exception as e:
                            st.warning(f"Could not generate overall schema recommendations: {str(e)}")
                            logger.error(f"Error in overall schema recommendations: {str(e)}")

                except Exception as e:
                    logger.error(f"Error analyzing schema: {str(e)}")
                    logger.debug(f"Schema analysis error details: {traceback.format_exc()}")
                    st.error(f"Error analyzing schema: {str(e)}")
                    st.markdown("""
                    Please ensure:
                    1. The project and dataset IDs are correct
                    2. You have appropriate permissions
                    3. The dataset exists and contains tables
                    """)

    # Benchmarking tab
    with subtab3:
        logger.info("User navigated to Benchmarking subtab")
        st.subheader("Query Benchmarking")
        st.write("Compare execution time and cost of different query implementations")

        # Sample benchmark input
        benchmark_queries = st.text_area(
            "Enter multiple query variants separated by '---'",
            height=200,
            value="""-- Query 1: Using JOIN
SELECT 
  customers.name,
  COUNT(orders.id) as order_count
FROM customers
JOIN orders ON customers.id = orders.customer_id
WHERE orders.order_date > '2023-01-01'
GROUP BY customers.name
---
-- Query 2: Using EXISTS
SELECT 
  customers.name,
  (SELECT COUNT(*) FROM orders WHERE customers.id = orders.customer_id AND orders.order_date > '2023-01-01') as order_count
FROM customers
WHERE EXISTS (SELECT 1 FROM orders WHERE customers.id = orders.customer_id AND orders.order_date > '2023-01-01')
"""
        )

        benchmark_button = st.button("⚡ Run Benchmark", type="primary")

        if benchmark_button and benchmark_queries:
            logger.info("User clicked Run Benchmark button")
            with st.spinner("Running benchmark..."):
                # Split queries
                queries = [q.strip() for q in benchmark_queries.split('---')]
                logger.debug(f"Benchmarking {len(queries)} query variants")

                # Get predictor
                logger.debug("Getting QueryPredictor for benchmark")
                predictor = get_predictor()

                # Run prediction on each query
                benchmark_results = []

                for i, query in enumerate(queries):
                    # Extract the comment/name if present
                    query_name = f"Query {i + 1}"
                    if query.startswith('--'):
                        first_line = query.split('\n')[0]
                        query_name = first_line.replace('--', '').strip()

                    logger.debug(f"Benchmarking query variant: {query_name}")

                    try:
                        # Estimate query size
                        size_result = get_query_size(query)
                        total_bytes = size_result['processed_bytes'] if size_result['processed_bytes'] else 100_000_000

                        # Predict query performance
                        start_time = time.time()
                        prediction = predictor.predict(query, total_bytes)
                        prediction_time = time.time() - start_time
                        logger.debug(f"Prediction for {query_name} completed in {prediction_time:.2f} seconds")

                        # Add to results
                        benchmark_results.append({
                            'query_name': query_name,
                            'query_text': query,
                            'slot_ms': prediction['predicted_slot_ms'],
                            'execution_ms': prediction['predicted_execution_ms'],
                            'cost_usd': prediction['estimated_cost_usd'],
                            'bytes_processed': total_bytes,
                            'formatted_bytes': size_result['formatted_size']
                        })
                        logger.info(f"Benchmark result for {query_name}: slot_ms={prediction['predicted_slot_ms']}, " +
                                    f"execution_ms={prediction['predicted_execution_ms']}, cost=${prediction['estimated_cost_usd']:.6f}")
                    except Exception as e:
                        logger.error(f"Error benchmarking query {query_name}: {str(e)}")
                        st.error(f"Error benchmarking {query_name}: {str(e)}")
                        # Add a placeholder result
                        benchmark_results.append({
                            'query_name': f"{query_name} (ERROR)",
                            'query_text': query,
                            'slot_ms': 0,
                            'execution_ms': 0,
                            'cost_usd': 0,
                            'bytes_processed': 0,
                            'formatted_bytes': 'Error'
                        })

                if benchmark_results:
                    # Create results dataframe
                    results_df = pd.DataFrame(benchmark_results)

                    # Choose best query based on cost
                    if not all(result['cost_usd'] == 0 for result in benchmark_results):
                        best_query_idx = results_df['cost_usd'].idxmin()
                        best_query = results_df.iloc[best_query_idx]
                        logger.info(f"Best query: {best_query['query_name']} with cost ${best_query['cost_usd']:.6f}")

                        # Show benchmark results
                        st.markdown("### Benchmark Results")

                        # Format results for display
                        display_df = results_df[
                            ['query_name', 'formatted_bytes', 'slot_ms', 'execution_ms', 'cost_usd']].copy()
                        display_df['slot_ms'] = display_df['slot_ms'].apply(lambda x: format_number(x, is_time=True))
                        display_df['execution_ms'] = display_df['execution_ms'].apply(
                            lambda x: format_number(x, is_time=True))
                        display_df['cost_usd'] = display_df['cost_usd'].apply(lambda x: f"${x:.6f}")

                        # Rename columns for display
                        display_df.columns = ['Query Variant', 'Data Processed', 'Slot Time', 'Execution Time', 'Cost']

                        st.dataframe(display_df, use_container_width=True)
                        logger.debug("Displayed benchmark results table")

                        # Visual comparison
                        logger.debug("Creating visual benchmark comparison")

                        # Create a radar chart for visual comparison
                        radar_data = []

                        for _, row in results_df.iterrows():
                            # Normalize metrics for better visualization
                            max_slot = results_df['slot_ms'].max()
                            max_exec = results_df['execution_ms'].max()
                            max_bytes = results_df['bytes_processed'].max()
                            max_cost = results_df['cost_usd'].max()

                            # Invert the metrics so smaller is better (larger on chart)
                            if max_slot > 0 and max_exec > 0 and max_bytes > 0 and max_cost > 0:
                                slot_norm = 1 - (row['slot_ms'] / max_slot) if max_slot > 0 else 0
                                exec_norm = 1 - (row['execution_ms'] / max_exec) if max_exec > 0 else 0
                                bytes_norm = 1 - (row['bytes_processed'] / max_bytes) if max_bytes > 0 else 0
                                cost_norm = 1 - (row['cost_usd'] / max_cost) if max_cost > 0 else 0

                                radar_data.append({
                                    'query': row['query_name'],
                                    'Slot Efficiency': slot_norm * 10,  # Scale to 0-10
                                    'Execution Efficiency': exec_norm * 10,
                                    'Data Efficiency': bytes_norm * 10,
                                    'Cost Efficiency': cost_norm * 10
                                })

                        # Convert to format for radar chart
                        radar_df = pd.DataFrame(radar_data)

                        if not radar_df.empty:
                            # Create radar chart for comparison
                            radar_fig = go.Figure()

                            for i, query in enumerate(radar_df['query']):
                                radar_fig.add_trace(go.Scatterpolar(
                                    r=[
                                        radar_df.iloc[i]['Slot Efficiency'],
                                        radar_df.iloc[i]['Execution Efficiency'],
                                        radar_df.iloc[i]['Data Efficiency'],
                                        radar_df.iloc[i]['Cost Efficiency'],
                                        radar_df.iloc[i]['Slot Efficiency']  # Close the loop
                                    ],
                                    theta=['Slot Efficiency', 'Execution Efficiency', 'Data Efficiency',
                                           'Cost Efficiency', 'Slot Efficiency'],
                                    fill='toself',
                                    name=query
                                ))

                            radar_fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 10]
                                    )
                                ),
                                title="Query Efficiency Comparison (Higher is Better)",
                                height=500
                            )

                            st.plotly_chart(radar_fig, use_container_width=True)

                        # Also add a bar chart comparison for raw metrics
                        metrics_fig = px.bar(
                            results_df,
                            x='query_name',
                            y=['slot_ms', 'execution_ms', 'cost_usd'],
                            title='Raw Metrics Comparison',
                            barmode='group',
                            labels={
                                'query_name': 'Query Variant',
                                'slot_ms': 'Slot Time (ms)',
                                'execution_ms': 'Execution Time (ms)',
                                'cost_usd': 'Cost ($)'
                            }
                        )

                        # Using log scale for better visualization
                        metrics_fig.update_layout(yaxis_type="log")
                        st.plotly_chart(metrics_fig, use_container_width=True)

                        # Highlight the best query
                        st.success(
                            f"📊 Best Query: **{best_query['query_name']}** with cost ${best_query['cost_usd']:.6f}")

                        with st.expander("View Best Query"):
                            st.code(best_query['query_text'], language="sql")
                            logger.debug(f"Displayed best query code: {best_query['query_name']}")

                        # Detailed comparison table
                        st.markdown("### Detailed Comparison")

                        # Calculate percentages compared to the worst option
                        worst_slot = results_df['slot_ms'].max()
                        worst_exec = results_df['execution_ms'].max()
                        worst_bytes = results_df['bytes_processed'].max()
                        worst_cost = results_df['cost_usd'].max()

                        comparison_rows = []
                        for _, row in results_df.iterrows():
                            if worst_slot > 0 and worst_exec > 0 and worst_bytes > 0 and worst_cost > 0:
                                slot_saving = ((worst_slot - row['slot_ms']) / worst_slot) * 100
                                exec_saving = ((worst_exec - row['execution_ms']) / worst_exec) * 100
                                bytes_saving = ((worst_bytes - row['bytes_processed']) / worst_bytes) * 100
                                cost_saving = ((worst_cost - row['cost_usd']) / worst_cost) * 100

                                comparison_rows.append({
                                    'Query': row['query_name'],
                                    'Slot Time': format_number(row['slot_ms'], is_time=True),
                                    'Slot Saving': f"{slot_saving:.1f}%",
                                    'Execution Time': format_number(row['execution_ms'], is_time=True),
                                    'Execution Saving': f"{exec_saving:.1f}%",
                                    'Data Processed': row['formatted_bytes'],
                                    'Data Saving': f"{bytes_saving:.1f}%",
                                    'Cost': f"${row['cost_usd']:.6f}",
                                    'Cost Saving': f"{cost_saving:.1f}%"
                                })

                        comparison_df = pd.DataFrame(comparison_rows)
                        st.table(comparison_df)
                    else:
                        st.warning("Could not determine the best query due to estimation errors.")

# Footer
logger.debug("Displaying application footer")
st.markdown("---")
st.caption("PipelineIQ - Resource Intelligence Platform v2.0 © 2025")
st.markdown("""
<div class="footer">
    <p>Powered by MIT Labs</p>
    <p>Need help optimizing your workloads? <a href="mailto:smhrizvi281@gmail.com">Contact our data experts</a>.</p>
</div>
""", unsafe_allow_html=True)

# Feature request button
with st.sidebar:
    st.title("PipelineIQ")
    st.write("Advanced Resource Intelligence")
    logger.debug("Set up application sidebar")

    st.markdown("---")
    st.subheader("Settings")

    # Theme selector
    theme = st.selectbox("UI Theme", ["Light", "Dark", "System Default"], index=0)
    logger.debug(f"User selected theme: {theme}")

    # Cost settings
    cost_per_slot = st.number_input(
        "Cost per slot-ms ($)",
        min_value=0.0000000001,
        max_value=0.0000001000,
        value=0.00000001,
        format="%.12f"
    )
    logger.debug(f"User set cost_per_slot to: {cost_per_slot}")

    # Data refresh settings
    refresh_interval = st.slider(
        "Data refresh interval (minutes):",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )
    logger.debug(f"User set refresh interval to: {refresh_interval} minutes")

    st.markdown("---")
    st.subheader("About")
    st.write(
        "PipelineIQ helps you optimize BigQuery performance and cost with predictive analytics and AI-powered recommendations.")

    # Connection status
    try:
        # Simple connection test to BigQuery
        from google.cloud import bigquery

        client = bigquery.Client()
        st.success("✅ Connected to BigQuery")
    except:
        st.error("❌ Not connected to BigQuery")

    # Session info
    st.markdown(f"Session ID: {SESSION_ID[:8]}...")

    # Feature request form in sidebar
    with st.expander("Request a Feature"):
        feature_request = st.text_area("Describe the feature you'd like:", height=100)
        email = st.text_input("Your email (optional):")
        if st.button("Submit Request"):
            st.success("Thank you for your feedback! We'll consider your request for future updates.")
