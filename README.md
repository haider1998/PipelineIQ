# PipelineIQ 📊

![Version](https://img.shields.io/badge/Version-2.0-1E88E5)
![BigQuery](https://img.shields.io/badge/BigQuery-Intelligence-4285F4)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![AI Powered](https://img.shields.io/badge/AI-Powered-9C27B0)

<div align="center">
  <h3>Intelligent Resource Prediction & Optimization of Data Pipeline</h3>
  <p><i>Analyze, predict, optimize, and save before you run a single query</i></p>
</div>

---

## 🔮 What is PipelineIQ?

**PipelineIQ** is a revolutionary AI-powered platform that transforms how data teams work with Google BigQuery. It predicts query performance, provides intelligent optimization recommendations, and delivers actionable cost-saving insights—all before you run a single query.

Instead of guessing resource requirements or dealing with unexpectedly expensive queries, PipelineIQ gives you complete visibility into how your queries will perform, where bottlenecks might occur, and how to optimize for maximum performance at minimum cost.

<div align="center">
  <img src="docs/Resource_Insights.png" alt="PipelineIQ Dashboard" width="800"/>
  <p><i>PipelineIQ's intuitive interface provides comprehensive insights into your BigQuery operations</i></p>
</div>

## 🌟 Key Features

### 📝 Advanced Query Analysis
- **Predictive Intelligence**: Forecast slot usage, execution time, and cost before execution
- **Complexity Analysis**: Identify complex query patterns with radar visualization
- **Historical Comparison**: Compare new queries against similar historical queries
- **Batch Processing**: Analyze entire query libraries in one operation
- **AI-Powered Optimization**: Get tailored rewrites of your queries for better performance
<div align="center">
  <img src="docs/LLM_Query_Optimizer.png" alt="LLM_Query_Optimizer Dashboard" width="800"/>
  <img src="docs/Query_Analyzer.png" alt="Query_Analyzer Dashboard" width="800"/>
  <img src="docs/Query_Insights.png" alt="Query_Insights Dashboard" width="800"/>
</div>

### 📊 Resource Intelligence
- **Usage Pattern Detection**: Identify trending metrics and usage patterns
- **Weekly & Daily Insights**: Visualize resource usage patterns by time period
- **Forecasting**: Project future resource needs and costs
- **Historical Analysis**: Track performance metrics over time
- **Query Type Distribution**: Understand your workload composition

### 💰 Cost Optimization
- **Slot Utilization Analysis**: Monitor and optimize slot usage
- **Commitment Recommendations**: Get data-driven slot commitment suggestions
- **Cost Distribution Analysis**: Identify high-cost queries and patterns
- **ROI Calculator**: Determine payback periods for optimization investments
- **Trend Analysis**: Track cost metrics over time with forecasting

<div align="center">
   <img src="docs/Cost_Optimization.png" alt="Cost_Optimization Dashboard" width="800"/>
</div>

### 🔍 Schema Intelligence
- **Table Analysis**: Get comprehensive insights into your dataset structure
- **Query Pattern Detection**: See how tables are actually being used
- **Smart Recommendations**: Receive tailored partitioning and clustering advice
- **Performance Impact**: Understand the cost implication of schema changes
- **Implementation DDL**: Get ready-to-run SQL for implementing recommendations

<div align="center">
   <img src="docs/Schema_Analyser.png" alt="Schema_Analyser Dashboard" width="800"/>
</div>

### ⚡ Performance Benchmarking
- **Multi-Query Comparison**: Test different implementations against each other
- **Visual Efficiency Analysis**: See performance differences in radar charts
- **Detailed Metrics**: Compare slot time, execution time, data processed and cost
- **Best Option Identification**: Automatically identify the optimal query variant
- **Savings Calculator**: Quantify the savings from choosing the optimal approach

## 🎯 Who Benefits from PipelineIQ?

PipelineIQ delivers specific value to different members of your data team:

| Role | Benefits |
|------|----------|
| **Data Engineers** | Optimize query performance, reduce costs, implement best practices automatically, benchmark alternative implementations |
| **Data Scientists** | Test query performance before running expensive operations, get insights into data access patterns, improve model feature engineering |
| **Database Architects** | Make data-driven schema design decisions, implement optimal partitioning and clustering strategies, track schema evolution |
| **BI Developers** | Create efficient dashboards, optimize report queries, reduce dashboard loading times |
| **DevOps Engineers** | Right-size BigQuery commitments, predict resource needs, optimize infrastructure spend |
| **Engineering Managers** | Control cloud costs, forecast budgeting needs, track optimization progress, demonstrate ROI |

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Cloud Platform account with BigQuery access
- Appropriate permissions to read and analyze query history

### Installation

```bash
# Clone the repository
git clone https://github.com/haider1998/PipelineIQ.git
cd PipelineIQ

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Google Cloud authentication
gcloud auth application-default login

# Launch the application
streamlit run streamlit_app.py
```

### Quick Start Guide

1. **Analyze a Single Query**:
   - Navigate to the Query Analyzer tab
   - Enter your SQL query or choose a sample
   - Click "Analyze Query" to see performance predictions

2. **Batch Process Multiple Queries**:
   - Prepare your SQL files or a ZIP archive
   - Select "Upload SQL File(s)" on the Query Analyzer tab
   - Enable "Batch analyze all SQL files"
   - Review performance metrics for all queries

3. **Review Resource Insights**:
   - Check the Resource Insights tab for historical patterns
   - Select different time periods to analyze trends
   - View weekly patterns and usage forecasts

4. **Optimize Costs**:
   - Visit the Cost Optimization tab
   - Review slot utilization and commitment recommendations
   - Identify high-impact optimization opportunities

5. **Analyze Schema**:
   - Use the Schema Analyzer in the Advanced Tools section
   - Enter your project and dataset IDs
   - Get partitioning and clustering recommendations

## 💡 What Makes PipelineIQ Different?

PipelineIQ stands apart from other BigQuery tools through:

### 🧠 Intelligent Prediction
Unlike basic estimation tools, PipelineIQ uses advanced ML models trained on actual execution data to provide accurate predictions.

### 🔄 Full Lifecycle Coverage
PipelineIQ addresses the entire BigQuery optimization lifecycle: prediction, analysis, recommendation, and validation.

### 📈 Real-Time Learning
The system continuously improves by learning from your actual query patterns and execution history.

### 💵 Tangible Cost Savings
Users report 20-40% cost reductions through implementing PipelineIQ's optimization recommendations.

### 🛠️ Actionable Recommendations
Beyond identifying issues, PipelineIQ provides concrete, ready-to-implement solutions.

### 🤖 AI-Powered Optimization
Leverages advanced LLMs to rewrite and optimize your queries for better performance.

## 🏗️ Architecture

<div align="center">
  <img src="docs/PipelineIQ Architecture Diagram.svg" alt="PipelineIQ Architecture" width="800"/>
</div>

PipelineIQ integrates several advanced components:

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│                    │     │                    │     │                    │
│  Streamlit UI      │────▶│  Analysis Engine   │────▶│  BigQuery API      │
│                    │     │                    │     │                    │
└────────────────────┘     └────────────────────┘     └────────────────────┘
         │                          │                          │
         │                          │                          │
         ▼                          ▼                          ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│                    │     │                    │     │                    │
│  ML Prediction     │◀───▶│  Recommendation    │◀───▶│  Historical Data   │
│  Models            │     │  Engine            │     │  Store             │
│                    │     │                    │     │                    │
└────────────────────┘     └────────────────────┘     └────────────────────┘
                                    │
                                    │
                                    ▼
                           ┌────────────────────┐
                           │                    │
                           │  LLM Optimization  │
                           │  Service           │
                           │                    │
                           └────────────────────┘
```

## 📂 Project Structure

```
PipelineIQ/
├── streamlit_app.py      # Main Streamlit application
├── big_query_estimate.py # BigQuery data access and analysis
├── model_prediction.py   # ML prediction services
├── llm_preprocessing.py  # LLM data preparation
├── llm_generator.py      # AI optimization services
├── config.py             # Configuration settings
├── models/               # Trained ML models
│   ├── slot_predictor.joblib
│   └── execution_predictor.joblib
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## 🛣️ Roadmap

Future development plans include:

- **Integration with dbt**: Analyze and optimize dbt models automatically
- **Multi-Cloud Support**: Extend to other cloud data warehouses
- **Add other GCP Products**: Include Dataflow, Dataproc, CloudRun, etc of Data pipelines
- **CI/CD Integration**: Incorporate query analysis into deployment pipelines
- **Enterprise Features**: SSO, team collaboration, and governance controls
- **Query Catalog**: Build a searchable library of optimized query patterns

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Contact us at **Syed Mohd Haider Rizvi** (smhrizvi281@gmail.com)

---

<div align="center">
  <p>Made with ❤️ by Syed Mohd Haider Rizvi © 2025</p>
  <p>
    <a href="https://github.com/haider1998">GitHub</a> •
    <a href="https://github.com/haider1998/PipelineIQ/tree/main">Documentation</a> •
    <a href="mailto:smhrizvi281@gmail.com">Contact</a>
  </p>
</div>