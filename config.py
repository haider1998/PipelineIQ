BQ_CONFIG = {
    'total_slots': 30000, # Total slots in the BigQuery reservation
    'cost_per_slot_ms': 0.00000001, # Cost per slot-millisecond ($0.01 per 1 million slots)
    'cost_per_slot_hour': 0.036 # Cost per slot-hour ($36 per 1,000 slots per hour)
}

TRAINING_DATA_PATH = 'data/training_data.csv' # Path to training data CSV file

API_KEY = r'{GEMINI-API-KEY}' # Your Gemini API key

PRPJECT_ID = '{YOUR-PROJECT-ID}' # Your GCP project ID

QUERY_HISTORY_TABLE = '{YOUR-PROJECT-ID.DATASET.HISTORY-TABLE}' # Table storing query execution history