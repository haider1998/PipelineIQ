import requests
import json
import argparse
from config import API_KEY


def query_gemini_api(message, api_key=None):
    """
    Query the Gemini 2.0 Pro API with a dynamic input message.

    Args:
        message (str): The input text message to send to Gemini
        api_key (str, optional): Your API key. If not provided, will prompt for it.

    Returns:
        dict: The JSON response from the API
    """
    # Get API key if not provided
    if not api_key:
        api_key = API_KEY

    # API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro-exp-02-05:generateContent?key={api_key}"

    # Prepare the request payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": message
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 1,
            "topK": 64,
            "topP": 0.95,
            "maxOutputTokens": 8192,
            "responseMimeType": "text/plain"
        }
    }

    # Set headers
    headers = {
        "Content-Type": "application/json"
    }

    # Print what we're about to send
    print(f"\nSending message to Gemini API: \"{message}\"")

    # Make the API call
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(response.json())
        return extract_response_text(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}


def extract_response_text(response):
    """Extract the text response from the Gemini API response JSON"""
    try:
        # Navigate the response structure to get the text
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        print(f"Error parsing response: {e}")
        print("Full response:", json.dumps(response, indent=2))
        return "Error extracting response text"


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Query the Gemini API with a dynamic message")
    parser.add_argument("--message", "-m", help="Message to send to Gemini")
    parser.add_argument("--api-key", "-k", help="Your Gemini API key")
    args = parser.parse_args()

    # Get message from command line or prompt for it
    message = args.message
    if not message:
        message = """LLM-READY OPTIMIZATION CONTEXT:
---------------------------------
# BigQuery Optimization Task

## Original Query
```sql

    SELECT 
        *
    FROM 
        `dfa50884-d.dfa_campaign_service_op675_usa_sxm_used_fz_db.usasxmm01_owner_used_ob_pfl`

```

## Query Intent
retrieving data from tables

## Performance Metrics
- Estimated data processed: 0.0 GB
- Estimated cost: $0.0

## Table Information

### Table: dfa50884-d.dfa_campaign_service_op675_usa_sxm_used_fz_db.usasxmm01_owner_used_ob_pfl
- Rows: 15
- Size: 0.0 GB
- Partitioning: date
- Clustering: None

#### Columns:
- type (STRING): Column of type STRING
- column (STRING): Column of type STRING
- field (STRING): Column of type STRING
- value (STRING): Column of type STRING
- date (DATETIME): Temporal column of type DATETIME

## Query Structure

### SELECT Columns
- All columns (*)

## Optimization Recommendations
- [MEDIUM] Query uses SELECT * which processes all columns.
  → Select only needed columns to reduce bytes processed.

## Your Task
Please rewrite this query to improve its performance and reduce costs while maintaining the exact same output results. Focus on:

1. Using partitioning and clustering effectively
2. Reducing the amount of data processed
3. Optimizing join operations
4. Improving filter conditions
5. Selecting only necessary columns

Provide the optimized query and explain your optimization strategies."""

    # Get response from API
    response = query_gemini_api(message, args.api_key)

    # Extract and print the text response
    if "error" not in response:
        text_response = extract_response_text(response)
        print("\nGemini Response:")
        print("-" * 40)
        print(text_response)
        print("-" * 40)
    else:
        print("\nError:", response["error"])


if __name__ == "__main__":
    main()

# import requests
# from openai import OpenAI
# from llm_api_token_generator import get_token
#
# token = get_token()
#
# def generate_results(message: str):
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": {message}
#         },
#     ]
#
#     client = OpenAI(
#         api_key=token,
#         base_url="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
#     )
#
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages
#     )
#
#     print(completion.choices[0].message.content)
#
#
# print(generate_results('test'))