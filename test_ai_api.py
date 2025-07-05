import requests
import json

# URL of your Flask AI service (it's running locally on port 5000)
BASE_URL = "http://127.0.0.1:5000" # Use 127.0.0.1 if testing from the same machine

print("--- Testing /predict_hsse_relevance ---")
# Example texts you'd send for proactive monitoring
proactive_text_1 = "Major chemical spill at the industrial park, emergency response initiated."
proactive_text_2 = "Local government announces new cultural festival for next month."
proactive_text_3 = "Safety audit reveals multiple violations at construction site in Georgetown."
proactive_text_4 = "Market prices for gold have surged this quarter."

# Send POST requests to the proactive monitoring endpoint
try:
    response_1 = requests.post(f"{BASE_URL}/predict_hsse_relevance", json={"text": proactive_text_1})
    print(f"Text 1: '{proactive_text_1}' -> {response_1.json()}")

    response_2 = requests.post(f"{BASE_URL}/predict_hsse_relevance", json={"text": proactive_text_2})
    print(f"Text 2: '{proactive_text_2}' -> {response_2.json()}")

    response_3 = requests.post(f"{BASE_URL}/predict_hsse_relevance", json={"text": proactive_text_3})
    print(f"Text 3: '{proactive_text_3}' -> {response_3.json()}")

    response_4 = requests.post(f"{BASE_URL}/predict_hsse_relevance", json={"text": proactive_text_4})
    print(f"Text 4: '{proactive_text_4}' -> {response_4.json()}")

except requests.exceptions.ConnectionError:
    print(f"Error: Could not connect to the Flask app at {BASE_URL}. Is ai_service_app.py running?")
except Exception as e:
    print(f"An unexpected error occurred during proactive monitoring test: {e}")


print("\n--- Testing /predict_report_category ---")
# Example report texts you'd send for categorization
report_text_1 = "Detailed report of a minor electrical fire in the main office server room on June 29, 2025."
report_text_2 = "Investigation into a fall from height incident involving scaffolding, worker sustained a sprained ankle."
report_text_3 = "Environmental inspection report indicating discharge of untreated wastewater into the local creek."
report_text_4 = "Routine maintenance check on heavy machinery completed without incidents."


# Send POST requests to the report categorization endpoint
try:
    response_5 = requests.post(f"{BASE_URL}/predict_report_category", json={"text": report_text_1})
    print(f"Report 1: '{report_text_1}' -> {response_5.json()}")

    response_6 = requests.post(f"{BASE_URL}/predict_report_category", json={"text": report_text_2})
    print(f"Report 2: '{report_text_2}' -> {response_6.json()}")

    response_7 = requests.post(f"{BASE_URL}/predict_report_category", json={"text": report_text_3})
    print(f"Report 3: '{report_text_3}' -> {response_7.json()}")

    response_8 = requests.post(f"{BASE_URL}/predict_report_category", json={"text": report_text_4})
    print(f"Report 4: '{report_text_4}' -> {response_8.json()}")

except requests.exceptions.ConnectionError:
    print(f"Error: Could not connect to the Flask app at {BASE_URL}. Is ai_service_app.py running?")
except Exception as e:
    print(f"An unexpected error occurred during report categorization test: {e}")