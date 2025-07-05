import os
import requests
import json
import time
from datetime import datetime, timezone

# For environment variables (make sure you've installed: pip install python-dotenv)
from dotenv import load_dotenv

# For Supabase integration (make sure you've installed: pip install supabase)
from supabase import create_client, Client

# --- Load environment variables from key.env file ---
# This line tells dotenv to look for 'key.env' specifically
load_dotenv('key.env')

# --- Supabase Configuration ---
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY: str = os.environ.get("SUPABASE_SERVICE_KEY") # This should be your service_role key

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Supabase URL or Service Key environment variables not set in key.env or system.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- AI Service Configuration ---
AI_SERVICE_URL = "http://127.0.0.1:5000" # Your running Flask AI service

# --- NewsAPI Configuration ---
NEWSAPI_KEY: str = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise ValueError("NEWSAPI_KEY environment variable not set in key.env.")

def get_current_utc_iso():
    """Returns current UTC time in ISO 8601 format for Supabase."""
    return datetime.now(timezone.utc).isoformat()

def fetch_news_articles(query="Guyana HSSE OR 'workplace accident Guyana' OR 'environmental incident Guyana'", language="en", page_size=20):
    """
    Fetches news articles from NewsAPI based on a query.
    Filters for relevant keywords and language.
    """
    print(f"Fetching news articles for query: '{query}'...")
    articles_data = []
    # Using 'everything' endpoint for broader search
    # You can customize the query to be more specific to your needs
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={NEWSAPI_KEY}"

    try:
        response = requests.get(url, timeout=10) # 10-second timeout for the API call
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if data["status"] == "ok":
            for article in data["articles"]:
                # Basic filtering to ensure essential fields exist
                if article.get('title') and article.get('description') and article.get('publishedAt'):
                    articles_data.append({
                        "source": article['source']['name'],
                        "original_post_url": article['url'],
                        "text": f"{article['title']}. {article['description']}", # Combine title and description
                        "post_date": article['publishedAt'] # Already in ISO format from NewsAPI
                    })
            print(f"Successfully fetched {len(articles_data)} articles.")
        else:
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")

    except requests.exceptions.Timeout:
        print("NewsAPI request timed out.")
    except requests.exceptions.ConnectionError:
        print("Could not connect to NewsAPI. Check internet connection or API endpoint.")
    except requests.exceptions.HTTPError as err:
        print(f"NewsAPI HTTP error: {err} - {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred during NewsAPI fetch: {e}")

    return articles_data

async def process_social_media_incident():
    """
    Fetches data from NewsAPI, sends to AI, and inserts into Supabase if an incident is detected.
    """
    # Use the actual news fetching function here
    data_to_analyze = fetch_news_articles()
    
    if not data_to_analyze:
        print("No articles to analyze. Exiting.")
        return

    detection_time = get_current_utc_iso()

    for item in data_to_analyze:
        text_to_send = item["text"]
        source = item["source"]
        original_url = item.get("original_post_url")
        post_date = item.get("post_date")

        print(f"\nProcessing from {source}: '{text_to_send[:70]}...'")

        # 1. Call your AI Service for HSSE relevance
        try:
            ai_relevance_response = requests.post(
                f"{AI_SERVICE_URL}/predict_hsse_relevance",
                json={"text": text_to_send},
                timeout=5 # Added timeout for AI service call
            )
            ai_relevance_response.raise_for_status() # Check for HTTP errors from Flask app
            is_hsse_related = ai_relevance_response.json().get("is_hsse_related")

            if not is_hsse_related:
                print("  -> Not HSSE-related. Skipping.")
                continue # Move to the next item

            print("  -> AI detected as HSSE-related. Getting category...")

            # 2. Call your AI Service for Report Categorization
            ai_category_response = requests.post(
                f"{AI_SERVICE_URL}/predict_report_category",
                json={"text": text_to_send},
                timeout=5 # Added timeout for AI service call
            )
            ai_category_response.raise_for_status() # Check for HTTP errors from Flask app
            predicted_category = ai_category_response.json().get("predicted_category")

            print(f"  -> AI predicted category: {predicted_category}")

            # 3. Prepare data for Supabase insertion
            supabase_data = {
                "source": source,
                "original_post_url": original_url,
                "incident_text": text_to_send,
                "ai_predicted_category": predicted_category,
                "social_media_post_date": post_date, # Use the date from the news article/social media post
                "detection_date": detection_time, # When your script ran
                "status": "pending_review" # Default status, requires human review
            }

            # 4. Insert into Supabase
            print("  -> Inserting into Supabase...")
            try:
                response = supabase.table("social_media_incidents").insert(supabase_data).execute()

                # Check if the insert was successful
                if response.data:
                    print(f"  Successfully inserted incident into Supabase (ID: {response.data[0]['id']}).")
                else:
                    # This branch means execute() completed but data is empty, usually if error is in response.error
                    if response.error:
                        print(f"  Error inserting into Supabase: {response.error.message} (Code: {response.error.code})")
                        print(f"  Details: {response.error.details}")
                    else:
                        print("  Supabase insert completed, but no data returned and no explicit error message.")

            except Exception as e:
                print(f"  An unexpected error occurred during Supabase insertion: {e}")
                if hasattr(e, 'message'): print(f"  Supabase error message: {e.message}")
                if hasattr(e, 'code'): print(f"  Supabase error code: {e.code}")
                if hasattr(e, 'details'): print(f"  Supabase error details: {e.details}")


        except requests.exceptions.ConnectionError:
            print(f"  Error: Could not connect to AI service at {AI_SERVICE_URL}. Is it running?")
            break # Stop processing if AI service is down
        except requests.exceptions.HTTPError as err:
            # This catches HTTP errors from the AI service (e.g., 400 if no text, 500 if internal error)
            print(f"  HTTP error occurred with AI service: {err} - {ai_relevance_response.text if 'ai_relevance_response' in locals() else 'No response'}")
        except Exception as e:
            print(f"  An unexpected error occurred during processing: {e}")

        time.sleep(1) # Be kind to APIs and give your Flask app a breather

if __name__ == "__main__":
    import asyncio
    asyncio.run(process_social_media_incident())
    print("\nProactive monitoring run complete.")