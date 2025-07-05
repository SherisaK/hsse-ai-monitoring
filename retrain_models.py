# retrain_models.py
import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv('key.env') # Load environment variables

# --- Supabase Configuration ---
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY: str = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Supabase URL or Service Key environment variables not set in .env or system.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- NLTK Data (ensure downloaded as in train_models.py) ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    filtered_words = [w for w in words if w not in stop_words and w.isalpha()]
    return " ".join(filtered_words)

async def fetch_feedback_data():
    """Fetches human-reviewed data from Supabase for retraining."""
    print("Fetching human-reviewed data from Supabase...")
    
    # Fetch confirmed incidents and false positives for proactive monitoring
    proactive_response_confirmed = supabase.table("social_media_incidents").select("incident_text, status").eq("status", "confirmed").execute()
    proactive_response_false = supabase.table("social_media_incidents").select("incident_text, status").eq("status", "false_positive").execute()

    proactive_data = []
    if proactive_response_confirmed.data:
        for item in proactive_response_confirmed.data:
            proactive_data.append({"text": item["incident_text"], "is_hsse_related": 1}) # 1 for confirmed
    if proactive_response_false.data:
        for item in proactive_response_false.data:
            proactive_data.append({"text": item["incident_text"], "is_hsse_related": 0}) # 0 for false positive
    
    # Fetch categorized incidents (either AI predicted & confirmed, or human corrected)
    # This assumes 'human_corrected_category' takes precedence, otherwise use 'ai_predicted_category'
    categorization_response = supabase.table("social_media_incidents").select("incident_text, ai_predicted_category, human_corrected_category, status").eq("status", "confirmed").execute()
    
    categorization_data = []
    if categorization_response.data:
        for item in categorization_response.data:
            final_category = item["human_corrected_category"] if item["human_corrected_category"] else item["ai_predicted_category"]
            if final_category: # Ensure category is not None
                categorization_data.append({"report_text": item["incident_text"], "category": final_category})

    print(f"Fetched {len(proactive_data)} proactive feedback examples.")
    print(f"Fetched {len(categorization_data)} categorization feedback examples.")

    return pd.DataFrame(proactive_data), pd.DataFrame(categorization_data)

async def retrain_models():
    print("--- Starting Model Retraining Script ---")

    proactive_feedback_df, categorization_feedback_df = await fetch_feedback_data()

    # --- Retraining Proactive Monitoring Model ---
    print("\n--- Retraining Proactive Monitoring Model ---")
    if proactive_feedback_df.empty:
        print("No proactive feedback data available for retraining. Skipping.")
    else:
        proactive_feedback_df['cleaned_text'] = proactive_feedback_df['text'].apply(clean_text)
        X_proactive = proactive_feedback_df['cleaned_text']
        y_proactive = proactive_feedback_df['is_hsse_related']

        # You might want to combine this with your original training data if you keep adding more
        # For simplicity here, we're retraining only on feedback, which might be small.
        # In production, you'd merge this with your existing large dataset.

        try:
            X_train_proactive, X_test_proactive, y_train_proactive, y_test_proactive = train_test_split(
                X_proactive, y_proactive, test_size=0.2, random_state=42, stratify=y_proactive
            )
        except ValueError as e:
            print(f"Warning: Could not stratify proactive data split for retraining. Error: {e}. Trying without stratification.")
            X_train_proactive, X_test_proactive, y_train_proactive, y_test_proactive = train_test_split(
                X_proactive, y_proactive, test_size=0.2, random_state=42
            )

        proactive_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_proactive_vec = proactive_vectorizer.fit_transform(X_train_proactive)
        X_test_proactive_vec = proactive_vectorizer.transform(X_test_proactive)

        proactive_model = LogisticRegression(max_iter=1000, random_state=42)
        proactive_model.fit(X_train_proactive_vec, y_train_proactive)
        
        y_pred_proactive = proactive_model.predict(X_test_proactive_vec)
        print("\nProactive Monitoring Model Retraining Performance Report:")
        print(f"Accuracy: {accuracy_score(y_test_proactive, y_pred_proactive):.4f}")
        print(classification_report(y_test_proactive, y_pred_proactive, zero_division=0))

        joblib.dump(proactive_model, 'proactive_monitoring_model.pkl')
        joblib.dump(proactive_vectorizer, 'proactive_monitoring_tfidf_vectorizer.pkl')
        print("Proactive monitoring model and vectorizer saved as .pkl files.")

    # --- Retraining Report Categorization Model ---
    print("\n--- Retraining Report Categorization Model ---")
    if categorization_feedback_df.empty:
        print("No categorization feedback data available for retraining. Skipping.")
    else:
        categorization_feedback_df['cleaned_report_text'] = categorization_feedback_df['report_text'].apply(clean_text)
        X_report = categorization_feedback_df['cleaned_report_text']
        y_report = categorization_feedback_df['category']

        try:
            X_train_report, X_test_report, y_train_report, y_test_report = train_test_split(
                X_report, y_report, test_size=0.2, random_state=42, stratify=y_report
            )
        except ValueError as e:
            print(f"Warning: Could not stratify report categorization data split for retraining. Error: {e}. Trying without stratification.")
            X_train_report, X_test_report, y_train_report, y_test_report = train_test_split(
                X_report, y_report, test_size=0.2, random_state=42
            )

        report_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_report_vec = report_vectorizer.fit_transform(X_train_report)
        X_test_report_vec = report_vectorizer.transform(X_test_report)

        report_model = LinearSVC(random_state=42, dual=False)
        report_model.fit(X_train_report_vec, y_train_report)

        y_pred_report = report_model.predict(X_test_report_vec)
        print("\nReport Categorization Model Retraining Performance Report:")
        print(f"Accuracy: {accuracy_score(y_test_report, y_pred_report):.4f}")
        print(classification_report(y_test_report, y_pred_report, zero_division=0))

        joblib.dump(report_model, 'report_categorization_model.pkl')
        joblib.dump(report_vectorizer, 'report_categorization_tfidf_vectorizer.pkl')
        print("Report categorization model and vectorizer saved as .pkl files.")

    print("\n--- Retraining Complete ---")

if __name__ == "__main__":
    import asyncio
    asyncio.run(retrain_models())