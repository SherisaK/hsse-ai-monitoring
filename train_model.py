import pandas as pd
import re
import joblib # To save/load your models and vectorizers

# NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn for text vectorization and classification models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC # Another good option for text classification
from sklearn.metrics import accuracy_score, classification_report

# --- NLTK Data Downloads (Run once if necessary) ---
# Ensure 'stopwords' and 'punkt' (and 'punkt_tab' if you had that issue) are downloaded.
# The try-except block here is for robustness, but it's best to run nltk.download()
# manually in a Python interpreter as done previously if you face issues.
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords data not found. Attempting to download...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    # 'punkt' tokenizer is needed for word_tokenize
    # If 'punkt_tab' error re-appears, you might need to run nltk.download('punkt_tab') manually
    word_tokenize("test")
except LookupError:
    print("NLTK punkt tokenizer data not found. Attempting to download...")
    nltk.download('punkt')
    word_tokenize("test") # Test again after download

# --- Text Preprocessing Function (MUST match the one in ai_service_app.py exactly!) ---
def clean_text(text):
    if not isinstance(text, str): # Ensure text is a string
        return ""
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters (keeps numbers)
    words = word_tokenize(text) # Tokenize into words
    # Filter out stop words and ensure tokens are alphabetic (can remove .isalpha() if numbers/symbols are important)
    filtered_words = [w for w in words if w not in stop_words and w.isalpha()]
    return " ".join(filtered_words)

print("--- Starting Model Training Script ---")

# --- Training the "Proactive Monitoring" Model ---
print("\n--- Training Proactive Monitoring Model ---")

# Load data for Proactive Monitoring
try:
    proactive_df = pd.read_csv('proactive_monitoring_data.csv')
    print("Proactive monitoring data loaded successfully.")
    print(f"Initial data shape: {proactive_df.shape}")
    print(proactive_df.head())
except FileNotFoundError:
    print("Error: 'proactive_monitoring_data.csv' not found. Please ensure it's in the same directory.")
    exit(1) # Exit if the file isn't found

# Apply cleaning
proactive_df['cleaned_text'] = proactive_df['text'].apply(clean_text)
print("Text cleaned for proactive monitoring data.")

# Define features (X) and labels (y)
X_proactive = proactive_df['cleaned_text']
y_proactive = proactive_df['is_hsse_related']

# Split data into training and testing sets with robust stratification
try:
    X_train_proactive, X_test_proactive, y_train_proactive, y_test_proactive = train_test_split(
        X_proactive, y_proactive, test_size=0.2, random_state=42, stratify=y_proactive
    )
except ValueError as e:
    print(f"Warning: Could not stratify proactive data split. Error: {e}. Trying without stratification.")
    X_train_proactive, X_test_proactive, y_train_proactive, y_test_proactive = train_test_split(
        X_proactive, y_proactive, test_size=0.2, random_state=42
    )
print(f"Data split: {len(X_train_proactive)} training samples, {len(X_test_proactive)} test samples.")

# Initialize and fit the TF-IDF Vectorizer for proactive model
proactive_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_proactive_vec = proactive_vectorizer.fit_transform(X_train_proactive)
X_test_proactive_vec = proactive_vectorizer.transform(X_test_proactive)
print("Proactive TF-IDF Vectorizer fitted and text transformed.")
print(f"Vectorized training data shape: {X_train_proactive_vec.shape}")

# Initialize and train the Classifier Model for proactive monitoring
proactive_model = LogisticRegression(max_iter=1000, random_state=42)
proactive_model.fit(X_train_proactive_vec, y_train_proactive)
print("Proactive monitoring model trained.")

# Evaluate the proactive model
y_pred_proactive = proactive_model.predict(X_test_proactive_vec)
print("\nProactive Monitoring Model Performance Report:")
print(f"Accuracy: {accuracy_score(y_test_proactive, y_pred_proactive):.4f}")
print(classification_report(y_test_proactive, y_pred_proactive, zero_division=0)) # Added zero_division=0 to suppress warnings
print("Proactive monitoring model and vectorizer saved as .pkl files.")
joblib.dump(proactive_model, 'proactive_monitoring_model.pkl')
joblib.dump(proactive_vectorizer, 'proactive_monitoring_tfidf_vectorizer.pkl')


# --- Training the "Report Categorization" Model ---
print("\n--- Training Report Categorization Model ---")

# Load data for Report Categorization (using the corrected filename)
try:
    # Explicitly define column names with header=0 for robustness
    report_df = pd.read_csv('Report_categorization.csv', names=['report_text', 'category'], header=0)
    print("Report categorization data loaded successfully.")
    print(f"Initial data shape: {report_df.shape}")
    print(report_df.head())
except FileNotFoundError:
    print("Error: 'Report_categorization.csv' not found. Please ensure it's in the same directory.")
    exit(1)

# Apply cleaning
report_df['cleaned_report_text'] = report_df['report_text'].apply(clean_text)
print("Text cleaned for report categorization data.")

# Define features (X) and labels (y)
X_report = report_df['cleaned_report_text']
y_report = report_df['category']

# Split data with robust stratification
try:
    X_train_report, X_test_report, y_train_report, y_test_report = train_test_split(
        X_report, y_report, test_size=0.2, random_state=42, stratify=y_report
    )
except ValueError as e:
    print(f"Warning: Could not stratify report categorization data split. Error: {e}. Trying without stratification.")
    X_train_report, X_test_report, y_train_report, y_test_report = train_test_split(
        X_report, y_report, test_size=0.2, random_state=42
    )
print(f"Data split: {len(X_train_report)} training samples, {len(X_test_report)} test samples.")

# Initialize and fit the TF-IDF Vectorizer for report categorization model
report_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_report_vec = report_vectorizer.fit_transform(X_train_report)
X_test_report_vec = report_vectorizer.transform(X_test_report)
print("Report categorization TF-IDF Vectorizer fitted and text transformed.")
print(f"Vectorized training data shape: {X_train_report_vec.shape}")

# Initialize and train the Classifier Model for report categorization
report_model = LinearSVC(random_state=42, dual=False)
report_model.fit(X_train_report_vec, y_train_report)
print("Report categorization model trained.")

# Evaluate the report categorization model
y_pred_report = report_model.predict(X_test_report_vec)
print("\nReport Categorization Model Performance Report:")
print(f"Accuracy: {accuracy_score(y_test_report, y_pred_report):.4f}")
print(classification_report(y_test_report, y_pred_report, zero_division=0)) # Added zero_division=0
print("Report categorization model and vectorizer saved as .pkl files.")
joblib.dump(report_model, 'report_categorization_model.pkl')
joblib.dump(report_vectorizer, 'report_categorization_tfidf_vectorizer.pkl')


print("\n--- Training Complete ---")