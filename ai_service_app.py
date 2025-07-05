from flask import Flask, request, jsonify
import joblib # To load your saved models and vectorizers
import re # For text cleaning (punctuation removal)
import nltk # For NLTK functions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global variables for models and vectorizers ---
# These will be loaded once when the Flask app starts
proactive_model = None
proactive_vectorizer = None
report_category_model = None
report_category_vectorizer = None
stop_words = None

# --- Text Preprocessing Function (MUST match the one in train_models.py exactly!) ---
def clean_text(text):
    if not isinstance(text, str): # Ensure text is a string
        return ""
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters
    words = word_tokenize(text) # Tokenize into words
    filtered_words = [w for w in words if w not in stop_words and w.isalpha()] # Remove stop words and non-alphabetic tokens
    return " ".join(filtered_words)

# --- Function to load models and vectorizers ---
# This will be called once when the Flask app starts
def load_models():
    global proactive_model, proactive_vectorizer, report_category_model, report_category_vectorizer, stop_words
    try:
        print("Loading AI models and vectorizers...")
        proactive_model = joblib.load('proactive_monitoring_model.pkl')
        proactive_vectorizer = joblib.load('proactive_monitoring_tfidf_vectorizer.pkl')
        report_category_model = joblib.load('report_categorization_model.pkl')
        report_category_vectorizer = joblib.load('report_categorization_tfidf_vectorizer.pkl')
        stop_words = set(stopwords.words('english')) # Load stop words once

        print("AI models and vectorizers loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}. Make sure .pkl files are in the same directory.")
        # Exit or raise an error to prevent the app from starting without models
        exit(1) # Exit with an error code

# --- Define API Endpoints ---

@app.route('/predict_hsse_relevance', methods=['POST'])
def predict_hsse_relevance():
    """
    Receives text and predicts if it's HSSE-related.
    Expected JSON input: {"text": "your text string here"}
    Returns JSON: {"is_hsse_related": true/false}
    """
    data = request.json
    text_to_analyze = data.get('text', '')

    if not text_to_analyze:
        return jsonify({"error": "No text provided"}), 400

    # Clean and vectorize the input text
    cleaned_text = clean_text(text_to_analyze)
    vectorized_text = proactive_vectorizer.transform([cleaned_text]) # transform expects a list

    # Make prediction
    prediction = proactive_model.predict(vectorized_text)[0] # Get the first (and only) prediction

    # Convert prediction to boolean if your labels were 0/1, otherwise return as is
    # Assuming your proactive_monitoring_data.csv has 0/1 for is_hsse_related
    is_hsse_related_bool = bool(prediction)

    return jsonify({"is_hsse_related": is_hsse_related_bool})

@app.route('/predict_report_category', methods=['POST'])
def predict_report_category():
    """
    Receives a report text and predicts its HSSE category.
    Expected JSON input: {"text": "your report text string here"}
    Returns JSON: {"predicted_category": "Category Name"}
    """
    data = request.json
    report_text = data.get('text', '')

    if not report_text:
        return jsonify({"error": "No text provided"}), 400

    # Clean and vectorize the input text
    cleaned_text = clean_text(report_text)
    vectorized_text = report_category_vectorizer.transform([cleaned_text]) # transform expects a list

    # Make prediction
    prediction = report_category_model.predict(vectorized_text)[0] # Get the first (and only) prediction

    return jsonify({"predicted_category": prediction})

# --- Run the Flask App ---
if __name__ == '__main__':
    # Load models before running the app
    load_models()

    print("Flask AI Service starting...")
    # This runs the development server. For production, use Gunicorn/Nginx or cloud services.
    app.run(debug=True, host='0.0.0.0', port=5000)
    # host='0.0.0.0' makes it accessible from other machines on your network (useful for testing)
    # port=5000 is the default port, change if it conflicts
    # debug=True allows for auto-reloading on code changes and provides more detailed errors