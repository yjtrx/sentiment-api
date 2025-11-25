from flask import Flask, request, jsonify, render_template_string
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

app = Flask(__name__)

# ============== Download NLTK Data on Startup ==============
print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded")
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")

# ============== Text Preprocessor Class ==============
class TextPreprocessor:
    """
    Text preprocessing for traditional NLP
    """
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()

    def clean_text(self, text):
        """
        Clean and preprocess text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Word tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Join words back
        return ' '.join(tokens)

# ============== Load Model ==============
print("Loading model...")

try:
    # Load the saved pipeline (TF-IDF + SVM)
    model = joblib.load('tfidf_svm.pkl')
    print(" Model loaded successfully")
    print(f"  Model type: {type(model)}")
    print(f"  Model classes: {model.classes_}")
    MODEL_LOADED = True
except Exception as e:
    print(f"✗ Error loading model: {e}")
    MODEL_LOADED = False
    model = None

# Initialize preprocessor (matching your training config)
preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)

# Label mapping (since model was trained with integer labels)
LABEL_MAPPING = {0: 'negative', 1: 'neutral', 2: 'positive'}

# ============== HTML Template ==============
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial Sentiment Analysis API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: bold;
        }
        .status.ready { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .endpoint {
            background: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
            border-radius: 5px;
        }
        code {
            background: #272822;
            color: #f8f8f2;
            padding: 15px;
            display: block;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            font-size: 13px;
        }
        .example {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        h3 { color: #667eea; margin-top: 0; }
        .badge { 
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }
        .badge.get { background: #28a745; color: white; }
        .badge.post { background: #007bff; color: white; }
        ul { line-height: 1.8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Financial Sentiment Analysis API</h1>
        
        <div class="status {{ status_class }}">
            {{ status_message }}
        </div>
        
        <p>Analyze sentiment of financial news and reports using TF-IDF + SVM classifier trained on the Financial PhraseBank dataset.</p>
        
        <p><strong>Model Details:</strong></p>
        <ul>
            <li>Algorithm: TF-IDF Vectorizer + Linear SVM</li>
            <li>Training Dataset: Financial PhraseBank (3,453 sentences)</li>
            <li>Classes: Positive, Negative, Neutral</li>
            <li>Accuracy: ~86%</li>
        </ul>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="badge get">GET</span> /</h3>
            <p>API documentation (this page)</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="badge get">GET</span> /health</h3>
            <p>Check service health and model status</p>
            <code>curl {{ base_url }}/health</code>
        </div>
        
        <div class="endpoint">
            <h3><span class="badge post">POST</span> /predict</h3>
            <p>Predict sentiment of financial text</p>
            
            <p><strong>Input Parameters:</strong></p>
            <ul>
                <li><code>text</code> (string, required): Financial text to analyze</li>
            </ul>
            
            <div class="example">
                <strong>Example Request:</strong>
                <code>curl -X POST "{{ base_url }}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "The company reported strong quarterly earnings with revenue up 15%"}'</code>
            </div>
            
            <div class="example">
                <strong>Example Response:</strong>
                <code>{
  "success": true,
  "input_text": "The company reported strong quarterly earnings with revenue up 15%",
  "preprocessed_text": "company reported strong quarterly earnings revenue",
  "sentiment": "positive",
  "confidence": 0.8734,
  "decision_scores": {
    "negative": -1.2345,
    "neutral": -0.5678,
    "positive": 2.1234
  },
  "probabilities": {
    "negative": 0.0532,
    "neutral": 0.1234,
    "positive": 0.8734
  },
  "message": "Positive financial sentiment"
}</code>
            </div>
        </div>
        
        <h2>Python Example</h2>
        <code>import requests
import json

url = "{{ base_url }}/predict"
text = "The company's profits declined sharply this quarter"

response = requests.post(url, json={"text": text})
result = response.json()

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")</code>
        
        <h2>More Examples</h2>
        
        <div class="example">
            <strong>Positive Example:</strong>
            <code>"Operating profit rose to EUR 13.1 mn from EUR 8.7 mn"</code>
        </div>
        
        <div class="example">
            <strong>Negative Example:</strong>
            <code>"The company's revenue decreased by 10% compared to last year"</code>
        </div>
        
        <div class="example">
            <strong>Neutral Example:</strong>
            <code>"The board of directors will meet on Friday to discuss the quarterly report"</code>
        </div>
        
        <h2>Error Handling</h2>
        <ul>
            <li><code>400 Bad Request</code>: Missing or invalid text</li>
            <li><code>500 Internal Server Error</code>: Server-side processing error</li>
            <li><code>503 Service Unavailable</code>: Model not loaded</li>
        </ul>
        
        <h2>Response Fields Explained</h2>
        <ul>
            <li><strong>sentiment</strong>: Predicted class (positive/negative/neutral)</li>
            <li><strong>confidence</strong>: Softmax probability of predicted class (0-1)</li>
            <li><strong>decision_scores</strong>: Raw SVM decision function scores (distance to hyperplane)</li>
            <li><strong>probabilities</strong>: Softmax-normalized probabilities for all classes (sum to 1.0)</li>
        </ul>
    </div>
</body>
</html>
"""

# ============== Routes ==============

@app.route('/')
def home():
    """Home page with API documentation"""
    base_url = request.url_root.rstrip('/')
    
    if MODEL_LOADED:
        status_class = "ready"
        status_message = "Model Loaded - Service Ready"
    else:
        status_class = "error"
        status_message = "Model Not Loaded - Service Unavailable"
    
    return render_template_string(
        HOME_HTML,
        base_url=base_url,
        status_class=status_class,
        status_message=status_message
    )

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "service": "Financial Sentiment Analysis",
        "model": "TF-IDF + Linear SVM",
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0",
        "training_dataset": "Financial PhraseBank",
        "classes": ["positive", "negative", "neutral"],
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/ (GET)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment of financial text"""
    
    # Check if model is loaded
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded",
            "message": "Service is currently unavailable"
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "message": "Request body must be JSON with 'text' field"
            }), 400
        
        text = data.get('text', '')
        
        # Validate input
        if not text or not isinstance(text, str):
            return jsonify({
                "error": "Missing or invalid 'text' field",
                "message": "Please provide text as a string"
            }), 400
        
        if len(text.strip()) == 0:
            return jsonify({
                "error": "Empty text",
                "message": "Text cannot be empty"
            }), 400
        
        if len(text) > 5000:
            return jsonify({
                "error": "Text too long",
                "message": "Text must be less than 5000 characters"
            }), 400
        
        # Preprocess text
        print(f"Processing: {text[:100]}...")
        preprocessed_text = preprocessor.clean_text(text)
        print(f"Preprocessed: '{preprocessed_text}'")
        
        # Make prediction (returns integer: 0, 1, or 2)
        prediction_int = model.predict([preprocessed_text])[0]
        
        # Map integer to sentiment label
        sentiment = LABEL_MAPPING[prediction_int]
        
        # Get decision scores
        decision_scores = model.decision_function([preprocessed_text])[0]
        
        # Create decision scores dict with sentiment labels
        scores_dict = {
            'negative': float(decision_scores[0]),
            'neutral': float(decision_scores[1]),
            'positive': float(decision_scores[2])
        }
        
        # Convert to probabilities using softmax
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        softmax_probs = exp_scores / np.sum(exp_scores)
        confidence = float(np.max(softmax_probs))
        
        prob_dict = {
            'negative': float(softmax_probs[0]),
            'neutral': float(softmax_probs[1]),
            'positive': float(softmax_probs[2])
        }
        
        print(f"Prediction: {sentiment} (confidence: {confidence:.2f})")
        
        # Return response
        return jsonify({
            "success": True,
            "input_text": text,
            "preprocessed_text": preprocessed_text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "message": f"{sentiment.capitalize()} financial sentiment"
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "message": "Failed to process request"
        }), 500

# ============== Run Server ==============
if __name__ == '__main__':
    print("=" * 60)
    print("Starting Financial Sentiment Analysis API")
    print("=" * 60)
    if MODEL_LOADED:
        print("Model: TF-IDF + Linear SVM")
        print(f"Classes: {model.classes_}")
        print("Label Mapping:", LABEL_MAPPING)
    else:
        print("Model not loaded - service will not work!")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
