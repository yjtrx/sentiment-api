import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✓ NLTK data downloaded")
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
    print("✓ Model loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    print(f"✗ Error loading model: {e}")
    MODEL_LOADED = False
    model = None

# Initialize preprocessor (matching your training config)
preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=True)

# ============== Prediction Function ==============
def predict(text, preprocessor):
    """Predict sentiment of financial text"""
    
    if not MODEL_LOADED:
        print("Model not loaded!")
        return None
    
    # Preprocess text
    print(f"\nProcessing: {text}")
    preprocessed_text = preprocessor.clean_text(text)
    print(f"Preprocessed: {preprocessed_text}")
    
    # Make prediction
    prediction = model.predict([preprocessed_text])[0]
    print(f"Prediction: {prediction}")
    
    # Get decision function scores
    decision_scores = model.decision_function([preprocessed_text])[0]  # FIX: Use preprocessed_text, not text
    
    # Get class labels
    classes = model.classes_  # FIX: Get classes from model
    
    # Create scores dict
    scores_dict = {classes[i]: float(decision_scores[i]) for i in range(len(classes))}
    
    # Convert to probabilities using softmax
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    softmax_probs = exp_scores / np.sum(exp_scores)
    confidence = float(np.max(softmax_probs))
    
    prob_dict = {classes[i]: float(softmax_probs[i]) for i in range(len(classes))}
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"{'='*50}")
    print(f"Sentiment: {label_mapping[prediction]}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nDecision Scores:")
    for cls, score in scores_dict.items():
        print(f"  {cls}: {score:.4f}")
    print(f"\nProbabilities:")
    for cls, prob in prob_dict.items():
        print(f"  {cls}: {prob:.4f} ({prob*100:.2f}%)")
    print(f"{'='*50}\n")
    
    return {
        'sentiment': prediction,
        'confidence': confidence,
        'decision_scores': scores_dict,
        'probabilities': prob_dict
    }

# ============== Run Tests ==============
if __name__ == "__main__":
    print("\n🧪 Testing Model Locally (Direct Model Access)\n")
    
    # Test examples
    test_cases = [
        "The company rose to 10% profits",
        "The company reported strong quarterly earnings with revenue up 15%",
        "The company faced significant losses and declining market share",
        "The board of directors will meet on Friday"
    ]
    
    for text in test_cases:
        result = predict(text, preprocessor)
        print()  # Extra spacing between tests