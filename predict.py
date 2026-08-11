import re
import string
import joblib


# ==========================================
# 1. LOAD SAVED MODEL AND VECTORIZER
# ==========================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

model = joblib.load(
    "models/logistic_regression.pkl"
)

print("Model and vectorizer loaded successfully!")


# ==========================================
# 2. TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 3. PREDICT NEWS
# ==========================================

def predict_news(text):

    # Clean the input
    cleaned = clean_text(text)

    # Convert text into TF-IDF features
    vector = vectorizer.transform([cleaned])

    # Make prediction
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"


# ==========================================
# 4. GET USER INPUT
# ==========================================

print("\n==========================================")
print("       FAKE NEWS DETECTION SYSTEM")
print("==========================================")

user_text = input("\nEnter a news headline or article:\n")

result = predict_news(user_text)

print("\nPrediction:", result)