import streamlit as st
import re
import string
import joblib
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_all_models():

    # -----------------------------
    # Classical ML model
    # -----------------------------

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    logistic_model = joblib.load(
        "models/logistic_regression.pkl"
    )


    # -----------------------------
    # LSTM model
    # -----------------------------

    lstm_model = load_model(
        "models/lstm_model.keras"
    )


    # -----------------------------
    # LSTM tokenizer
    # -----------------------------

    with open(
        "models/lstm_tokenizer.pkl",
        "rb"
    ) as file:

        lstm_tokenizer = pickle.load(file)


    return (
        vectorizer,
        logistic_model,
        lstm_model,
        lstm_tokenizer
    )


# Load everything
(
    vectorizer,
    logistic_model,
    lstm_model,
    lstm_tokenizer
) = load_all_models()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOGISTIC REGRESSION PREDICTION
# ============================================================

def predict_logistic(text):

    # Clean input
    cleaned = clean_text(text)

    # Convert to TF-IDF
    vector = vectorizer.transform(
        [cleaned]
    )

    # Predict
    prediction = logistic_model.predict(
        vector
    )[0]

    return int(prediction)


# ============================================================
# LSTM PREDICTION
# ============================================================

def predict_lstm(text):

    # Clean input
    cleaned = clean_text(text)

    # Convert text into sequences
    sequence = lstm_tokenizer.texts_to_sequences(
        [cleaned]
    )

    # Pad sequence to length 250
    padded = pad_sequences(
        sequence,
        maxlen=250,
        padding="post",
        truncating="post"
    )

    # Get probability
    probability = lstm_model.predict(
        padded,
        verbose=0
    )[0][0]

    # Convert probability into class
    prediction = int(
        probability > 0.5
    )

    return prediction, float(probability)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📰 Fake News Detection")

st.markdown(
    """
    ### Detect whether a news article is likely to be real or fake

    Enter a news headline or article below and the
    machine learning and deep learning models will
    classify it.
    """
)

st.divider()


# ============================================================
# NEWS INPUT
# ============================================================

news_text = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder="Paste a news headline or article here..."
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔍 Predict News",
    use_container_width=True
):

    # Check empty input
    if news_text.strip() == "":

        st.warning(
            "⚠️ Please enter a news article first."
        )

    else:

        # --------------------------------------
        # Logistic Regression prediction
        # --------------------------------------

        logistic_prediction = predict_logistic(
            news_text
        )


        # --------------------------------------
        # LSTM prediction
        # --------------------------------------

        lstm_prediction, lstm_probability = predict_lstm(
            news_text
        )


        # --------------------------------------
        # Display results
        # --------------------------------------

        st.divider()

        st.subheader("🔎 Prediction Results")


        # Logistic Regression
        if logistic_prediction == 1:

            st.error(
                "🚨 Logistic Regression: FAKE NEWS"
            )

        else:

            st.success(
                "✅ Logistic Regression: REAL NEWS"
            )


        # LSTM
        if lstm_prediction == 1:

            st.error(
                "🚨 LSTM: FAKE NEWS"
            )

        else:

            st.success(
                "✅ LSTM: REAL NEWS"
            )


        # --------------------------------------
        # LSTM probability
        # --------------------------------------

        st.write(
            f"**LSTM Fake News Probability:** "
            f"{lstm_probability * 100:.2f}%"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader("📊 Model Performance")

st.write(
    "Accuracy comparison of the models evaluated "
    "on the test dataset."
)


# Accuracies from your model evaluation
model_accuracies = {

    "Logistic Regression": 0.9873,

    "Naive Bayes": 0.9345,

    "SVM": 0.9939,

    "Random Forest": 0.9967,

    "LSTM": 0.9984
}


# Display accuracy
for model_name, accuracy in model_accuracies.items():

    st.write(
        f"**{model_name}** — "
        f"{accuracy * 100:.2f}%"
    )

    st.progress(
        accuracy
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ About the Project")

st.write(
    """
    This project detects fake and real news using
    Natural Language Processing and Machine Learning.

    The system compares classical machine learning
    models with a Bidirectional LSTM deep learning model.

    Models evaluated:

    • Logistic Regression
    • Naive Bayes
    • SVM
    • Random Forest
    • Bidirectional LSTM

    Text preprocessing includes URL removal,
    number removal, punctuation removal and
    whitespace normalization.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake News Detection | "
    "TF-IDF + Machine Learning + Bidirectional LSTM"
)