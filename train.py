import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
import joblib
# ==========================================
# 1. LOAD DATASET
# ==========================================

true_file_path = "dataset/True.csv"
fake_file_path = "dataset/Fake.csv"

df_true = pd.read_csv(true_file_path)
df_fake = pd.read_csv(fake_file_path)

print("Files loaded successfully!")


# ==========================================
# 2. ASSIGN LABELS
# ==========================================

# 0 = Real News
# 1 = Fake News

df_true["label"] = 0
df_fake["label"] = 1


# ==========================================
# 3. COMBINE DATASETS
# ==========================================

combined_df = pd.concat(
    [df_true, df_fake],
    ignore_index=True
)


# ==========================================
# 4. COMBINE TITLE AND TEXT
# ==========================================

combined_df["full_text"] = (
    combined_df["title"] + " " + combined_df["text"]
)

combined_df = combined_df[["full_text", "label"]]


# ==========================================
# 5. SHUFFLE DATASET
# ==========================================

combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 6. DISPLAY DATASET INFORMATION
# ==========================================

print("\n--- Dataset is ready! ---")
print("Total rows:", len(combined_df))

print("\nClass Balance:")
print(combined_df["label"].value_counts())


# ==========================================
# 7. REMOVE MISSING VALUES
# ==========================================

combined_df.dropna(
    subset=["full_text", "label"],
    inplace=True
)


# ==========================================
# 8. TEXT CLEANING
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


combined_df["clean_text"] = combined_df["full_text"].apply(
    clean_text
)

print("\nText cleaning done!")


# ==========================================
# 9. PREPARE X AND Y
# ==========================================

X = combined_df["clean_text"].values
y = combined_df["label"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# 10. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))

# ==========================================
# 11. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

print("\nTF-IDF Vectorization complete!")
# Save TF-IDF vectorizer
joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("TF-IDF vectorizer saved!")

print("Training data shape:", X_train_vec.shape)
print("Testing data shape:", X_test_vec.shape)

# ==========================================
# 13. TRAIN AND EVALUATE MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Naive Bayes": MultinomialNB(),

    "SVM": SVC(kernel="linear"),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}
results = {}

for name, model in models.items():

    print(f"\n--- Training {name} ---")

    # Train model
    model.fit(X_train_vec, y_train)

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        f"models/{filename}"
    )

    print(f"{name} saved!")

    # Make predictions
    y_pred = model.predict(X_test_vec)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Real (0)", "Fake (1)"]
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Real (0)", "Fake (1)"]
        )
    )