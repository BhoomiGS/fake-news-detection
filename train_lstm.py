import pandas as pd
import re
import string
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    SpatialDropout1D,
    Bidirectional
)

from tensorflow.keras.callbacks import EarlyStopping


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
# 5. SHUFFLE DATA
# ==========================================

combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 6. REMOVE MISSING VALUES
# ==========================================

combined_df.dropna(
    subset=["full_text", "label"],
    inplace=True
)


# ==========================================
# 7. TEXT CLEANING
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

print("Text cleaning done!")


# ==========================================
# 8. PREPARE DATA
# ==========================================

X = combined_df["clean_text"].values
y = combined_df["label"].values


# ==========================================
# 9. LSTM PARAMETERS
# ==========================================

VOCAB_SIZE = 10000
MAX_LENGTH = 250
EMBEDDING_DIM = 128


# ==========================================
# 10. TOKENIZATION
# ==========================================

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X)

X_sequences = tokenizer.texts_to_sequences(X)


# ==========================================
# 11. PADDING
# ==========================================

X_padded = pad_sequences(
    X_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Padded data shape:", X_padded.shape)


# ==========================================
# 12. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_padded,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 13. BUILD BIDIRECTIONAL LSTM
# ==========================================

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    )
)

model.add(
    SpatialDropout1D(0.3)
)

model.add(
    Bidirectional(
        LSTM(
            64,
            dropout=0.3,
            recurrent_dropout=0.3
        )
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ==========================================
# 14. COMPILE MODEL
# ==========================================

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 15. TRAIN MODEL
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)


# ==========================================
# 16. EVALUATE MODEL
# ==========================================

print("\nEvaluating LSTM...")

y_pred_probs = model.predict(X_test)

y_pred = (
    y_pred_probs > 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nLSTM Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Real (0)",
            "Fake (1)"
        ]
    )
)


# ==========================================
# 17. SAVE LSTM MODEL
# ==========================================

model.save(
    "models/lstm_model.keras"
)

print("\nLSTM model saved successfully!")


# ==========================================
# 18. SAVE TOKENIZER
# ==========================================

import pickle

with open(
    "models/lstm_tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )

print("LSTM tokenizer saved successfully!")