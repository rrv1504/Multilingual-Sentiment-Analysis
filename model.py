import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sentiment_utils import clean_text

# ---------------- LOAD DATA ----------------
df = pd.read_csv("amazon_style_reviews.csv")

print("Columns:", df.columns)

# Use required columns
df = df[["review", "label"]]

# Clean labels and keep only supported classes
df["label"] = df["label"].astype(str).str.lower().str.strip()
df = df[df["label"].isin(["positive", "negative"])].copy()

print("\nLabel Distribution:")
print(df["label"].value_counts())


# ---------------- CLEAN TEXT ----------------
df["cleaned"] = df["review"].apply(clean_text)

# Duplicate reviews can leak into both train and test sets and create
# unrealistically high accuracy, so keep only unique cleaned reviews.
before_dedup = len(df)
df = df.drop_duplicates(subset=["cleaned"]).copy()

print(f"\nRows before deduplication: {before_dedup}")
print(f"Rows after deduplication: {len(df)}")
print("\nLabel Distribution After Deduplication:")
print(df["label"].value_counts())


# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)


# ---------------- VECTORIZATION ----------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Feature size:", X_train_vec.shape)


# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=200, class_weight="balanced")
model.fit(X_train_vec, y_train)


# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))


# ---------------- SAVE ----------------
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel trained and saved successfully!")
