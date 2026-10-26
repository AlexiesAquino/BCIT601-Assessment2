
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load the service request dataset
df = pd.read_csv(
    "data/raw/BCIT601_starter_service_requests_dataset.csv"
)

# Clean the request text
df["request_text"] = (
    df["request_text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

# Define input text and target categories
X = df["request_text"]
y = df["category"]

# Use the same split settings for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Experiment 1: Unigrams (individual words)
baseline_vectoriser = TfidfVectorizer(ngram_range=(1, 1))

X_train_baseline = baseline_vectoriser.fit_transform(X_train)
X_test_baseline = baseline_vectoriser.transform(X_test)

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_baseline, y_train)

baseline_predictions = baseline_model.predict(X_test_baseline)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)


# Experiment 2: Unigrams and bigrams
improved_vectoriser = TfidfVectorizer(ngram_range=(1, 2))

X_train_improved = improved_vectoriser.fit_transform(X_train)
X_test_improved = improved_vectoriser.transform(X_test)

improved_model = LogisticRegression(max_iter=1000)
improved_model.fit(X_train_improved, y_train)

improved_predictions = improved_model.predict(X_test_improved)

improved_accuracy = accuracy_score(
    y_test, improved_predictions
)


# Compare the results
print("BASELINE MODEL: Unigrams only")
print(f"Accuracy: {baseline_accuracy:.2%}")
print(classification_report(y_test, baseline_predictions, zero_division=0))

print("\nIMPROVED EXPERIMENT: Unigrams + Bigrams")
print(f"Accuracy: {improved_accuracy:.2%}")
print(classification_report(y_test, improved_predictions, zero_division=0))

print("\nACCURACY COMPARISON")
print(f"Baseline: {baseline_accuracy:.2%}")
print(f"Experiment: {improved_accuracy:.2%}")
print(f"Difference: {(improved_accuracy - baseline_accuracy) * 100:+.2f} percentage points")
