
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load and clean the dataset
df = pd.read_csv(
    "data/raw/BCIT601_starter_service_requests_dataset.csv"
)

df["request_text"] = (
    df["request_text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

X = df["request_text"]
y = df["category"]

# Use the same split for every experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit TF-IDF using training text only
vectoriser = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf = vectoriser.fit_transform(X_train)
X_test_tfidf = vectoriser.transform(X_test)

# Compare different C values
for c_value in [0.1, 1.0, 10.0]:
    model = LogisticRegression(
        C=c_value,
        max_iter=1000
    )

    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 45)
    print(f"Logistic Regression C = {c_value}")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))
