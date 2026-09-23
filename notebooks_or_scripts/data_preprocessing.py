import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the raw service request dataset
file_path = "data/raw/BCIT601_starter_service_requests_dataset.csv"

df = pd.read_csv(file_path)

# Display the first five rows
print("First five rows of the dataset:")
print(df.head())

# Display the dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column information
print("\nDataset information:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Display category distribution
print("\nCategory distribution:")
print(df["category"].value_counts())

# Clean the request text
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = " ".join(text.split())

    return text


df["request_text"] = df["request_text"].apply(clean_text)

# Display original-style cleaned examples
print("\nCleaned request text examples:")
print(df["request_text"].head())


# Save the cleaned dataset
processed_file_path = "data/processed/service_requests_processed.csv"

df.to_csv(processed_file_path, index=False)

print(f"\nProcessed dataset saved to: {processed_file_path}")

# Separate input text and target category
X = df["request_text"]
y = df["category"]

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set size:")
print(len(X_train))

print("\nTesting set size:")
print(len(X_test))


# Create the TF-IDF vectoriser
vectoriser = TfidfVectorizer()

# Learn vocabulary from the training data and transform it
X_train_tfidf = vectoriser.fit_transform(X_train)

# Transform the testing data using the same vocabulary
X_test_tfidf = vectoriser.transform(X_test)

print("\nTF-IDF training data shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF testing data shape:")
print(X_test_tfidf.shape)


# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model using the TF-IDF training data
model.fit(X_train_tfidf, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions on the testing data
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel accuracy:")
print(f"{accuracy:.2%}")

# Display the classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Save the trained Logistic Regression model
model_path = "model/service_request_model.joblib"
joblib.dump(model, model_path)

# Save the TF-IDF vectoriser
vectoriser_path = "model/tfidf_vectoriser.joblib"
joblib.dump(vectoriser, vectoriser_path)

print(f"\nModel saved to: {model_path}")
print(f"TF-IDF vectoriser saved to: {vectoriser_path}")