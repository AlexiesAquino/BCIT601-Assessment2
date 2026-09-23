# AI-Powered Local City Council Service Request Classification System

## Project Overview

This project develops an AI-powered service request classification system for a Local City Council. The system uses machine learning to automatically classify public service requests into four categories: road, waste, water, and safety.

The system processes service request text using text preprocessing and TF-IDF vectorisation before applying a Logistic Regression classification model. The trained model is integrated into a Flask API that allows users to submit service request text and receive a predicted category and confidence score.

The project aims to support council staff by providing an automated initial classification of incoming service requests, helping organise requests more efficiently while keeping human review involved in the decision-making process.

## Dataset

The project uses the provided BCIT601 starter service requests dataset. The dataset contains 80 service request records and six fields:

- `request_id` – unique identifier for each service request
- `request_text` – text submitted by the public
- `category` – target label for machine learning classification
- `suburb` – location information
- `priority` – priority level of the request
- `date_reported` – date the request was reported

The dataset contains four service request categories: road, waste, water, and safety, with 20 records in each category.

Initial data inspection found no missing values and no duplicate records. The request text was cleaned by converting the text to lowercase and removing unnecessary leading and trailing whitespace.

The processed dataset was saved as:

`data/processed/service_requests_processed.csv`


## Machine Learning Approach and Results

The project uses a supervised machine learning approach to classify service request text. The cleaned `request_text` data was converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) vectorisation.

The dataset was divided into 80% training data and 20% testing data. This resulted in 64 training records and 16 testing records. A Logistic Regression model was trained using the TF-IDF features to predict the service request category.

The trained model achieved an accuracy of 87.50% on the testing dataset. The classification report showed the following results:

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Road     | 1.00      | 0.75   | 0.86     |
| Safety   | 0.67      | 1.00   | 0.80     |
| Waste    | 1.00      | 0.75   | 0.86     |
| Water    | 1.00      | 1.00   | 1.00     |

The trained Logistic Regression model and TF-IDF vectoriser were saved using Joblib so they can be loaded by the Flask API.


## API Design and Endpoints

A Flask API was developed to provide access to the trained machine learning model. The API loads the saved Logistic Regression model and TF-IDF vectoriser and provides three main endpoints.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Checks whether the API is running |
| `/categories` | GET | Returns the available service request categories |
| `/predict` | POST | Accepts service request text and returns a predicted category and confidence score |

The `/predict` endpoint accepts a JSON request containing `request_text`. The text is cleaned and transformed using the saved TF-IDF vectoriser before being passed to the trained model. The API returns the predicted category and the highest class probability as the confidence score.
The API also validates the input. If `request_text` is missing or empty, the API returns an HTTP 400 error with an appropriate error message.


## Testing and Results

The API was tested using valid and invalid service request inputs. Five valid `/predict` requests and two invalid requests were tested.

### Valid Prediction Tests

| Test | Input | Predicted Category | Confidence | Result |
|------|-------|--------------------|------------|--------|
| 1 | Large pothole on the road | road | 0.34 | Expected |
| 2 | Rubbish bin was not collected | waste | 0.46 | Expected |
| 3 | Water leak from a pipe | water | 0.63 | Expected |
| 4 | Suspicious person near community centre | water | 0.28 | Unexpected |
| 5 | Deep crack in footpath | road | 0.35 | Expected |

The fifth valid test requests a road-related service and was correctly classified as `road`. Test 4 produced an unexpected prediction because the safety-related request was classified as `water` with a confidence of 0.28. This demonstrates that the model may produce incorrect classifications for some requests and that human review remains important.

### Invalid Input Tests

| Test | Input | HTTP Status | Result |
|------|-------|-------------|--------|
| 6 | Empty `request_text` | 400 | Correctly rejected |
| 7 | Missing `request_text` | 400 | Correctly rejected |

Both invalid requests were handled correctly by the API. Instead of attempting to make a prediction, the API returned an HTTP 400 error with the message `request_text is required`.


## Project Structure

The project is organised into separate folders for the API, data, machine learning model, scripts, outputs, and tests.

```text
BCIT601_Assessment2/
├── api/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── BCIT601_starter_service_requests_dataset.csv
│   └── processed/
│       └── service_requests_processed.csv
├── model/
│   ├── service_request_model.joblib
│   └── tfidf_vectoriser.joblib
├── notebooks_or_scripts/
│   └── data_preprocessing.py
├── outputs/
├── tests/
├── README.md
└── requirements.txt

## How to Run the Project

### 1. Create and activate the virtual environment

From the project root directory, create the virtual environment:

```powershell
python -m venv .venv

Activate it in PowerShell: .\.venv\Scripts\Activate.ps1
Install the project dependencies using: pip install -r requirements.txt
Start the API using: python api/app.py
The API will run locally at: http://127.0.0.1:5000

The following endpoints can be accessed: 
GET  http://127.0.0.1:5000/health
GET  http://127.0.0.1:5000/categories
POST http://127.0.0.1:5000/predict