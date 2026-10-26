
import requests

BASE_URL = "http://127.0.0.1:5000"


# Test 1: Health endpoint
response = requests.get(f"{BASE_URL}/health")
assert response.status_code == 200
assert response.json() == {"status": "ok"}


# Test 2: Categories endpoint
response = requests.get(f"{BASE_URL}/categories")
assert response.status_code == 200
assert response.json() == {
    "categories": ["road", "safety", "waste", "water"]
}


# Test 3: Valid prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"request_text": "There is a large pothole on the road"}
)
assert response.status_code == 200

result = response.json()
assert "predicted_category" in result
assert result["predicted_category"] in [
    "road", "safety", "waste", "water"
]
assert "confidence" in result
assert 0 <= result["confidence"] <= 1


# Test 4: Empty request text
response = requests.post(
    f"{BASE_URL}/predict",
    json={"request_text": ""}
)
assert response.status_code == 400


# Test 5: Missing request text
response = requests.post(
    f"{BASE_URL}/predict",
    json={}
)
assert response.status_code == 400


# Test 6: Non-string request text
response = requests.post(
    f"{BASE_URL}/predict",
    json={"request_text": 123}
)
assert response.status_code == 400


print("All API tests passed successfully!")
