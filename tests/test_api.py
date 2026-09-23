import requests


BASE_URL = "http://127.0.0.1:5000"


# Test 1: Health endpoint
response = requests.get(f"{BASE_URL}/health")

assert response.status_code == 200
assert response.json()["status"] == "ok"


# Test 2: Categories endpoint
response = requests.get(f"{BASE_URL}/categories")

assert response.status_code == 200
assert response.json() == ["road", "safety", "waste", "water"]


# Test 3: Valid prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"request_text": "There is a large pothole on the road"}
)

assert response.status_code == 200

result = response.json()

assert "category" in result
assert "confidence" in result


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


print("All API tests passed successfully!")