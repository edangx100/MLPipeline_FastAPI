import requests


data = {
    "age": 40,
    "workclass": "Private",
    "education": "Masters",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hoursPerWeek": 60,
    "nativeCountry": "United-States"
}
r = requests.post('https://proj3-mlops.herokuapp.com/', json=data)

assert r.status_code == 200

print(f"Response status code: {r.status_code}")
print(f"Model inference result: {r.json()}")