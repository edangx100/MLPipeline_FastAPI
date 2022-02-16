from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings"}


def test_get_malformed():
    r = client.get("/items")
    assert r.status_code != 200


def test_post_more():
    r = client.post("/", json={
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
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_less():
    r = client.post("/", json={
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "maritalStatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 20,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}