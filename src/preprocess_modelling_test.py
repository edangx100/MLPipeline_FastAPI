import pandas as pd
import numpy as np
import pytest
import data_preprocess
import modeling
from pickle import load


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/cleaned/census.csv")
    return df


def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """
    encoder_filename = "data/model/encoder.sav"
    labelbinarizer_filename = "data/model/labelbinarizer.sav"

    encoder = load(open(encoder_filename, 'rb'))
    lb = load(open(labelbinarizer_filename, 'rb'))

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, _, _ = data_preprocess.process_data(
        data,
        categorical_features=cat_features,
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder_labelbinarizer(data):
    """
    Check split have same number of rows for X and y
    """
    encoder_filename = "data/model/encoder.sav"
    labelbinarizer_filename = "data/model/labelbinarizer.sav"

    encoder_test = load(open(encoder_filename, 'rb'))
    lb_test = load(open(labelbinarizer_filename, 'rb'))

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    _, _, encoder, lb = data_preprocess.process_data(
        data,
        categorical_features=cat_features,
        label="salary", training=True)

    _, _, _, _ = data_preprocess.process_data(
        data,
        categorical_features=cat_features,
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference():
    """
    Check inference type
    """
    model_filename = "data/model/trained_model.sav"
    encoder_filename = "data/model/encoder.sav"
    labelbinarizer_filename = "data/model/labelbinarizer.sav"

    model = load(open(model_filename, 'rb'))
    encoder = load(open(encoder_filename, 'rb'))
    lb = load(open(labelbinarizer_filename, 'rb'))

    array = np.array([[
                     39,
                     "State-gov",
                     "Bachelors",
                     "Never-married",
                     "Adm-clerical",
                     "Not-in-family",
                     "White",
                     "Male",
                     40,
                     "United-States"
                     ]])
    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = data_preprocess.process_data(
        df_temp,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False)

    pred = modeling.inference(model, X)
    y = lb.inverse_transform(pred)[0]

    assert y == ">=50K" or y == "<=50K"