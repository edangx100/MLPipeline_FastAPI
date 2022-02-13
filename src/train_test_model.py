import pandas as pd
from sklearn.model_selection import train_test_split
from pickle import dump
import src.data_preprocess as data_preprocess
import src.modeling as modeling


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("data/cleaned/census.csv")
    train, _ = train_test_split(df, test_size=0.20)
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

    X_train, y_train, encoder, lb = data_preprocess.process_data(
        train, categorical_features=cat_features,
        label="salary", training=True
    )
    trained_model = modeling.train_model(X_train, y_train)

    model_filename = "data/model/trained_model.sav"
    encoder_filename = "data/model/encoder.sav"
    labelbinarizer_filename = "data/model/labelbinarizer.sav"

    dump(trained_model, open(model_filename, 'wb'))
    dump(encoder, open(encoder_filename, 'wb'))
    dump(lb, open(labelbinarizer_filename, 'wb'))