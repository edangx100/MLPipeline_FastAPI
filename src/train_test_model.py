import pandas as pd
from sklearn.model_selection import train_test_split
from pickle import dump
import src.data_preprocess as data_preprocess
import src.modeling as modeling
import logging


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("data/cleaned/census.csv")
    train, test = train_test_split(df, test_size=0.20)
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

    # Preparation for model training
    X_train, y_train, encoder, lb = data_preprocess.process_data(
        train, categorical_features=cat_features,
        label="salary", training=True
    )
    # Model training
    trained_model = modeling.train_model(X_train, y_train)

    # Preparation for model testing
    X_test, y_test, _, _ = data_preprocess.process_data(
        test,
        categorical_features=cat_features,
        label="salary", encoder=encoder, lb=lb, training=False)
    y_preds = trained_model.predict(X_test)

    # Model testing
    precision, recall, fbeta = modeling.compute_model_metrics(y_test, y_preds)
    line = f'\n Model Classification Metrics over test data:\n Precision: {precision} Recall: {recall} FBeta: {fbeta}'
    logging.info(line)

    # Preperation to save model and encoders
    model_filename = "data/model/trained_model.sav"
    encoder_filename = "data/model/encoder.sav"
    labelbinarizer_filename = "data/model/labelbinarizer.sav"

    # Save model and encoders
    dump(trained_model, open(model_filename, 'wb'))
    dump(encoder, open(encoder_filename, 'wb'))
    dump(lb, open(labelbinarizer_filename, 'wb'))

    # Save test data, to be used for slice scoring
    df.to_csv("data/cleaned/test.csv", index=False)