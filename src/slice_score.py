import pandas as pd
from pickle import load
import src.data_preprocess as data_preprocess
import src.modeling as modeling
import logging


def get_scores():
    """
    Execute score checking
    """
    # Load test data
    test = pd.read_csv("data/cleaned/test.csv")

    model_filename = "data/model/trained_model.sav"
    encoder_filename = "data/model/encoder.sav"
    labelbinarizer_filename = "data/model/labelbinarizer.sav"

    # Load trained model and encoders
    model = load(open(model_filename, 'rb'))
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

    # compute slice metrics
    slice_values = []
    for feature in cat_features:
        for cls in test[feature].unique():
            df_temp = test[test[feature] == cls]

            X_test, y_test, _, _ = data_preprocess.process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = model.predict(X_test)

            precision, recall, fbeta = modeling.compute_model_metrics(y_test, y_preds)

            line = f'[{feature}->{cls}] Precision: {precision} Recall: {recall} FBeta: {fbeta}'
            logging.info(line)
            slice_values.append(line)

    with open('data/model/slice_output.txt', 'w') as output:
        for slice_value in slice_values:
            output.write(slice_value + '\n')