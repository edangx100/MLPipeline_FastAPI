from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from pandas.core.frame import DataFrame
import numpy as np
from pickle import load
import src.data_preprocess as data_preprocess
import src.modeling as modeling


class inputData(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    maritalStatus: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: str
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    hoursPerWeek: int
    nativeCountry: str


app = FastAPI()


@app.get("/")
async def say_greetings():
    return {"message": "Greetings"}


@app.post("/")
async def inference(input_data: inputData):
    model_filename = "data/model/trained_model.sav"
    encoder_filename = "data/model/encoder.sav"
    labelbinarizer_filename = "data/model/labelbinarizer.sav"

    model = load(open(model_filename, 'rb'))
    encoder = load(open(encoder_filename, 'rb'))
    lb = load(open(labelbinarizer_filename, 'rb'))

    print(encoder)

    array = np.array([[
                     input_data.age,
                     input_data.workclass,
                     input_data.education,
                     input_data.maritalStatus,
                     input_data.occupation,
                     input_data.relationship,
                     input_data.race,
                     input_data.sex,
                     input_data.hoursPerWeek,
                     input_data.nativeCountry
                     ]])
    df_temp = DataFrame(data=array, columns=[
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
    return {"prediction": y}