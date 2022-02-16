# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Ed created the model. Logistic Regression from scikit-learn was used.

## Intended Use

This model performs the task of predicting the salary of a person using features of each person.

## Training Data

The data was obtained from https://archive.ics.uci.edu/ml/datasets/census+income.
The original data has 48,842 rows, and 80% of it was used for training.
To use the data for training a One-Hot Encoder aws used on the features and a label binarizer was used on the labels.

## Evaluation Data

The data was obtained from  https://archive.ics.uci.edu/ml/datasets/census+income.
The original data has 48,842 rows, and 20% of it was used for testing.
To use the data for testing a One-Hot Encoder aws used on the features and a label binarizer was used on the labels.

## Metrics

The model was evaluated using Accuracy score. The value is around 0.832.

## Ethical Considerations
Dataset contains features of race and gender. 
The model may be skewed toward a particular group of people (ie. race, gender). 

## Caveats and Recommendations

Further investigation can be done address possible race and gender bias, such as not using race and gender features during model training.