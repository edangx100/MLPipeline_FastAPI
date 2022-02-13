import pandas as pd
import pytest
import src.data_cleaning


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = src.data_cleaning.__clean(df)
    return df


def test_data_shape(data):
    """
    Data is assumed to have no null values
    """
    assert data.shape == data.dropna().shape


def test_columns(data):
    """
    Check that columns not required are removed
    """
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns