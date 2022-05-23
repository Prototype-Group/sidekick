import functools
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import sidekick


@pytest.fixture
def data() -> pd.DataFrame:
    columns = [
        "Date",
        "Year",
        "Week",
        "Day",
        "Store Name",
        "Location",
        "Store Type",
        "Advertising Level",
        "Special Promotion",
        "Holiday",
        "Open",
        "Revenue",
    ]
    df = pd.read_csv(
        "https://storage.googleapis.com/bucket-8732/datalibrary/forecast_sales.csv"
    )

    # Corrupt the dataset
    df.loc[[0, 5, 100], "Year"] = np.NaN
    df.loc[[0, 1, 3], "Week"] = np.NaN
    # TODO: need to handle str and enums for imputations
    # df.loc[[0, 1, 2], "Day"] = np.NaN
    return df


def test_dataset_is_corrupted(data: pd.DataFrame):
    assert data.isnull().values.any()


def test_drop_missing(data: pd.DataFrame):
    table = sidekick.drop_missing_values(data)
    assert table.isnull().values.any() == False


def test_impute_mean(data: pd.DataFrame):
    method = "mean"
    assert data.isnull().values.any()
    table = sidekick.impute_values(data.copy(), columns="Year", method=method)
    assert table.isnull().values.any() == True
    assert table["Year"].isnull().values.any() == False

    table = sidekick.impute_values(data.copy(), columns=["Year"], method=method)
    assert table.isnull().values.any() == True
    assert table["Year"].isnull().values.any() == False

    table = sidekick.impute_values(data.copy(), columns=["Year", "Week"], method=method)
    assert table.isnull().values.any() == False


def test_impute_mode(data: pd.DataFrame):
    method = "mode"
    assert data.isnull().values.any()
    table = sidekick.impute_values(data.copy(), columns="Year", method=method)
    assert table.isnull().values.any() == True
    assert table["Year"].isnull().values.any() == False

    table = sidekick.impute_values(data.copy(), columns=["Year"], method=method)
    assert table.isnull().values.any() == True
    assert table["Year"].isnull().values.any() == False

    table = sidekick.impute_values(data.copy(), columns=["Year", "Week"], method=method)
    assert table.isnull().values.any() == False


def test_impute_value(data: pd.DataFrame):
    method = "replace"
    value = -1
    assert data.isnull().values.any()
    table = sidekick.impute_values(
        data.copy(), columns="Year", method=method, value=value
    )
    assert table.isnull().values.any() == True
    assert table["Year"].isnull().values.any() == False
    assert table["Year"].value_counts()[value] == 3

    table = sidekick.impute_values(
        data.copy(), columns=["Year"], method=method, value=value
    )
    assert table.isnull().values.any() == True
    assert table["Year"].isnull().values.any() == False
    assert table["Year"].value_counts()[value] == 3

    table = sidekick.impute_values(
        data.copy(), columns=["Year", "Week"], method=method, value=value
    )
    assert table.isnull().values.any() == False
    assert table["Year"].value_counts()[value] == 3
    assert table["Week"].value_counts()[value] == 3


def test_impute_drop(data: pd.DataFrame):
    method = "drop"
    assert data.isnull().values.any()
    table = sidekick.impute_values(data, columns="Year", method=method)
    assert table["Year"].isnull().values.any() == False

    table = sidekick.impute_values(data, columns=["Year"], method=method)
    assert table["Year"].isnull().values.any() == False

    table = sidekick.impute_values(
        data, columns=data.columns, target="Revenue", method=method
    )
    assert table.isnull().values.any() == False
