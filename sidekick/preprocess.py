import typing
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from PIL import Image

VALID_IMPUTATIONS = ["mean", "mode", "drop", "replace", "interpolate"]

"""
Knowledge center instructions:
- https://peltarion.com/knowledge-center/documentation/datasets-view/data-preprocessing

Dataset:
- https://storage.googleapis.com/bucket-8732/datalibrary/forecast_sales.csv
"""


def _to_list(lst: typing.Union[str, typing.List[str]]) -> typing.List[str]:
    return [lst] if isinstance(lst, (int, str, float)) else lst


def convert_to_categorical(table: pd.DataFrame):
    """Convert pandas DataFrame string/objects to categorical."""
    table = table.apply(lambda x: pd.Categorical(x).codes if is_object_dtype(x) else x)
    return table.apply(lambda x: x.astype("category") if is_object_dtype(x) else x)


def remove_duplicate_table_rows(table: pd.DataFrame) -> pd.DataFrame:
    """_summary_def process_image(image: Image.Image,
    mode: str = 'center_crop_or_pad',
    size: Tuple[int, int] = None,
    file_format: str = None) -> Image.Image:
    """
    return table[~table.duplicated()]


# TODO
# https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python?scriptVersionId=94433095&cellId=43
def drop_missing_values(table: pd.DataFrame):
    table = table.dropna()
    table.reset_index(inplace=True)
    return table


# TODO
def _impute(
    table: pd.DataFrame,
    column: str,
    method: str,
    invalid_values: typing.List[typing.Any],
    value: typing.Optional[int] = None,
):

    if method == "mean":
        table.loc[table[column].isin(invalid_values), column] = table[column].mean()
    elif method == "mode":
        table.loc[table[column].isin(invalid_values), column] = table[column].mode()[0]
    elif method == "replace":
        if value is not None:
            table.loc[table[column].isin(invalid_values), column] = value
        else:
            raise ValueError(
                f"Imputation method 'value' needs to be given a default value (text or number) and not: {value}"
            )
    elif method == "drop":
        table = table[~table[column].isin(invalid_values)]
    else:
        raise ValueError(
            """
            The selected imputation method is not valid.
            Must be one of '{[str(imp)+", " for imp in VALID_IMPUTATIONS]}'
            """
        )
    return table


def impute_table_columns(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    method="mean",
    invalid_values: typing.Optional[typing.List[typing.Any]] = None,
    target: typing.Optional[str] = None,
    value: typing.Optional[int] = None,
):
    if invalid_values is None:
        invalid_values = [None, "None", np.NaN]

    columns = [columns] if isinstance(columns, str) else columns
    if target in columns:
        columns = [c for c in columns if c != target]

    assert len(columns) != len(
        table.columns
    ), f"Can not pass all columns for imputation without selecting a target feature!"
    assert (
        len(columns) > 0
    ), f"Imputation should not be done on the target feature: {target}"
    assert (
        method in VALID_IMPUTATIONS
    ), f"The selected imputation method must be one of: {[str(imp)+', ' for imp in VALID_IMPUTATIONS]}'"

    for c in columns:
        table = _impute(
            table=table,
            column=c,
            method=method,
            invalid_values=invalid_values,
            value=value,
        )
    table.reset_index(drop=True)
    return table


# TODO:
def split():
    pass


def min_max_scaler():
    # from sklearn.preprocessing import StandardScaler
    pass


if __name__ == "__main__":
    df = pd.read_csv(
        "https://storage.googleapis.com/bucket-8732/datalibrary/forecast_sales.csv"
    )

    # Corrupt the dataset
    df.loc[[0, 5, 100], "Year"] = np.NaN
    df.loc[[0, 1, 2], "Day"] = np.NaN

    method = "drop"
    assert df.isnull().values.any()
    table = impute_table_columns(df.copy(), columns="Year", method=method)
    assert table.isnull().values.any() == False
    print(table.head())
    print(table.describe())
    print(table.info())
    # table = impute_table_columns(df.copy(), columns=["Year"], method=method)
    # assert table.isnull().values.any() == False
