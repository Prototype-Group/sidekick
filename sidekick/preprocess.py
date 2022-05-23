import typing
from functools import partial

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.model_selection import train_test_split

VALID_IMPUTATIONS = {"auto", "mean", "mode", "drop", "replace", "interpolate"}
NOT_VALID_VALUES = {None, "None", np.NaN, "", "N/A", "n/a", float("nan")}


def _to_list(lst: typing.Union[str, typing.List[str]]) -> typing.List[str]:
    """Convert value to list if not already a list"""
    if isinstance(lst, (int, str, float)):
        lst = [lst]
    elif isinstance(lst, type(None)):
        lst = []
    return lst


def convert_to_categorical(table: pd.DataFrame):
    """Convert DataFrame to categorical values (int)"""
    table = table.apply(lambda x: pd.Categorical(x).codes if is_object_dtype(x) else x)
    return table.apply(lambda x: x.astype("category") if is_object_dtype(x) else x)


def drop_duplicates(table: pd.DataFrame) -> pd.DataFrame:
    return table[~table.duplicated()]


def drop_missing_values(
    table: pd.DataFrame,
    invalid_values: typing.Optional[typing.List[typing.Any]] = None,
):
    """Remove all rows with missing values

    Args:
        table: DataFrame
        invalid_values, optional: Values to remove and are counted as missing. Defaults to None.

    Returns:
        DataFrame without missing values
    """
    if invalid_values is None:
        invalid_values = NOT_VALID_VALUES

    for column in table.columns:
        table = table[~table[column].isin(invalid_values)]
    table.reset_index(drop=True, inplace=True)
    return table


# TODO: Defaults to impute floats to mean. Not good for features, s.a. years.
#       Should have a method for explicit casting
def _impute_values(
    table: pd.DataFrame,
    column: str,
    method: str = "auto",
    invalid_values: typing.Optional[typing.List[typing.Any]] = None,
    value: typing.Optional[int] = None,
):
    """Impute missing values in a DataFrame's column

    Helper function for `impute`.

    Args:
        table: DataFrame
        column: Columns to impute
        method: Type of imputation. Defaults to 'auto'.
        invalid_values: Values that count as missing values
        value: Value to impute. Only valid if `method='replace'`. Defaults to None.

    Returns:
        DataFrame with imputed values for the column
    """

    def _impute(value: typing.Union[typing.Any, typing.Callable]):
        table.loc[table[column].isin(invalid_values), column] = value

    col = table[column]

    if method == "auto":
        _impute(value=(col.mode()[0] if is_object_dtype(col) else col.mean()))
    elif method == "mean":
        _impute(value=col.mean())
    elif method == "mode":
        _impute(value=col.mode()[0])
    elif method == "replace":
        if value is not None:
            _impute(value=value)
        else:
            raise ValueError(
                f"""
                Imputation method 'value' needs to be given a default value (text or number).
                Found: {value}
                """
            )
    elif method == "drop":
        table = table[~col.isin(invalid_values)]
    else:
        raise ValueError(
            f"""
            The selected imputation method is not valid.
            Must be one of '{[str(imp)+", " for imp in VALID_IMPUTATIONS]}'
            """
        )
    return table


def impute_values(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    method: str = "auto",
    invalid_values: typing.Optional[typing.List[typing.Any]] = None,
    target: typing.Optional[str] = None,
    value: typing.Optional[int] = None,
):
    """Impute missing values in a DataFrame

    The following methods for imputation is valid:
        - 'auto': Replace missing values with mean or mode depending on the column type
        - 'mean': Replace missing values with column mean
        - 'mode': Replace missing values with column mean
        - 'drop': Drop rows where values are missing in column
        - 'replace': Replace missing values in column with a value

    Args:
        table: DataFrame
        column: Columns to impute
        method: {'auto', 'mean', 'mode', 'drop', 'replace', 'interpolate'}
                Type of imputation. Defaults to 'auto'.
        invalid_values: Values that count as missing values
        value: Value to impute. Only valid if `method='replace'`. Defaults to None.

    Returns:
        DataFrame with imputed values for the column
    """

    if invalid_values is None:
        invalid_values = NOT_VALID_VALUES

    columns = _to_list(columns)
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
        table = _impute_values(
            table=table,
            column=c,
            method=method,
            invalid_values=invalid_values,
            value=value,
        )
    table.reset_index(drop=True, inplace=True)
    return table


def select_or_filter(
    table: pd.DataFrame,
    column: str,
    values: typing.Optional[typing.Union[typing.Any, typing.List[typing.Any]]] = None,
    conditions: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    invert_selection: bool = False,
):
    """Helper function for selecting exact values or based on condition from a DataFrame

    Match either exact values (values) or a logical condition (conditions).
    The found matches/selections are inverted based on if `inverted=True`.

    Example:
        >>> df = pd.DataFrame(data={'col1': [3, 14, -2, 28],
                                    'col2': [8, 21, 1, 34]})
        >>> sidekick.select_or_filter(df, column='col1', values=[3], invert=True)
          col1  col2
        0    3	   8
        >>> sidekick.select_or_filter(df, column='col1', values=[3], invert=False)
          col1  col2
        0   14	  21
        1   -2     1
        2   28    34
        >>> sidekick.select_or_filter(df, column=df.columns[0], values=[3], conditions=["<2"], invert=True)
           col1  col2
        0   14	  21
        1   28    34

    Args:
        table: DataFrame
        column: Column to find values or match condition for
        values: Values to find exact values for. Defaults to None.
        conditions: Logical conditions to match for. Defaults to None.
        invert: If selected values should be inverted (true for filtering, false for selecting). Defaults to False.

    Returns:
        DataFrame which fulfill the selection or filtering criteria
    """

    def _selected(
        table: pd.DataFrame, mask: "pd.BoolMask", invert: bool
    ) -> pd.DataFrame:
        return table[~mask if invert else mask]

    # Select/filter based on "exact" matched values
    values = _to_list(values)
    mask = table[column].isin(values)
    results = _selected(table, mask, invert_selection)

    # Select/filter based on condition
    conditions = _to_list(conditions)
    for cond in conditions:
        try:
            mask = eval(f"table[column] {cond}")
            results = pd.merge(
                results,
                _selected(table, mask, invert_selection),
                how=("inner" if invert_selection else "outer"),
            )
        except Exception as e:
            print(f"Could not evaluate condition: {cond}... Got exception: {e}")

    results.reset_index(drop=True, inplace=True)
    return results


filter_values = partial(select_or_filter, invert_selection=True)
select_values = partial(select_or_filter, invert_selection=False)


def split(
    table: pd.DataFrame,
    test_size: float = 0.2,
    shuffle: bool = False,
    random_state: int = 42,
    stratify: typing.Optional[pd.DataFrame] = None,
    **kwargs,
):
    """Create train_test_split

    Can be called on twice to first create the train and test split from the full data
    then called on the test set with even fraction to create the test and validation set.

    Example:
        >>> df = pd.read_csv(
            "https://storage.googleapis.com/bucket-8732/datalibrary/forecast_sales.csv"
        )
        >>> train, test = split(table, test_size=0.2, shuffle=True, stratify=table["Day"])
        >>> test, valid = split(test, test_size=0.5)

    Args:
        table: DataFrame
        test_size: Fraction of dataset dedicated to the test set.
        shuffle: If data should be shuffled or follow current order in dataset
        random_state: Pseudo random number for reproducible state
        stratify: If splits should respect specific distributions. Only valid if `shuffle=True`

    Returns:
        Two splits of the DataFrame (train, test)
    """

    assert test_size <= 1 and test_size >= 0
    train, test = train_test_split(
        table,
        train_size=1 - test_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
        **kwargs,
    )
    return train, test


def add_split(table: pd.DataFrame, dataset: pd.DataFrame, split: str):
    """Append which "Split" DataFrame values belongs to after split

    Expects splits to be generated example functions:
        - `train_test_split`
        - `pd.DataFrame.sample`
        - DataFrame indexing

    Example:
        >>> df = pd.read_csv(
            "https://storage.googleapis.com/bucket-8732/datalibrary/forecast_sales.csv"
        )
        >>> train, test = split(table, test_size=0.2, shuffle=True, stratify=table["Day"])
        >>> test, valid = split(test, test_size=0.5)

        >>> sidekick.add_split(table, dataset=train, split="Train")
        >>> sidekick.add_split(table, dataset=test,  split="Test")
        >>> sidekick.add_split(table, dataset=valid, split="Valid")
        >>> table.head()

    Args:
        table: DataFrame
        dataset: Dataset obtained from a dataset split
        split: Name of dataset split

    Returns:
        Split appended to the table under the column name "Split"
    """

    table.loc[dataset.index, "Split"] = split
    return table


if __name__ == "__main__":
    df = pd.read_csv(
        "https://storage.googleapis.com/bucket-8732/datalibrary/forecast_sales.csv"
    )

    # Corrupt the dataset
    df.loc[[0, 5, 100], "Year"] = np.NaN
    df.loc[[0, 1, 2], "Day"] = np.NaN

    # Clean data
    table = drop_duplicates(df)
    table = impute_values(table, columns="Year", method="auto")
    table = drop_missing_values(table)
    table = filter_values(table, column="Revenue", conditions="<5")
