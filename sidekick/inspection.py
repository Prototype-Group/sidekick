import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .preprocess import _to_list, convert_to_categorical

CORR_FN = typing.Callable[[np.ndarray, np.ndarray], float]
VALID_PLOT_KINDS = [
    "strip",
    "swarm",
    "box",
    "violin",
    "boxen",
    "point",
    "bar",
    "count",
]

## Show
#########################################
def show_missing(table: pd.DataFrame):
    # Display how many missing features and percentages
    # TODO: May not be correct, should match "None", np.NaN and isnull
    total = table.isnull().sum().sort_values(ascending=False)
    percent = (table.isnull().sum() / table.isnull().count()).sort_values(
        ascending=False
    )
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    return missing_data


## Plot
#########################################


def _set_figsize(figsize: typing.Optional[typing.Tuple[int, int]] = None):
    if figsize is None:
        figsize = (12, 9)
    return plt.subplots(figsize=figsize)


def plot_correlation(
    table: pd.DataFrame,
    method: typing.Union[str, CORR_FN] = "pearson",
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
):
    valid_methods = ["pearson", "kendall", "spearman"]
    assert method in valid_methods or isinstance(
        method, callable
    ), f"Method: {method} must be one of: {[str(m)+', ' for m in valid_methods]} or a correlation function"

    if figsize is None:
        figsize = (12, 9)
    f, ax = plt.subplots(figsize=figsize)

    corr = convert_to_categorical(table).corr(method=method, min_periods=1)
    return sns.heatmap(corr, vmax=1, square=True)


def plot_pairs(
    table: pd.DataFrame,
    columns: typing.List[str],
    height: float = 2.5,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
):
    columns = _to_list(columns)
    df = table[columns]
    df = convert_to_categorical(df)
    return sns.pairplot(df, height=height)


def plot(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    target: typing.Optional[str] = None,
    kind: typing.Optional[str] = None,
    hue: typing.Optional[str] = None,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    **kwargs,
):
    assert kind in VALID_PLOT_KINDS, f"Not a valid seaborn plot kind: {kind}"
    columns = _to_list(columns)

    # TODO: Changing this does not create new fig size!!
    fig, ax = _set_figsize(figsize)

    for c in columns:
        sns.catplot(x=c, y=target, hue=hue, kind=kind, data=table, **kwargs)


def plot_histogram(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    target: typing.Optional[str] = None,
    kind: str = "bar",
    hue: typing.Optional[str] = None,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    **kwargs,
):
    assert kind in ["bar", "box"]
    plot(
        table=table,
        columns=columns,
        target=target,
        kind=kind,
        hue=hue,
        figsize=figsize,
        **kwargs,
    )


def plot_count(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    kind: str = "count",
    hue: typing.Optional[str] = None,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    **kwargs,
):
    assert kind in ["count"]
    plot(table=table, columns=columns, kind=kind, hue=hue, figsize=figsize, **kwargs)


def plot_scatter(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    target: str,
    kind: str = "strip",
    hue: typing.Optional[str] = None,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    **kwargs,
):
    assert kind in ["strip", "swarm"]
    plot(
        table=table,
        columns=columns,
        target=target,
        kind="bar",
        hue=hue,
        figsize=figsize,
        **kwargs,
    )


# TODO
def plot_outliers(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
):
    columns = _to_list(columns)
    df = convert_to_categorical(table[columns])
    _set_figsize(figsize)
    return sns.boxplot(data=df, palette="Set2")
