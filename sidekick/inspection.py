import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .preprocess import NOT_VALID_VALUES, _to_list, convert_to_categorical

CORR_FN = typing.Callable[[np.ndarray, np.ndarray], float]
VALID_PLOT_KINDS = {
    "strip",
    "swarm",
    "box",
    "violin",
    "boxen",
    "point",
    "bar",
    "count",
    "hist",
    "histogram",
}


def show_missing(
    table: pd.DataFrame,
    invalid_values: typing.Optional[
        typing.Union[typing.Any, typing.List[typing.Any]]
    ] = None,
) -> pd.DataFrame:
    """Summarize fraction of missing values in a pandas DataFrame.

    Args:
        table: DataFrame
        invalid_values: Values to be removed. Will default to multiple different variants to say 'None'.

    Returns:
       Summary of missing values in the table
    """
    if invalid_values is None:
        invalid_values = NOT_VALID_VALUES

    total = table.count()
    missing_total = table.isin(invalid_values).sum()
    not_missing = total - missing_total
    percent = (missing_total / total) * 100

    missing_data = pd.concat(
        [not_missing, missing_total, percent],
        axis=1,
        keys=["Not Missing", "Total Missing", "% Missing"],
    )
    missing_data.sort_values(by=["Total Missing"], inplace=True, ascending=False)
    return missing_data


def _set_figsize(
    figsize: typing.Optional[typing.Tuple[int, int]] = None
) -> typing.Tuple[int, int]:
    """Set graph figures height and width.

    Args:
        figsize: Initial figure size (height, width). Defaults to None

    Returns:
        figure size (height, width)
    """
    if figsize is None:
        figsize = (4, 5)
    return figsize


def plot_correlation(
    table: pd.DataFrame,
    method: typing.Union[str, CORR_FN] = "pearson",
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
):
    """Correlation plot.

    Visualize a seaborn correlation plot for the features in a pandas DataFrame.
    Extends: https://seaborn.pydata.org/generated/seaborn.heatmap.html

    Categorical features are converted to numbers, since required for the correlation plot.
    The same correlation methods are provided as in seaborn:

        - 'pearson'
        - 'kendall'
        - 'spearman'

    Args:
        table: DataFrame
        method: {'pearson', 'kendall', 'spearman'}
                Correlation methods. Defaults to "pearson",
        figsize: Initial figure size (height, width). Defaults to None

    Returns:
        Correlation plot between all features in the table
    """

    valid_methods = {"pearson", "kendall", "spearman"}
    assert method in valid_methods or isinstance(
        method, callable
    ), f"Method: {method} must be one of: {[str(m)+', ' for m in valid_methods]} or a correlation function"

    corr = convert_to_categorical(table).corr(method=method, min_periods=1)
    return sns.heatmap(corr, vmax=1, square=True)


def plot_pairs(
    table: pd.DataFrame,
    columns: typing.List[str],
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
):
    """Plot multiple features together in a scatter plot.

    Args:
        table: DataFrame
        columns: Name of the columns/features to plot
        figsize: Initial figure size (height, width) of the combined plots. Defaults to None.

    Returns:
        A pair plot between all the columns in the table
    """
    columns = _to_list(columns)
    df = table[columns]
    height = _set_figsize(figsize=figsize)[0] / len(columns)
    df = convert_to_categorical(df)
    return sns.pairplot(df, height=height)


def plot(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    target: typing.Optional[str] = None,
    kind: typing.Optional[str] = None,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    **kwargs,
):
    """Plotting function.

    Extends seaborn's plotting capabilities, namely

        - seaborn.histplot
        - seaborn.catplot

    Args:
        table: DataFrame
        columns: Features to plot on the x-axis
        target: Target feature to plot on the y-axis.
                Optional, depending on the plot type. Defaults to None.
        kind: {"strip", "swarm", "box", "violin", "boxen",
                "point", "bar", "count", "hist", "histogram"}
                The type of plot to use. Defaults to None.
        figsize: Initial figure size (height, width). Defaults to None.
    """
    assert kind in VALID_PLOT_KINDS, f"Not a valid seaborn plot kind: {kind}"
    columns = _to_list(columns)

    for c in columns:
        plt.figure()
        if kind in ("hist", "histogram"):
            sns.histplot(x=c, y=target, data=table, **kwargs)
        else:
            if ("height" or "aspect") not in kwargs:
                (height, width) = _set_figsize(figsize)
                kwargs["height"] = height
                kwargs["aspect"] = width / height
            sns.catplot(x=c, y=target, kind=kind, data=table, **kwargs)


def plot_histogram(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    target: typing.Optional[str] = None,
    kind: str = "hist",
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    **kwargs,
):
    """Plot histogram.

    Args:
        table: DataFrame
        columns: Features to plot on the x-axis
        target: Target feature to plot on the y-axis.
                Optional, depending on the plot type. Defaults to None.
        kind: {"bar", "box", "hist", "histogram"}
                The type of plot to use. Defaults to None.
        figsize: Initial figure size (height, width). Defaults to None.
    """
    assert kind in {"bar", "box", "hist", "histogram"}
    plot(
        table=table,
        columns=columns,
        target=target,
        kind=kind,
        figsize=figsize,
        **kwargs,
    )


def plot_scatter(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    target: str,
    kind: str = "strip",
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    **kwargs,
):
    """Plot scatter plot.

    Args:
        table: DataFrame
        columns: Features to plot on the x-axis
        target: Target feature to plot on the y-axis.
                Optional, depending on the plot type. Defaults to None.
        kind: {"strip", "swarm"}
                The type of plot to use. Defaults to None.
        figsize: Initial figure size (height, width). Defaults to None.
    """
    assert kind in {"strip", "swarm"}
    plot(
        table=table,
        columns=columns,
        target=target,
        kind="bar",
        figsize=figsize,
        **kwargs,
    )


def plot_outliers(
    table: pd.DataFrame,
    columns: typing.Union[str, typing.List[str]],
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
):
    """Visualize outliers per column with box plots.

    Args:
        table: DataFrame
        columns: Features to plot on the x-axis
        figsize: Initial figure size (height, width). Defaults to None.
    """
    columns = _to_list(columns)
    df = table[columns]
    df = convert_to_categorical(df)

    plt.figure()
    (height, width) = _set_figsize(figsize)
    _, ax = plt.subplots(figsize=(height, width))
    return sns.boxplot(data=df, palette="Set2", ax=ax)
