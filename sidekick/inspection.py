import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CORR_FN = typing.Callable[[np.ndarray, np.ndarray], float]


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

    corr = table.apply(lambda x: pd.factorize(x)[0]).corr(
        method="pearson", min_periods=1
    )
    f, ax = plt.subplots(figsize=figsize)
    return sns.heatmap(corr, vmax=1, square=True)


def plot_pairs(table: pd.DataFrame, columns: typing.List[str], height=2.5):
    # scatterplot
    sns.set()
    df = table.apply(lambda x: pd.factorize(x)[0])
    return sns.pairplot(df[columns], height=height)
    # plt.show()


def plot_histogram(table: pd.DataFrame, columns):
    # TODO
    pass


def plot_scatter(table: pd.DataFrame, x, y):
    # TODO
    pass
