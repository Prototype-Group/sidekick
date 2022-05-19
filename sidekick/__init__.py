import pkg_resources

from . import deployment, encode
from .dataset import create_dataset, process_image
from .dataset_client import DatasetClient
from .deployment import Deployment
from .inspection import (
    plot,
    plot_correlation,
    plot_count,
    plot_histogram,
    plot_outliers,
    plot_pairs,
    plot_scatter,
    show_missing,
)
from .preprocess import (
    convert_to_categorical,
    drop_missing_values,
    impute_table_columns,
    remove_duplicate_table_rows,
    split,
)

__all__ = [
    "Deployment",
    "DatasetClient",
    "create_dataset",
    "deployment",
    "encode",
    "process_image",
    "convert_to_categorical",
    "drop_missing_values",
    "impute_table_columns",
    "remove_duplicate_table_rows",
    "split",
    "plot_correlation",
    "plot_pairs",
    "plot_histogram",
    "plot",
    "plot_count",
    "plot_scatter",
    "plot_outliers",
    "show_missing",
]

try:
    __version__ = pkg_resources.get_distribution("sidekick").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.0.0-local"
