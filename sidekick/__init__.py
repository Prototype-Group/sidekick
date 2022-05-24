import pkg_resources

from . import deployment, encode
from .dataset import create_dataset, process_image
from .dataset_client import DatasetClient
from .deployment import Deployment
from .inspection import (
    plot,
    plot_correlation,
    plot_histogram,
    plot_outliers,
    plot_pairs,
    plot_scatter,
    show_summary,
)
from .preprocess import (
    add_split,
    convert_to_categorical,
    drop_duplicates,
    drop_missing_values,
    filter_values,
    impute_values,
    select_values,
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
    "impute_values",
    "drop_duplicates",
    "filter_values",
    "select_values",
    "split",
    "add_split",
    "plot",
    "plot_correlation",
    "plot_histogram",
    "plot_outliers",
    "plot_pairs",
    "plot_scatter",
    "show_summary",
]

try:
    __version__ = pkg_resources.get_distribution("sidekick").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.0.0-local"
