import pkg_resources

from . import deployment, encode
from .dataset import create_dataset, process_image
from .dataset_client import DatasetClient
from .deployment import Deployment
from .inspection import plot_correlation, plot_pairs
from .preprocess import (
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
    "impute_table_columns",
    "remove_duplicate_table_rows",
    "drop_missing_values",
    "split",
    "plot_correlation",
    "plot_pairs",
]

try:
    __version__ = pkg_resources.get_distribution("sidekick").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.0.0-local"
