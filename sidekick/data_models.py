from typing import Tuple


class FeatureSpec:
    def __init__(
        self,
        name: str,
        dtype: str,
        shape: Tuple[int, ...],
        categories: Tuple[str, ...] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.categories = categories

    def __repr__(self):
        return "FeatureSpec(name={0}, dtype={1}, shape={2}".format(
            self.name, self.dtype, self.shape
        ) + (
            ", categories={0})".format(self.categories)
            if self.categories
            else ")"
        )
