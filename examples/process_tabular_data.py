# %%
import pathlib

import numpy as np
import pandas as pd
from IPython.display import display

import sidekick

# %%
"""
TODO: Introduce the problem and describe what to do

- Describe the structure
- For each section
    - Describe problem and how to check for it
    - Then how to solve it

"""

"""
##  Load the data
Provided synthetic data to show the steps often required to prepare a tabular dataset
"""
dataset_path = pathlib.Path("./synthetic_insurance_ruined.csv")
df = pd.read_csv(dataset_path, index_col=0)
display(df.head())

# %%
# Let's get some information about what our dataset looks like:
display(df.info())
display(df.describe())


# %%
"""
## Find missing values

Missing values are ...

From this, you can see what features are missing per column
"""
table = sidekick.drop_duplicates(df)
display(sidekick.show_summary(table))
"""
Here we have a couple of options to deal with it:

- The easiest and likely safest way to do it is to drop all rows with missing values. The con with this approach is that your dataset becomes vastly smaller, and as much data as possible for training is always a good thing!
- For continuos values (measurements, prices, age). A good praxis is to replace it with the "mean" of all values
- For categorical values (colors, dog breed, a persons sex, Insurance rating). Those values have predefined categories, and there, using the "mode" (most common category) is the most common
- When you want to set an explicit value, you can also provide that using "replace" with a specific "value"
- For a simple solution

- We also provide an "auto" imputation, which will try to guess the type of data you have and choose and apply an imputation method for you.

The columns to impute can be on multiple columns at once. You can also do it on all of the columns, but we then need to define which column should be your target. The reason is the target (what you want to learn a model to predict)

If you are unsure what type data your columns contain, you can find by inspecting the column `Column Type` by calling:
```python
sidekick.show_summary(table)
```

- For categories of type float64 and sometimes int64, the "mean" or "replace" imputation is the most likely.
- For string, object and sometimes int64 typed columns, the "mode" inputation is the most likely to use.
"""
# %%
# #### Mean imputation
# For the first value, we see that the values are continous
# Therefore, apply the "mean" imputation
display(table["Annual Income (USD)"].head())
table = sidekick.impute_values(table, columns="Annual Income (USD)", method="mean")

# %%
# We can see now that the values have been filled in
display(table["Annual Income (USD)"].head())

# %%
# #### TODO: Mode imputation
display(sidekick.show_summary(table))
display(table["Industry"].head())
table = sidekick.impute_values(table, columns="Industry", method="mode")
display(table["Industry"].head())

# %%
# #### TODO: Replace with values imputation
display(table["Previously Bought insurance"])
table = sidekick.impute_values(
    table, columns="Previously Bought insurance", method="replace", value=0
)
display(table["Previously Bought insurance"].head())

# %%
# #### TODO: drop missing values
display(table["Insurance Score"].head())
table = sidekick.impute_values(table, columns="Insurance Score", method="drop")

# %%
# There also is a method for dropping missing values
table = sidekick.impute_values(table, columns=["Previously Bought insurance"])

# Or even drop multiple columns
table = sidekick.impute_values(
    table, columns=["Previously Bought insurance", "Insurance Score"]
)

# %%
# You can even drop all rows with missing values at once (two different, but valid ways)!
table = sidekick.impute_values(
    table, columns=table.columns, target="Insurance Level", method="drop"
)
table = sidekick.drop_missing_values(table)

# %%
# #### TODO: Auto impute
# We also have an automatic imputation, that will try to guess the imputation to be done per column.
# This, or dropping rows, is usually the simplest and fastest solution
# This method is set bu default, so passing `method='auto'` and leaving it out is equivalent
table = sidekick.impute_values(table, columns=table.columns, target="Insurance Level")
table = sidekick.impute_values(
    table, columns=table.columns, target="Insurance Level", method="auto"
)

# %%
