# %%
import pathlib

import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns


import sidekick

# %%
"""
TODO: Introduce the problem and describe what to do

- Describe the structure
- For each section
    - Describe the problem
    - Describe how to detect the problem
    - Describe how to fix the problem
"""

"""
##  Load the data
Provided synthetic data to show the steps often required to prepare a tabular dataset
"""
# %%
dataset_path = pathlib.Path("./synthetic_insurance_ruined.csv")
df = pd.read_csv(dataset_path, index_col=0)
display(df.head())

# %%
# Let's get some information about what our dataset looks like:
display(df.info())
display(df.describe())
# %%
"""
# Data Leakage (row level)
## What is the problem?
A model has been trained on leaky data is likely to perform poorly while the model will likely appeaer to perform deceptively well on the validation/test set

## How to deal with it
Simply applying the drop_duplicates method as below will remove any duplicate rows.
"""
# %%
table = sidekick.drop_duplicates(df)
# %%
"""
# Dataset  Split
Before proceeding with the following steps it is a good idea to define your datasets.
The reason for doing it at this point is that you would not want any imputed data in the validation/test sets.
Note: This is not an issue if you simply drop rows with missing values.
## How to split your data?
Applying the create_subsets function will creaete a new column: "Set", indicating which set the row belongs to. 
It will ensure that there is no missing data in the validation/test sets, note that this affeects the sizes of the sets.
"""
# %%
table = sidekick.create_subsets(table, test_size = 0.2, valid_size = 0.25)

# %%
"""
Aside from ensuring that there is no imputed data in your test/validation set, it is good to be aware of the data distributions for the different sets.
To start off, we can check out the size of the sets.
"""
# %%
display(sns.countplot(table["Set"]))
# %%
"""
We can also compare the distribution of the data for the different sets as such
"""
# %%
"""
Full data
"""
# %%
display(sns.countplot(table["Insurance Level"]))
# %%
"""
Train Set
"""
display(sns.countplot(table.loc[table["Set"]=="Train","Insurance Level"]))
# %%
"""
Validation Set
"""
display(sns.countplot(table.loc[table["Set"]=="Valid","Insurance Level"]))
# %%
"""
Test Set
"""
display(sns.countplot(table.loc[table["Set"]=="Test","Insurance Level"]))
# %%
"""
If there is a major difference between the distribution of the data in the difference sets particularily for your target column
It is worth it to considering using stratified sampling when creating the subsets, this is done using 
sidekick.create_subsets(table, stratify=table["col"]) where col is the name of the column to sample for.
"""
# %%
"""
For categorical columns countplot is useful but for numerical columns, histplot or distplot are preferable.
All columns can be visualized in this manner, you would simply change the "Insurance Level" to the column you wish to inspect.
Note: If the columns contain missing values the visualization methods might not work, how to deal with this is covered below.
"""
# %%
""" 
# Missing Values
## What is the problem?
It is impossible to train a model on data containing missing values.
## How to detect the problem?
Using the show_summary function, we can highligt if there are missing values and in which columns.
"""
# %%
display(sidekick.show_summary(table))
# %%
""""
## How to deal with it?

- The easiest and likely safest way to do it is to drop all rows with missing values. The con with this approach is that your dataset becomes vastly smaller, and as much data as possible for training is always a good thing!
- For continuos values (measurements, prices, age). A good praxis is to replace it with the "mean" of all values
- For categorical values (colors, dog breed, a persons sex, Insurance rating). Those values have predefined categories, and there, using the "mode" (most common category) is the most common
- When you want to set an explicit value, you can also provide that using "replace" with a specific "value"
- For a simple solution

- We also provide an "auto" imputation, which will try to guess the type of data you have and choose and apply an imputation method for you.
# TODO Never impute target column
The columns to impute can be on multiple columns at once. You can also do it on all of the columns, but we then need to define which column should be your target. The reason is the target (what you want to learn a model to predict)

If you are unsure what type data your columns contain, you can find by inspecting the column `Column Type` by calling:
```python
sidekick.show_summary(table)
```

- For categories of type float64 and sometimes int64, the "mean" or "replace" imputation is the most likely.
- For string, object and sometimes int64 typed columns, the "mode" inputation is the most likely to use.

Note: You should not impute data for your target column, it is preferable to drop the rows where the target value is missing.
"""
# %%
# #### Mean imputation
# For the first value, we see that the values are continous
# Therefore, apply the "mean" imputation
display(table["Annual Income (USD)"].head())
table = sidekick.impute_values(table, columns="Annual Income (USD)", method="mean") # TODO warning message

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
# After having imputed or removed the missing data, we make sure that we no longer have any
display(sidekick.show_summary(table))
# %%
# TODO column types
# TODO replace value in column ("1" vs 1)
# %%
# Class imbalance
## What is the problem?
# Having imbalanced classes can lead to deceptively good looking model performance, and undesirable model behaviour. 
# For example in a binary classification scenarion, if your data consists of 90% class 1 and 10 % class 2, a model that learns to always predict class 1 will have a 90% accuracy.
# %%
## How do you detect the problem?
# The simplest way is to plot your target column as we did above
# Also, The percentages of specific classes can be calculated as such:
# %%
display(df['Insurance Level'].value_counts(normalize=True))
# %% 
# In this dataset the data is clearly imbalanced, with the Silver class consisting of more data than the other two classes combined.
# %%
## How do you deal with the problem?
# There are a few ways of tackling imbalanced classes: See the imbalanced data sections here: https://peltarion.com/knowledge-center/documentation/datasets-view/data-preprocessing/tabular-data-preprocessing
# It is important to be aware that you have imbalanced classes, because this indicates that metrics such as accuracy are not good indicators of model performance.
# Here are some suggestions for more suitable metrics to focus here: https://peltarion.com/knowledge-center/documentation/evaluation-view/measure-performance-when-working-with-imbalanced-data

# %%
