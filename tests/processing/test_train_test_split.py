import pandas as pd
import pytest

import sidekick


def test_train_test():
    df = pd.read_csv(
        "https://storage.googleapis.com/bucket-8732/datalibrary/forecast_sales.csv"
    )
    df.drop_duplicates()
    df = sidekick.drop_missing_values(df)

    train, test = sidekick.split(df, test_size=0.2, shuffle=True, stratify=df["Day"])
    test, valid = sidekick.split(test, test_size=0.5)

    assert "Split" not in df.columns

    sidekick.add_split(df, dataset=train, split="Train")
    assert "Split" in df.columns
    assert df["Split"].isnull().sum() > 0

    sidekick.add_split(df, dataset=test, split="Test")
    sidekick.add_split(df, dataset=valid, split="Valid")
    assert df["Split"].isnull().sum() == 0
