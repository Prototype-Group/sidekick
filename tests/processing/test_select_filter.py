import pandas as pd
import pytest

import sidekick

COLUMN = "col1"
df = pd.DataFrame(data={"col1": [3, 14, -2, 28], "col2": [8, 21, 1, 34]})

#####################
#    Select
#####################
def test_select_value():
    out = sidekick.select_values(df, column=df.columns[0], values=[3])
    assert list(out[COLUMN]) == [3]


def test_select_multiple_values():
    out = sidekick.select_values(df, column=df.columns[0], values=[3, 28])
    assert list(out[COLUMN]) == [3, 28]


def test_select_value_not_exists():
    out = sidekick.select_values(df, column=df.columns[0], values=[-9999])
    assert list(out[COLUMN]) == []


def test_select_by_condition():
    out = sidekick.select_values(df, column=df.columns[0], conditions=["<4"])
    assert list(out[COLUMN]) == [3, -2]

    out = sidekick.select_values(df, column=df.columns[0], conditions=[">4"])
    assert list(out[COLUMN]) == [14, 28]


def test_select_multiple_by_condition():
    out = sidekick.select_values(df, column=df.columns[0], conditions=["<4", ">14"])
    assert list(out[COLUMN]) == [3, -2, 28]


def test_select_by_value_and_condition():
    out = sidekick.select_values(
        df, column=df.columns[0], values=[3], conditions=[">14"]
    )
    assert list(out[COLUMN]) == [3, 28]


#####################
#    Filter
#####################
def test_filter_value():
    out = sidekick.filter_values(df, column=df.columns[0], values=[3])
    assert list(out[COLUMN]) == [14, -2, 28]


def test_filter_multiple_values():
    out = sidekick.filter_values(df, column=df.columns[0], values=[3, 28])
    assert list(out[COLUMN]) == [14, -2]


def test_filter_value_not_exists():
    out = sidekick.filter_values(df, column=df.columns[0], values=[-9999])
    assert list(out[COLUMN]) == [3, 14, -2, 28]


def test_filter_by_condition():
    out = sidekick.filter_values(df, column=df.columns[0], conditions=["<4"])
    assert list(out[COLUMN]) == [14, 28]

    out = sidekick.filter_values(df, column=df.columns[0], conditions=[">4"])
    assert list(out[COLUMN]) == [3, -2]


def test_filter_multiple_by_condition():
    out = sidekick.filter_values(df, column=df.columns[0], conditions=["<4", ">14"])
    assert list(out[COLUMN]) == [14]


def test_filter_by_value_and_condition():
    out = sidekick.filter_values(
        df, column=df.columns[0], values=[3], conditions=[">14"]
    )
    assert list(out[COLUMN]) == [14, -2]
