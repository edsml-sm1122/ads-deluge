"""Test Module for median_price.py."""

import flood_tool
import numpy as np
import pandas as pd
import sklearn

from pytest import mark
from flood_tool import median_price

median_price_model = median_price.MedianPriceModel(method=1)
POSTCODES = ['TN6 3AW','BN1 5PF']   

def test_load_data():
    """Check """
    data = median_price_model.load_data()
    assert(np.asarray(data, dtype=object).shape==(4,))
    assert(len(data[0])==28000)
    assert(len(data[1])==12000)
    assert(len(data[2])==28000)
    assert(len(data[3])==12000)
    assert(issubclass(type(data[0]), pd.DataFrame))
    assert(issubclass(type(data[1]), pd.DataFrame))
    assert(issubclass(type(data[2]), pd.Series))
    assert(issubclass(type(data[3]), pd.Series))

def test_train_model():
    """Check """
    data = median_price_model.train_model()
    assert(len(data.named_steps)==2)
    assert(type(data[-1])==sklearn.neighbors._regression.KNeighborsRegressor)

def test_predict():
    """Check """
    data = median_price_model.predict(POSTCODES)
    assert(issubclass(type(data), pd.Series))
    assert(len(POSTCODES)==len(data))
    assert(data.index==POSTCODES).all()

if __name__ == "__main__":
    test_load_data()
    test_train_model()
    test_predict()