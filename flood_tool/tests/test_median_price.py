"""Test Module for median price estimate."""

import flood_tool
import numpy as np
import pandas as pd
import sklearn

from pytest import mark
from flood_tool import median_price

tool = flood_tool.Tool()
tool.train()
median_price_model = median_price.MedianPriceModel(method=1)

POSTCODES = ['BN1 5PF', 'TN6 3AW']
METHODS = tool.get_house_price_methods()

def test_median_house_price_estimate_type():
    """Check that return type is a pd.Series and that its length is the same as the number of postcodes"""
    for method in METHODS.values():
        data = tool.get_median_house_price_estimate(POSTCODES, method=method)
        assert(issubclass(type(data), pd.Series))
        assert(len(data)==len(POSTCODES))

def test_median_house_price_estimate_index():
    """Check that indices of the returned series are the postcodes"""
    for method in METHODS.values():
        data = tool.get_median_house_price_estimate(POSTCODES, method=method)
        assert(data.index == POSTCODES).all()

def test_train_KNN_model():
    """Check KNN model is trained with method 1"""
    data = median_price_model.train_model()
    assert(len(data.named_steps)==2)
    assert(type(data[-1])==sklearn.neighbors._regression.KNeighborsRegressor)

if __name__ == "__main__":
    test_median_house_price_estimate_type()
    test_median_house_price_estimate_index()
    test_train_KNN_model()

