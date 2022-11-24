"""Test Module for median price estimate."""

import flood_tool
import numpy as np
import pandas as pd
import sklearn

from pytest import mark
from flood_tool import flood_prob

tool = flood_tool.Tool()
tool.train()
flood_prob_model = flood_prob.FloodProbModel()


POSTCODES = ['BN1 5PF', 'TN6 3AW']
METHODS = tool.get_flood_class_from_locations_methods()

def test_get_flood_class_from_postcodes_type():
    """Check that return type is a pd.Series and that its length is the same as the number of postcodes"""
    for method_test in METHODS.values():
        data = tool.get_flood_class_from_postcodes(POSTCODES, method=method_test)
        assert(issubclass(type(data), pd.Series))
        assert(len(data)==len(POSTCODES))

def test_get_flood_class_from_postcodes_index():
    """Check that indices of the returned series are the postcodes"""
    for method_test in METHODS.values():
        data = tool.get_flood_class_from_postcodes(POSTCODES, method=method_test)
        assert(data.index == POSTCODES).all()

if __name__ == "__main__":
    test_get_flood_class_from_postcodes_type()
    test_get_flood_class_from_postcodes_index()

