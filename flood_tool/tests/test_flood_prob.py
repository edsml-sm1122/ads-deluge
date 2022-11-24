"""Test Module for flood probability prediction."""

import flood_tool
import numpy as np
import pandas as pd
import sklearn

from pytest import mark
from flood_tool import flood_prob

tool = flood_tool.Tool()
flood_prob_model = flood_prob.FloodProbModel()

POSTCODES = ['BN1 5PF', 'TN6 3AW']
EASTINGS = [417997.0, 535049.0]
NORTHINGS = [97342.0, 169939.0]
LATITUDES = [52, 52.5]
LONGITUDES = [-1, 0]

CLASS_METHODS = tool.get_flood_class_from_postcodes_methods()
LCLASS_METHODS = tool.get_flood_class_from_locations_methods()

def test_length():
    """Check return series is same length as number of postcodes"""
    for method in CLASS_METHODS.values():
        data = tool.get_flood_class_from_postcodes(POSTCODES, method)
        assert(len(data)==len(POSTCODES))

def test_length2():
    """Check return series is same length as number of locations"""
    for method in LCLASS_METHODS.values():
        data = tool.get_flood_class_from_OSGB36_locations(EASTINGS, NORTHINGS, method)
        assert(len(data)==len(EASTINGS))

def test_length3():
    """Check return series is same length as number of locations"""
    for method in LCLASS_METHODS.values():
        data = tool.get_flood_class_from_WGS84_locations(LONGITUDES, LATITUDES, method)
        assert(len(data)==len(LONGITUDES))

if __name__ == "__main__":
    test_length()
    test_length2()
    test_length3()

