"""Test Module."""

import flood_tool
import numpy as np

from pytest import mark



tool = flood_tool.Tool()
tool.train()

def test_get_easting_northing():
    """Check that correct easting/northing is outputted from the postcode"""

    data = tool.get_easting_northing(['BN1 5PF'])

    assert np.isclose(data.iloc[0].easting, 530401.0).all()
    assert np.isclose(data.iloc[0].northing, 105619.0).all()

def test_get_lat_long():
    """Check that correct lat/lon is outputted from postcode"""

    data = tool.get_lat_long(['BN1 5PF'])

    assert np.isclose(data.iloc[0].lat, 50.8354, 1.0e-3).all()
    assert np.isclose(data.iloc[0].lon, -0.1495, 1.0e-3).all()

if __name__ == "__main__":
    test_get_easting_northing()
    test_get_lat_long()
