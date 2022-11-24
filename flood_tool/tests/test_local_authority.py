"""Test estimate local authority model"""
import sys
print(sys.path)

import flood_tool
import numpy as np
import pandas as pd

from pytest import mark



tool = flood_tool.Tool()
tool.train()

def test_estimate_local_authority():
    """Check that return type is a pd.Series and that its lenght is the same as input"""

    data = tool.get_easting_northing(['BN1 5PF'])
    print(data)

    local_authority_estimate = tool.get_local_authority_estimate(data.easting.tolist(), data.northing.tolist(), method=1)

    print(f'length of input: {len(data)}')
    print(f'lenght of output: {len(local_authority_estimate)}')
    assert len(data) == len(local_authority_estimate)
    assert type(local_authority_estimate) == pd.Series

if __name__ == "__main__":
    test_estimate_local_authority()