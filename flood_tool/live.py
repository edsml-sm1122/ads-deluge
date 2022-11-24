"""Interactions with rainfall and river data."""

import numpy as np
import pandas as pd

__all__ = ["get_station_data_from_csv"]


def get_station_data_from_csv(filename, station_reference):
    """Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.

    >>> data = get_station_data_from_csv('resources/wet_day.csv')
    """
    frame = pd.read_csv(filename)
    frame = frame.loc[frame.stationReference == station_reference]

    return pd.to_numeric(frame.value.values)

def get_live_station_data(station_reference):
    """Return readings for a specified recording station from live API.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.

    >>> data = get_live_station_data('0184TH')
    """
    url='https://environment.data.gov.uk/flood-monitoring/id/stations/'+station_reference
    dfurl=pd.read_json(url)
    return pd.to_numeric(dfurl.loc['measures']['items']['latestReading']['value'])
