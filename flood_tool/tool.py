"""Example module in template package."""

import os

import numpy as np
import pandas as pd
from .median_price import *
from .geo import *
from .flood_prob import *
from .local_authority import *


__all__ = ['Tool']


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='', sample_labels='',
                 household_file=''):

        """
        Parameters
        ----------

        full_postcode_file : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        household_file : str, optional
            Filename of a .csv file containing information on households
            by postcode.
        """

        if postcode_file == '':
            full_postcode_file = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_unlabelled.csv'))

        if household_file == '':
            household_file = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          'households_per_sector.csv'))

        self.postcodedb = pd.read_csv(full_postcode_file)

    def train(self, labelled_samples=''):
        """Train the model using a labelled set of samples.
        
        Parameters
        ----------
        
        labelled_samples : str, optional
            Filename of a .csv file containing a labelled set of samples.
        """

        if labelled_samples == '':
            labelled_samples = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_sample.csv'))

    def get_easting_northing(self, postcodes):
        """Get a frame of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NaN.
         """

        frame = self.postcodedb.copy()
        frame = frame.set_index('postcode')

        return frame.loc[postcodes, ['easting', 'northing']]

    def get_lat_long(self, postcodes):
        """Get a frame containing GPS latitude and longitude information for a
        collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NAN.
        """
        frame = self.postcodedb.copy()
        lat,lon = get_gps_lat_long_from_easting_northing(frame.easting, frame.northing)
        res = pd.DataFrame(lat,columns=['lat'],index = frame.postcode)
        res['lon'] = lon
        return res

    @staticmethod
    def get_flood_class_from_postcodes_methods():
        """
        Get a dictionary of available flood probablity classification methods
        for postcodes.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_postcode method.
        """
        models_dic = {'RandomForestRegressor':0,
                      'KNeighborsRegressor':1,
                      'XGBRegressor':2,
                      'GradientBoostingRegressor':3,
                      'BaggingRegressor':4,
                      'MLPRegressor':5}
        return models_dic

    def get_flood_class_from_postcodes(self, postcodes, method_num=0):
        """
        Generate series predicting flood probability classification
        for a collection of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            get_flood_class_from_postcodes_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by input postcodes.
        """
        
        model = FloodProbModel(selected_method=method_num)
        model.train_model()
        X_fetched = model.get_X(postcodes)
        X_pred = pd.Series(model.predict(X_fetched), index=[postcodes])
        
        return X_pred

    @staticmethod
    def get_flood_class_from_locations_methods():
        """
        Get a dictionary of available flood probablity classification methods
        for locations.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_class_from_OSGB36_locations and
             get_flood_class_from_OSGB36_locations method.
        """
        return {'all_zero_risk': 0}

    def get_flood_class_from_OSGB36_locations(self, eastings, northings, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of OSGB36_locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        if method == 0:
            return pd.Series(data=np.ones(len(eastings), int),
                             index=[(est, nth) for est, nth in
                                    zip(eastings, northings)],
                             name='riskLabel')
        else:
            raise NotImplementedError

    def get_flood_class_from_WGS84_locations(self, longitudes, latitudes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_class_from_locations_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        if method == 0:
            return pd.Series(data=np.ones(len(longitudes), int),
                             index=[(lng, lat) for lng, lat in
                                    zip(longitudes, latitudes)],
                             name='riskLabel')
        else:
            raise NotImplementedError

    @staticmethod
    def get_house_price_methods():
        """
        Get a dictionary of available flood house price regression methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_median_house_price_estimate method.
        """
        return {'all_england_median': 0}

    def get_median_house_price_estimate(self, postcodes, method=0):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_house_price_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """

        if method == 0:
            return pd.Series(data=np.full(len(postcodes), 245000.0),
                             index=np.asarray(postcodes),
                             name='medianPrice')
        else:
            median_price_model = MedianPriceModel('resources/postcodes_sampled.csv', method)
            median_price_pred = median_price_model.predict(postcodes=postcodes)
            return median_price_pred

    @staticmethod
    def get_local_authority_methods():
        """
        Get a dictionary of available local authorithy classification methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_altitude_estimate method.
        """
        return {'Do Nothing': 0, 'K-Neighbors': 1}

    def get_local_authority_estimate(self, eastings, northings, method=0):
        """
        Generate series predicting local authorities for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : int (optional)
            optionally specify (via a value in
            self.get_local_authority_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of local authorities indexed by eastings and northings.
        """

        if method == 0:
            return pd.Series(data=np.full(len(eastings), 'Unknown'),
                             index=[(est, nth) for est, nth in
                                    zip(eastings, northings)],
                             name='localAuthority')
        elif method == 1:
            filepath1 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))
            local_authority_model = LocalAuthorityModel(filepath1, method)
            local_authority_pred = local_authority_model.predict(eastings, northings)
            return local_authority_pred
        else:
            raise IndexError('Method should be either 0 or 1')

    def get_local_authority_estimate_postcodes(self, postcodes, method=0):
        """
        Generate series predicting local authorities for a sequence
        of postcodes

        Parameters
        ----------

        postcodes: sequence of strings
        method : int (optional)
            optionally specify (via a value in
            self.get_local_authority_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of local authorities indexed by postcodes.
        """
        east_north_df = get_easting_northing(self, postcodes)
        eastings = east_north_df['eastings']
        northings = east_north_df['northings']
    
        local_auth_east_north =  get_local_authority_estimate(eastings,northings,method=method)
        return local_auth_east_north.reset_index().set_index(east_north_df.index).drop(columns=['easting', 'northing'])

    def get_total_value(self, postal_data):
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.


        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcodesectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """

        raise NotImplementedError

    def get_annual_flood_risk(self, postcodes,  risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        risk_labels = risk_labels or self.get_flood_class(postcodes)

        cost = self.get_total_value(risk_labels.index)

        raise NotImplementedError
