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
        dicti_method = {'KNNClassifier':0,
                        'RandomForestClassifier':1,
                        'SVC':2
                       }
        return dicti_method

    def get_flood_class_from_postcodes(self, postcodes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of postcodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            get_flood_class_from_postcodes_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """

        if method == 0:
            selected_method = 'KNN'
        elif method == 1:
            selected_method = 'RandomForest'
        elif method == 2:
            selected_method = 'SVC'
        else:
            raise IndexError('Method should be between 0 and 2')
        
        model = FloodProbModel(selected_method)
        model.train_model()
        
        if isinstance(postcodes, str):
            X_fetched = model.get_X(postcodes)
        else:
            X = []
            for i in postcodes:
                X.append(model.get_X(i))
            X_fetched = pd.concat(X, ignore_index=True, axis=0)
        
        return pd.Series(model.predict(X_fetched), index=[postcodes])
       
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
        dicti_method = {'KNNClassifier':0,
                        'RandomForestClassifier':1,
                        'SVC':2
                       }
        return dicti_method

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
        return {'all_england_median': 0, 'KNN':1}

    def get_median_house_price_estimate(self, postcodes, method=1):
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
        if isinstance(postcodes, str):
                postcodes=[postcodes]
        if method == 0:
            return pd.Series(data=np.full(len(postcodes), 245000.0),
                             index=np.asarray(postcodes),
                             name='medianPrice')
        else:
            model = MedianPriceModel()
            return model.predict(postcodes)

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
        east_north_df = self.get_easting_northing(self, postcodes)
        eastings = east_north_df['eastings']
        northings = east_north_df['northings']
    
        local_auth_east_north =  self.get_local_authority_estimate(eastings,northings,method=method)
        return local_auth_east_north.reset_index().set_index(east_north_df.index).drop(columns=['easting', 'northing']) 

    def get_local_authority_estimate_latitude_longitude(self, phi, lam, method=0):
        """
        Generate series predicting local authorities for a sequence
        of postcodes

        Parameters
        ----------
        phi : Latitude in degrees or radians, sequence of floats
        lam : Longitude in degrees or radians, sequence of floats
        method : int (optional)
            optionally specify (via a value in
            self.get_altitude_methods) the regression
            method to be used.
        
        Returns
        -------

        pandas.Series
            Series of local_authority.
        """
        eastings, northings = get_easting_northing_from_gps_lat_long(self, phi, lam) 

        local_auth_east_north =  self.get_local_authority_estimate(eastings,northings,method=method)
        return local_auth_east_north.set_index((east, north) for east, north in zip(eastings, northings))

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
        filepath1 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))
        filepath2 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'))
        df1 = pd.read_csv(filepath1).drop(columns=['riskLabel', 'medianPrice'])
        df2 = pd.read_csv(filepath2)
        data = pd.concat([df1, df2], ignore_index=True, axis=0)
        data.drop_duplicates(inplace=True)
        data = data['postcode']
        
        if isinstance(postal_data, str):# if a string is passed, instead of a sequence
            postal_data = [postal_data] # we change the postal_data to be a 1 member list
        
        final_postcode_list = []
        n_house_list = []
        for p in postal_data:
            for i in data:
                if p in i:
                    final_postcode_list.append(i)
        median_series = self.get_median_house_price_estimate(final_postcode_list)
        
        filepath3 = os.sep.join((os.path.dirname(__file__), 'resources', 'households_per_sector.csv'))
        df_house = pd.read_csv(filepath3)
        df_house['HouseNumber'] = df_house['households'] / df_house['number of postcode units']
        
        counter = 0 # it turns out that postcode sector column in the file doesnt cover all the postcode
                    # so we will append 0 to n_house_list if house number is not found in the household file
        for j in final_postcode_list:
            for k in range(len(df_house)):
                if df_house['postcode sector'].iloc[k] in j:
                    n_house_list.append(df_house['HouseNumber'].iloc[k])
            counter += 1
            if len(n_house_list) < counter:
                n_house_list.append(0)
        
        df_final = pd.DataFrame(data=median_series)
        df_final['nb_houses'] = n_house_list
        df_final['total_value'] = df_final['medianPrice'] * df_final['nb_houses']
        
        return df_final['total_value']   

    def get_annual_flood_risk(self, postcodes, risk_labels=None):
        """
        Return a series of annual flood risk of a
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
        if isinstance(postcodes, str):
            postcodes = [postcodes]
        
        flood_prob_dic = {1:0.01, 2:0.05, 3:0.1, 
                          4:0.5, 5:1, 6:1.5, 7:2, 
                          8:3, 9:4, 10:5}
        risk = []
        total_value = Tool.get_total_value(self, postcodes)
        flood_class = Tool.get_flood_class_from_postcodes(self, postcodes)
        
        for l, m in zip(total_value, flood_class):
            r = 0.05 * l * flood_prob_dic[m]/100   #/100 for percent
            risk.append(r)

        return pd.Series(risk, index=postcodes)
        
    def get_postcode_from_OSGB36(self, eastings, northings):
        """
        Generate series with nearest postcode
        for a collection of OSGB36_locations.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.

        Returns
        -------

        pandas.Series
            Series of postcodes with easting and northing pair as multi-index.
        """
        if isinstance(eastings, float) | isinstance(eastings, int): #and we assume that if the easting is string, so is northing
            eastings = [eastings]
            northings = [northings]
        
        if len(eastings) != len(northings):
            raise IndexError('Length of eastings and northings is not same!')

        filepath1 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))
        filepath2 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'))
        df1 = pd.read_csv(filepath1)
        df2 = pd.read_csv(filepath2)
        df1 = df1.drop(columns=['sector', 'localAuthority', 'riskLabel', 'medianPrice'])
        df2 = df2.drop(columns=['sector', 'localAuthority'])
        data = pd.concat([df1, df2], ignore_index=True, axis=0)
        data.drop_duplicates(inplace=True)
        
        postcodes = []
        ser_index = []

        for i, j in zip(eastings, northings):
            data['distance'] = np.sqrt((data['easting']-i)**2 + (data['northing'] - j)**2)
            postcodes.append(data[data['distance'] == data['distance'].min()][['postcode']]['postcode'].iloc[0])
            ser_index.append((data[data['distance'] == data['distance'].min()][['easting']]['easting'].iloc[0],
                              data[data['distance'] == data['distance'].min()][['northing']]['northing'].iloc[0]
                             ))
        
        index = pd.MultiIndex.from_tuples(ser_index)
        
        return pd.Series(postcodes, index=index)

