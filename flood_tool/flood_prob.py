""" A module to define, train the Model, and predict Flood Probability."""

import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE

class FloodProbModel:
    """ Class for training selected model and predicting flood probability."""
    
    def __init__(self, postcode_file='', postcode_prediction_file='', selected_method=0):
        """
        Define a regressor model with given postcode datasets.
        
        Parameters
        ----------
        postcode_file: str, optional
            Filename of a .csv file containing geographic location
            data for postcodes with labelled risklevel and medianprice.

        postcode_prediction_file : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        selected_method: int, optional
            Available option: 0 for RandomForestRegressor, 1 for KNeighborsRegressor, 
            2 for GradientBoostingRegressor, 3 for BaggingRegressor, 4 for MLPRegressor
        """
        
        self.models_dic = {0:RandomForestRegressor(max_features=8, n_estimators=189, oob_score=True),
                      1:KNeighborsRegressor(n_neighbors=10)}
                    #   2:GradientBoostingRegressor(),
                    #   3:BaggingRegressor(),
                    #   4:MLPRegressor()}
        
        if selected_method>=0 and selected_method<=6:
            self.model = self.models_dic[selected_method]
        else:
            raise IndexError('Method should be 0 or 1')

        if postcode_file == '':
            filepath1 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))
            self.df_postcodes_sampled = pd.read_csv(filepath1)
        elif postcode_file != '':
            self.df_postcodes_sampled = pd.read_csv(postcode_file)

        if postcode_prediction_file == '':
            filepath2 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'))
            self.df_postcodes_unlabelled = pd.read_csv(filepath2)
        elif postcode_prediction_file != '':
            self.df_postcodes_unlabelled = pd.read_csv(postcode_prediction_file)
        
        
    def train_model(self, oversample=False):
        """
        Return trained model (using postcode_sampled.csv, training set)

        Parameters
        ----------
        oversample: bool
            If true, target will be balanced using SMOTE method
        
        ----------
        
        Returns
        -------
        trained model:
            selected trained regressor model using given postcode dataset.

        Example
        -------
        >>> model = FloodProbModel()
        >>> model.train_model()
        """

        y_train = self.df_postcodes_sampled['riskLabel']
        X_train = self.df_postcodes_sampled.drop(columns=['postcode', 'sector', 'localAuthority', 'riskLabel', 'medianPrice'])

        preproc = ColumnTransformer([
            ('num_transformer', MinMaxScaler(), X_train.select_dtypes(include=np.number).columns),
            ('cat_transformer', OneHotEncoder(sparse=False), X_train.select_dtypes(exclude=np.number).columns)
        ])
        self.preproc = preproc
        X_train = self.preproc.fit_transform(X_train)
        
        if oversample is True: 
           sm = SMOTE(k_neighbors=5,random_state = 42) 
           X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
           X_train, y_train = X_train_res, y_train_res
        
        self.model.fit(X_train, y_train)
        
        return self.model
        
    def predict(self, X_input):
        """
        Return flood probability  
        
        Parameters
        ----------
        X_input: pd.Dataframe
            with preporcessed [easting, northing, altitude, soilType] features as columns
        
        ----------
        
        Returns
        -------
        flood probability: 1D ndarray 
            with length = row of X_input 
            value = rounded predicted flood class (int)

        Example
        -------
        >>> model = FloodProbModel()
        >>> model.train_model()
        >>> t = pd.DataFrame([[469395.0, 108803.0, 30, 'Planosols']], 
        >>>     columns=['easting', 'northing', 'altitude', 'soilType'])
        >>> model.predict(t)
        array([1])
        
        >>> model = FloodProbModel()
        >>> model.train_model()
        >>> df = pd.read_csv('resources/postcodes_unlabelled.csv')
        >>> df.drop(columns=['postcode', 'sector', 'localAuthority'], inplace=True)
        >>> model.predict(df)
        array([1, 1, 1, ..., 1, 1, 6])
        """

        X_input = self.preproc.transform(X_input)
        probability = self.model.predict(X_input)
        probability_round = self.round_y_pred(probability)

        return probability_round
    
    def get_X(self, postcodes):
        """
        find corresponding [easting, northing, altitude, soilType] data in labelled and sampled dataset
        
        Parameters
        ----------
        postcode: str
        
        ----------
        
        Returns
        -------
        pd.DataFrame with [easting, northing, altitude, soilType] as columns

        Example
        -------
        >>> model = FloodProbModel()
        >>> model.get_X('PO7 8PR')
           easting   northing  altitude  soilType
        0  469395.0  108803.0  30        Planosols
        """
    
        df1 = self.df_postcodes_sampled.drop(columns=['sector', 'localAuthority', 'riskLabel', 'medianPrice'])
        df2 = self.df_postcodes_unlabelled.drop(columns=['sector', 'localAuthority'])
        data = pd.concat([df1, df2], ignore_index=True, axis=0)
        data.drop_duplicates(inplace=True)
        
        if isinstance(postcodes, str):
            X = data[data['postcode']==postcodes].drop(columns='postcode')
            return X
        else:
            X = []
            for i in postcodes:
                X.append(data[data['postcode']==i].drop(columns='postcode'))
            return pd.concat(X, ignore_index=True, axis=0)

    def round_y_pred(self, y_pred):
        """
        round the input dataset into corresponding class
        
        Parameters
        ----------
        y_pred: array
            with value of float
        
        ----------
        
        Returns
        -------
        array with value of integer
        """

        y_pred_round = []
        for num in y_pred:
            if num <= 1:
                y_pred_round.append(1)
            elif num >=10:
                y_pred_round.append(10)
            else:
                y_pred_round.append(round(num))

        return y_pred_round