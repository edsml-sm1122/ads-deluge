""" A module to define, train the Model, and predict Flood Probability."""

import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE

class FloodProbModel:
    """ Class for training selected model and predicting flood probability."""
    
    def __init__(self, selected_method=0):
        """
        Define a classifier model.
        
        Parameters
        ----------
        model: str
            Available option: 'KNN' for KNeighborsClassifier, 'RandomForest' for
            RandomForestClassifier, 'SVC' for Support Vector Classifier
        """
        filepath1 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))
        filepath2 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'))
        self.df1 = pd.read_csv(filepath1)
        self.df2 = pd.read_csv(filepath2)
        
        self.models_dic = {0:RandomForestRegressor(max_features=8, n_estimators=189, oob_score=True),
                      1:KNeighborsRegressor(n_neighbors=10),
                      2:XGBRegressor(),
                      3:GradientBoostingRegressor(),
                      4:BaggingRegressor(),
                      5:MLPRegressor()}
        
        if selected_method>=0 and selected_method<=6:
            self.model = self.models_dic[selected_method]
        else:
            raise IndexError('Method should be between 0 and 6')
        
    def train_model(self, oversample=False, accuracy_scoring=False):
        """
        Return trained model (using postcode_sampled.csv, training set) 
        and, if accuracy_scoring is set True, accuracy score (using test set) 

        Parameters
        ----------
        oversample: bool
            If true, target will be balanced using SMOTE method
        accuracy_scoring: bool
            If true, will return accuracy score alongside trained model
        
        ----------
        
        Returns
        -------
        trained model:
            selected trained classifier model
        accuracy score: float

        Example
        -------
        >>> model = FloodProbModel('KNN')
        >>> model.train_model()
        KNeighborsClassifier(), 0.752106
        """

        y = self.df1.riskLabel
        X = self.df1.drop(columns=['postcode', 'sector', 'localAuthority', 'riskLabel', 'medianPrice'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

        preproc = ColumnTransformer([
            ('num_transformer', MinMaxScaler(), X_train.select_dtypes(include=np.number).columns),
            ('cat_transformer', OneHotEncoder(sparse=False), X_train.select_dtypes(exclude=np.number).columns)
        ])
        self.preproc = preproc

        self.preproc.fit(X_train)
        X_train = self.preproc.transform(X_train)
        X_test = self.preproc.transform(X_test)
        
        if oversample is True: 
           sm = SMOTE(k_neighbors=5,random_state = 42) 
           X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
           X_train, y_train = X_train_res, y_train_res
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        if accuracy_scoring is True:
            return(self.model, accuracy_score(y_test, y_pred))
        else:
            return self.model
        
    def predict(self, X_input):
        """
        Return flood probability  
        
        Parameters
        ----------
        X_input: pd.Dataframe
            with [easting, northing, altitude, soilType] as columns
        
        ----------
        
        Returns
        -------
        flood probability: 1D ndarray with length = row of X_input

        Example
        -------
        >>> model = FloodProbModel('KNN')
        >>> model.train_model()
        >>> t = pd.DataFrame([[469395.0, 108803.0, 30, 'Planosols']], 
        >>>     columns=['easting', 'northing', 'altitude', 'soilType'])
        >>> model.predict(t)
        array([1])
        
        >>> model = FloodProbModel('KNN')
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
        Return pd.DataFrame 
        with [easting, northing, altitude, soilType] as columns 
        
        Parameters
        ----------
        postcode: str
        
        ----------
        
        Returns
        -------
        pd.DataFrame with [easting, northing, altitude, soilType] as columns

        Example
        -------
        >>> model = FloodProbModel('KNN')
        >>> model.get_X('PO7 8PR')
           easting   northing  altitude  soilType
        0  469395.0  108803.0  30        Planosols
        """
    
        df1 = self.df1.drop(columns=['sector', 'localAuthority', 'riskLabel', 'medianPrice'])
        df2 = self.df2.drop(columns=['sector', 'localAuthority'])
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
        Return np.array
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