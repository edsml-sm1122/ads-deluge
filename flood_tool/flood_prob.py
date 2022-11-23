""" A module to define, train the Model, and predict Flood Probability."""

import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE #waiting for approval

class FloodProbModel:
    """ Class for training selected model and predicting flood probability."""
    
    def __init__(self, model='KNN'):
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
        self.model = model

        
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
        
        #if oversample == True: #not used until approved by instructor
        #    sm = SMOTE(k_neighbors=5,random_state = 42) 
        #    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        #    X_train, y_train = X_train_res, y_train_res

        if self.model == 'KNN':
            selected = KNeighborsClassifier()
        elif self.model == 'SVC':
            selected = SVC()
        elif self.model == 'RandomForest':
            selected = RandomForestClassifier()
        
        selected.fit(X_train, y_train)
        self.selected = selected
        y_pred = selected.predict(X_test)
        
        if accuracy_scoring == True:
            return(selected, accuracy_score(y_test, y_pred))
        else:
            return(selected)
        
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
        probability = self.selected.predict(X_input)
        
        return probability
    
    def get_X(self, postcode):
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
        
        X = data[data['postcode']==postcode].drop(columns='postcode')
        return X