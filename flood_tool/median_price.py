import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import pickle

class MedianPriceModel():  
    """ Class for training selected model and predicting median house price."""
    def __init__(self, labelled_data='', unlabelled_data='', method=1):
        """
        Define and train regression model.
        
        Parameters
        ----------
        method: int (opt)
            Available options:  {1: KNeighborsRegressor()} 
        """
        self.method=method
        if labelled_data == '':
            labelled_data = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))
        else:
            labelled_data = labelled_data
        
        if unlabelled_data == '':
            unlabelled_data = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'))
        else:
            unlabelled_data = unlabelled_data

        self.labelled_df = pd.read_csv(labelled_data)
        self.unlabelled_df = pd.read_csv(unlabelled_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()
        self.trained_model = self.train_model()
        
        return None

    def load_data(self):
        """
        Load training data.

        Returns
        --------
        X_train:
            pandas.DataFrame of training data features 
        X_test:
            pandas.DataFrame of testing data features
        y_train:
            pandas.Series of target for training
        y_test:
            pandas.Series of target for testing

        Example
        -------
        >>> model = MedianPriceModel()
        >>> X_train, X_test, y_train, y_test = model.load_data()
        """

        # Load data
        df = self.labelled_df
        df = df.drop_duplicates()

        # Create column for log of prices
        df['logPrice'] = np.log(df.medianPrice)
        
        # Split data randomly and make X and Y
        train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
        X_train = train_set.drop(['riskLabel','medianPrice','logPrice'],axis=1)
        X_test = test_set.drop(['riskLabel','medianPrice','logPrice'],axis=1)
        y_train = train_set['logPrice']
        y_test = test_set['logPrice']

        # Remove negative altitudes
        X_train.loc[X_train['altitude']<0,'altitude'] = np.nan

        return X_train, X_test, y_train, y_test
    


    def train_model(self):
        """
        Train model on training data.

        Returns
        -------
        Trained pipeline.

        Example
        -------
        >>> model = MedianPriceModel(method=1)
        >>> trained_model = model.train_model()
        
        """
        method_dict = {1:KNeighborsRegressor(n_neighbors=5, algorithm='ball_tree', p=2, weights='distance')}
        
        # Create pipeline
        num_pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', MinMaxScaler())])
        preproc = ColumnTransformer([('num_pipe', num_pipe, ['easting','northing'])], remainder='drop')
        pipe = Pipeline([('preproc',preproc), ('model', method_dict[self.method])])

        # Train model and save to disk
        pipe.fit(self.X_train, self.y_train)
        filename = 'finalised_model.sav'
        pickle.dump(pipe, open(filename, 'wb'))

        return pipe

    def predict(self, postcodes):
        """
        Return median price estimate.

        Parameters
        ----------
        postcodes : sequence of strs
            Sequence of postcodes.
        
        Returns
        --------
        pandas.Series
            Series of median house price estimates indexed by postcodes.

        Example
        --------
        >>> model = MedianPriceModel(method=1)
        >>> model.predict(postcodes=['TN6 3AW'])
        TN6 3AW    141700.0
        dtype: float64
        """

        if isinstance(postcodes, str):
            postcodes=[postcodes]
        data1 = self.unlabelled_df
        data2 = self.labelled_df.drop(columns=['riskLabel', 'medianPrice'])
        new_data = pd.concat([data1, data2])
        new_data.drop_duplicates(inplace=True)
        new_data = new_data.set_index('postcode').loc[postcodes].reset_index()
        loaded_model = pickle.load(open('finalised_model.sav', 'rb'))
        y_pred = loaded_model.predict(new_data)
        pred = np.exp(y_pred)

        return pd.Series(pred, index=postcodes)