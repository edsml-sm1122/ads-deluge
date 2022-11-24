import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
from scipy.stats import zscore

class MedianPriceModel:  
    """ Class for training selected model and predicting median house price."""
    def __init__(self, method):
        """
        Define and train regression model.
        
        Parameters
        ----------
        method: int
            Available options:   {0: all of england median, 1: KNeighborsRegressor()} 

        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()
        self.method=method
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
        >>> model = MedianPriceModel(method=1)
        >>> X_train, X_test, y_train, y_test = model.load_data()
        """

        # Load data
        filepath1 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_sampled.csv'))
        df = pd.read_csv(filepath1)
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
        method_dict = {1:KNeighborsRegressor(n_neighbors=1, algorithm='ball_tree', p=2)}
        
        # Create pipeline
        num_pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', RobustScaler())])
        preproc = ColumnTransformer([('num_pipe', num_pipe, ['easting','northing','altitude'])], remainder='drop')
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
        postcodes : list of strs
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
        filepath2 = os.sep.join((os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'))
        df_unlabelled = pd.read_csv(filepath2)
        new_data = df_unlabelled.set_index('postcode').loc[postcodes].reset_index()
        loaded_model = pickle.load(open('finalised_model.sav', 'rb'))
        y_pred = loaded_model.predict(new_data)
        pred = np.exp(y_pred)

        return pd.Series(pred, index=postcodes)