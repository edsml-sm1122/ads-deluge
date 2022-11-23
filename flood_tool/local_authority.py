""" A module to define, train the Model, and predict the Local Authority"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class LocalAuthorityModel:
    """ Class for training selected model and predicting local authority."""

    def __init__(self, path, method):
        """
        Define a classifier model.
        
        Parameters
        ----------
        method: str
            Available option: 1 for KNeighborsClassifier       
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(path)
        self.model = self.create_pipeline(method)
        self.model = self.train_model()
        return None

    def load_data(self, path):
        """
        Defining the target data and spliting the features data
        Return a training set and a testing set
        
        Parameters
        ----------
        path: string
            The path of the dataset
        
        ----------
        
        Returns
        -------
        X_train:
            feature training set 
        X_test:
            feature testing set
        y_train:
            target training set
        y_test:
            target testing set
        trained model:
            selected trained classifier model
        accuracy score: float
        """

        # Load data
        data = pd.read_csv(path)
        data = data.drop_duplicates()

        #Define the target and the features used to do the prediction
        y = data.localAuthority
        X = data[['easting','northing']]

        #Encode target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        return train_test_split(X, y, train_size=0.8)

    def create_pipeline(self, method):
        """
        Return a model with the selected method
        Parameters
        ----------
        method: str
            Available option: 'K-Neighbors' for KNeighborsClassifier
        ----------
        
        Returns
        -------
        model:
            selected classifier model
        """

        method_dict = {0: KNeighborsClassifier()}
        pipe = Pipeline([
            ('scaler', None),
            ('model', method_dict[method])
        ])

        return pipe

    def train_model(self):
        """
        Return trained model
        ----------
        
        Returns
        -------
        trained model:
            Trained classifier model
        """
        grid_dict = {'scaler':[None, MinMaxScaler()],
                    'model__n_neighbors': [1,2,3,5,10]}

        search = GridSearchCV(self.model, grid_dict, cv=5, n_jobs=-1)
        search.fit(self.X_train, self.y_train)

        return search.best_estimator_

    def predict(self, eastings, northings):
        """
        Return local authority  
        
        Parameters
        ----------
        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        
        ----------
        
        Returns
        -------
            predicted local authority as a pd.Series

        Example
        -------
        >>> local_authority_model = LocalAuthorityModel(filepath, method=1)
        >>> local_authority_pred = local_authority_model.predict(eastings, northings)
        """
        new_samples = pd.DataFrame([eastings, northings], index=['easting','northing']).T

        pred = self.model.predict(new_samples)
        pred = self.label_encoder.inverse_transform(pred)

        index = pd.MultiIndex.from_tuples([(est, nth) for est, nth in zip(eastings,northings)])

        return pd.Series(pred, index=index,name='localAuthority')




    