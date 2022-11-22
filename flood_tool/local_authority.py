import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class LocalAuthorityModel:

    def __init__(self, path, method):
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(path)
        self.model = self.create_pipeline(method)
        self.model = self.train_model()
        return None

    def load_data(self, path):
        # Load data
        data = pd.read_csv(path)
        data = data.drop_duplicates()

        #Define the target and the features used to do the prediction
        y = data.localAuthority
        print(y.shape)
        X = data[['easting', 'northing']]
        print(X.shape)

        #Encode target
        y = LabelEncoder().fit_transform(y)

        return train_test_split(X, y, train_size=0.8)

    def create_pipeline(self, method):
        method_dict = {1: KNeighborsClassifier()}
        pipe = Pipeline([
            ('scaler', None),
            ('model', method_dict[method])
        ])

        return pipe

    def train_model(self):
        grid_dict = {'scaler':[None, MinMaxScaler()],
                    'model__n_neighbors': [1,2,3,5,10]}

        search = GridSearchCV(self.model, grid_dict, cv=5)
        search.fit(self.X_train, self.y_train)

        return search.best_estimator_

    def predict(self, eastings, northings):
        new_samples = pd.DataFrame([eastings, northings], index=['easting', 'northing']).T
        print(new_samples)
        pred = self.model.predict(new_samples)

        return pd.Series(pred, index=[(est, nth) for est, nth in
                                    zip(eastings, northings)],
                             name='localAuthority')

