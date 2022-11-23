import pandas as pd
import numpy as np
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
from scipy.stats import zscore

class MedianPriceModel:  
    def __init__(self, path, method):
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(path)
        self.model = self.create_pipeline(method)
        self.model = self.train_model()
        return None

    def load_data(self, path):
        # Load data
        df = pd.read_csv(path)
        df = df.drop_duplicates()

        # Convert prices to log
        df['logPrice'] = np.log(df.medianPrice)
        
        # Split and make X and Y
        train_set, test_set = train_test_split(df, test_size=0.3)
        X_train = train_set.drop(['riskLabel','medianPrice','logPrice'],axis=1)
        X_test = test_set.drop(['riskLabel','medianPrice','logPrice'],axis=1)
        y_train = train_set['logPrice']
        y_test = test_set['logPrice']

        # Remove negative altitudes
        X_train.loc[X_train['altitude']<0,'altitude'] = np.nan

        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self, method):
        method_dict = {1: KNeighborsRegressor()}

        # Transform columns
        num_pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', RobustScaler())])
        cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        preproc = ColumnTransformer([
            ('num_pipe', num_pipe, ['easting','northing','altitude']),
            ('cat_pipe', cat_pipe, ['soilType'])], remainder='drop')

        # Combine preprocessing and model
        pipe = Pipeline([('preproc',preproc), ('model', method_dict[method])])

        return pipe

    def train_model(self):
        param_grid = {'model__n_neighbors':[1,2,3,5,10,40,100],'model__weights':['uniform','distance'], 
        'model__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 'model__p':[1,2]}

        cv = GridSearchCV(self.model, scoring='neg_root_mean_squared_error', param_grid=param_grid, cv=5, n_jobs=-1)
        cv.fit(self.X_train, self.y_train)

        return cv.best_estimator_

    def predict(self, postcodes):
        unlabelled = pd.read_csv('./flood_tool/resources/postcodes_unlabelled.csv')
        new_data = unlabelled.set_index('postcode').loc[postcodes].reset_index()
        y_pred = self.model.predict(new_data)
        pred = np.exp(y_pred)

        return pd.Series(pred, index=postcodes)

