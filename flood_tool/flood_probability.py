import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_union
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from geo import *

def get_X(postcodes):
        df = pd.read_csv('resources/postcodes_sampled.csv')
        df[['lat','lon']] = pd.DataFrame(get_gps_lat_long_from_easting_northing(df['easting'],df['northing'], rads=False, dms=False)).T
        frame = df.copy()
        frame = frame.set_index('postcode')
        postcodes = np.array(postcodes)

        return frame.loc[postcodes, ['easting', 'northing','altitude','soilType','lat','lon']]

def preprocessing(df, prediction=False):
    if prediction == False:
        y = df['riskLabel']
        X = df.drop(columns=['riskLabel', 'medianPrice', 'sector', 'postcode','localAuthority'])
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 42)

        cat_pipe = make_column_transformer((OneHotEncoder(),X_train.select_dtypes(exclude=np.number).columns), remainder='drop')
        num_pipe = make_column_transformer((StandardScaler(),X_train.select_dtypes(include=np.number).columns), remainder='drop')
        preproc = make_union(cat_pipe, num_pipe)

        preproc.fit(X_train)
        X_train = preproc.transform(X_train)
        X_test = preproc.transform(X_test)

        l_enc = LabelEncoder().fit(y_train)
        y_train = l_enc.transform(y_train)
        y_test = l_enc.transform(y_test)

        return X_train, X_test, y_train, y_test
    elif prediction == True:
        df_all = pd.read_csv('resources/postcodes_sampled.csv')
        df_all[['lat','lon']] = pd.DataFrame(get_gps_lat_long_from_easting_northing(df['easting'],df['northing'], rads=False, dms=False)).T

        X = df_all.drop(columns=['riskLabel', 'medianPrice', 'sector', 'postcode','localAuthority'])
        X_train, X_test= train_test_split(X, test_size=0.3, random_state = 42)

        cat_pipe = make_column_transformer((OneHotEncoder(),X_train.select_dtypes(exclude=np.number).columns), remainder='drop')
        num_pipe = make_column_transformer((StandardScaler(),X_train.select_dtypes(include=np.number).columns), remainder='drop')
        preproc = make_union(cat_pipe, num_pipe)
        preproc.fit(X_train)
        
        X_pred = preproc.transform(df)

        return X_pred

def trained_model():
    df = pd.read_csv('resources/postcodes_sampled.csv')
    df[['lat','lon']] = pd.DataFrame(get_gps_lat_long_from_easting_northing(df['easting'],df['northing'], rads=False, dms=False)).T

    X_train, X_test, y_train, y_test = preprocessing(df) 

    knn_class = KNeighborsClassifier().fit(X_train, y_train)

    return knn_class

def model_predict(postcodes,method=0):
    if method == 0:
            return pd.Series(data=np.ones(len(postcodes), int), index=np.asarray(postcodes),name='riskLabel')
    else:
        model = trained_model()

        X_pred = get_X(postcodes)
        X_pred = preprocessing(X_pred)

        risklabel = model.predict(X_pred)

        return risklabel



