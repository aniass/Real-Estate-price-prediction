'''Real Estate price prediction script'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_squared_error


URL_DATA = r'\data\Real_estate.csv'


def rename_columns(df):
    '''Rename columns in DataFrame'''
    df.rename(columns={
        'X2 house age': 'house age',
        'X3 distance to the nearest MRT station': 'nearest_station',
        'X4 number of convenience stores': 'number of stores',
        'X5 latitude': 'latitude',
        'X6 longitude': 'longitude',
        'Y house price of unit area': 'house price'
    }, inplace=True)
    return df


def clean_data(df):
    '''Function to clean data and drop unnecessary columns'''
    df.drop(['No', 'X1 transaction date'],axis=1,inplace=True)
    data = rename_columns(df)
    return data


def read_data(path):
    '''Function to read text data'''
    df = pd.read_csv(path, encoding = 'unicode_escape')
    df = clean_data(df)
    return df


def splitting_data(df, test_size=0.2, random_state=0):
    ''' Function to split data on train and test set'''
    X = df.drop(columns='house price')
    y = df.loc[:, 'house price']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def scores(model, X_test, y_test):
    '''Function to calculate R-squared and RMSE scores for the models'''
    pred = model.predict(X_test)
    r_squared = round(r2_score(y_test, pred), 2)
    rmse = round(np.sqrt(mean_squared_error(y_test,pred)), 2)
    return r_squared, rmse
    

def calculate_models(X_train, X_test, y_train, y_test):
    ''' Calculating models with scores'''
    models = pd.DataFrame()
    regressors = [
        LinearRegression(),
        Ridge(alpha=100, tol=0.0001, random_state=42),
        Lasso(alpha=5.1, positive=True, selection='random', random_state=42),
        ElasticNet(alpha= 0.1, l1_ratio=0.9, selection='random', random_state=42),
        SGDRegressor(early_stopping=True, validation_fraction=0.1, n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000, random_state=42),
        RandomForestRegressor(random_state=42),
        XGBRegressor(n_estimators=100, learning_rate=0.05, subsample=0.75,colsample_bytree=1, max_depth=7)]
     
    for regressor in regressors:
        model = Pipeline(steps=[('scaler', StandardScaler()),
                    ('regressor', regressor)])
        model.fit(X_train, y_train)
        
        r_squared, rmse = scores(model, X_test, y_test)
        param_dict = {
                     'Model': regressor.__class__.__name__,
                     'R-Squared_score':r_squared,
                     'RMSE': rmse
        }
        models = models.append(pd.DataFrame(param_dict, index=[0]))
        
    models.reset_index(drop=True, inplace=True)
    print(models.sort_values(by='R-Squared_score', ascending=False))


if __name__ == '__main__':
    df = read_data(URL_DATA)
    X_train, X_test, y_train, y_test = splitting_data(df)
    calculate_models(X_train, X_test, y_train, y_test)
