"""
Real Estate price prediction - best model script.
Hyperparameter tuning of Random Forest model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import metrics


DATA_PATH = r'data\prices_clean.csv'


def read_data(path):
    """Function to read text data"""
    df = pd.read_csv(DATA_PATH)
    return df


def split_and_scale_data(df, target_column='house price', test_size=0.2, random_state=0):
    '''Function to split data on train and test set'''
   
    X = df.drop(columns=target_column)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def tune_random_forest_model(X_train, y_train):
    '''Hyperparameter tuning of Random Forest model'''
    
    rf_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'max_depth': list(range(1, 10)),
        'n_estimators': [100, 200, 500, 1000],
        'min_samples_split': [10, 30, 50]
    }
    grid_search = GridSearchCV(rf_model, param_grid, cv=4, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found:", grid_search.best_params_)
   
    best_rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    best_rf.fit(X_train, y_train)
    return best_rf


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("R^2 Score:", r2_score(y_test, predictions))
    print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, predictions)))
    print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, predictions))
    print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, predictions))


def main():
    df = read_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_and_scale_data(df)
    model = tune_random_forest_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    

if __name__ == '__main__':
    main()
    