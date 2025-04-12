import pandas as pd


URL_DATA = r'data\Real_estate.csv'


def rename_columns(df1):
    df1.rename(columns={
        'X2 house age': 'house age',
        'X3 distance to the nearest MRT station': 'nearest_station',
        'X4 number of convenience stores': 'number of stores',
        'X5 latitude': 'latitude',
        'X6 longitude': 'longitude',
        'Y house price of unit area': 'house price'
    }, inplace=True)
    return df1


def clean_data(path):
    """Function to read and clean data"""
    df = pd.read_csv(path, encoding = 'unicode_escape')
    df.drop(['No', 'X1 transaction date'],axis=1,inplace=True)
    data = rename_columns(df)
    return data



if __name__ == '__main__':
    data = clean_data(URL_DATA)
    data.to_csv('C:\Python Scripts\Projects_done\Real_estate_regression\prices_clean4.csv', index=False)
    print(data)
