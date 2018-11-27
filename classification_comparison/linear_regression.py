import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def load_dataframe(path):
    df = pd.read_csv(path)

    # add extra column to make easier to understand the index value
    weight_dict = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity',
        5: 'Extreme Obesity'
    }
    df['Index_health'] = df.apply((lambda x: weight_dict.get(x['Index'], '')), axis=1)
    return df


def split_dataframe(df):
    # define target variable
    y = df['Index']

    # create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    return [X_train, X_test, y_train, y_test]


if __name__ == "__main__":
    df = load_dataframe('./data/500_Person_Gender_Height_Weight_Index.csv')
    X_train, X_test, y_train, y_test = split_dataframe(df)
    #print(X_train, X_test, y_train, y_test)