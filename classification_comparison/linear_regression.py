import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from utilities import identify_variables, split_dataframe


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

    # apply one hot encoding over categorical data
    one_hot = pd.get_dummies(df['Gender'])
    df = df.drop('Gender',axis = 1)
    df = df.join(one_hot)
    return df


if __name__ == "__main__":
    df = load_dataframe('./data/500_Person_Gender_Height_Weight_Index.csv')
    [X, y] = identify_variables(df, target='Index', excluded_columns=['Index', 'Index_health'])
    X_train, X_test, y_train, y_test = split_dataframe(X, y)

    # Create linear regression object
    regr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    # Train the model using the training sets
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

