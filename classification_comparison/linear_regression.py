import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


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


def split_dataframe(df):
    # define target variable
    X = df[['Female', 'Male', 'Height', 'Weight']]
    y = pd.DataFrame(df['Index'])
    print(y)

    # create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return [X_train, X_test, y_train, y_test]


if __name__ == "__main__":
    df = load_dataframe('./data/500_Person_Gender_Height_Weight_Index.csv')
    X_train, X_test, y_train, y_test = split_dataframe(df)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
