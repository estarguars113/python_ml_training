import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from utilities import apply_one_hot_encoding, identify_variables, split_dataframe


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
    df = apply_one_hot_encoding(df, ['Gender'])
    return df



if __name__ == "__main__":
    df = load_dataframe('./data/500_Person_Gender_Height_Weight_Index.csv')
    [X, y] = identify_variables(df, target='Index', excluded_columns=['Index', 'Index_health'])
    X_train, X_test, y_train, y_test = split_dataframe(X, y)
    # Create linear regression object
    regr = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=100)

    # Train the model using the training sets
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print(y_pred)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
