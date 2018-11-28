import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def identify_variables(df,target, columns=[], excluded_columns=[]):
    if(columns):
        X = np.array(df[columns])
    elif(excluded_columns):
        X = np.array(df[df.columns.difference(excluded_columns)])
        
    y = np.array(pd.DataFrame(df[target]))
    return [X, y]


def split_dataframe(X, y):
    # create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return [X_train, X_test, y_train, y_test]


def apply_one_hot_encoding(df, columns):
    one_hot = pd.get_dummies(df[columns])
    df = df.drop(columns, axis = 1)
    df = df.join(one_hot)
    return df



def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X.iloc[0], y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)