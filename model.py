import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression

class Model:
    '''
    This a class that takes different types of regression models and
    gives back plots of each model, the R-squared score, and the cross-validation score.
    '''
    
    def __init__(self, model, X, y, title):
        '''
        Initialize model and variables
        Split the data into train and test groups
        and fit the model.
        Args:
        model : Regression model via PolynomialRegression()
        X: predictor data
        y: target data
        title: short name of the model
        '''
        if not isinstance(model, Pipeline):
            raise TypeError("Input model must be of type sklearn.pipeline.Pipeline.")
        
        if pd.DataFrame(y).shape[1] != 1:
            raise ValueError("Input y must only have one column.")
        
        self.m = model
        
        self.X = X
        self.y = y
        self.t = title
        self.n = len(self.X.columns)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
        self.m.fit(self.X_train, self.y_train)

        
    def __str__(self):
        '''
        Returns some information about the model.
        '''
        return "This is a {} model which predicts {} using the following predictors: {}.".format(self.t, self.y.name, ', '.join(self.X.columns))
    
    def plot(self):
        '''
        Makes regression plots for each feature in predictor data.
        '''
        xfit = np.linspace(0.01, self.X_train.max(), 1000 * self.n) # new x
        yfit = self.m.predict(xfit) # corresponding f(new x)

        fig, ax = plt.subplots(1, self.n, figsize=(8*self.n, 8))
        for i in range(self.n):
            ax[i].scatter(self.X_train.iloc[:,i], self.y_train, label = "data points", alpha = 0.5)
            ax[i].plot(xfit[:,i], yfit, 'r', label='prediction model')
            ax[i].set(xlabel=self.X.columns[i], ylabel=self.y.name)
        fig.suptitle("Predicted Values vs. Training Set")
        plt.legend(fontsize=20)
        plt.show()
        
    def score(self):
        '''
        Returns: The coefficient of determination of the model on the test set
        '''
        return self.m.score(self.X_test, self.y_test)
    
    def cv_score(self, cv = 10):
        '''
        Returns: the cross-validation score with given number of splits
        '''
        return cross_val_score(self.m, self.X_train, self.y_train, cv = 10).mean()


def data_encoder(data):
    """
    Encode all categorical values in the dataframe into numeric values.
    
    @param data: the original dataframe
    @return data: the same dataframe with all categorical variables encoded
    """
    le = preprocessing.LabelEncoder()
    cols = data.columns
    numcols = data._get_numeric_data().columns
    catecols = list(set(cols) - set(numcols))
    
    le = preprocessing.LabelEncoder()
    data[catecols] = data[catecols].astype(str).apply(le.fit_transform)
    
    return data

def PolynomialRegression(degree=2, **kwargs):
    '''
    Generate a polynomial regression model with given degree.
    '''
    return make_pipeline(preprocessing.PolynomialFeatures(degree), LinearRegression(**kwargs))
