import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def is_numeric(y):
    # assuming y is a series
    check = np.issubdtype(y.dtype, np.number)
    return check

class OneHotEncoder(BaseEstimator, TransformerMixin):
    # a simple 1-hot encoder to binarize top n values in each column
    def __init__(self, columns=[], n_values=10):
        self.n_values = n_values
        self.columns = columns
        self.categories = {}

    def fit(self, X, y=None):
        if len(self.columns) == 0:
            self.columns = [col for col in X.columns if not is_numeric(X[col])]
            print (self.columns)
        for col in self.columns:
            top_n = X[col].value_counts()[:self.n_values].index.tolist()
            self.categories[col] = top_n
        return self

    def transform(self, X, y=None):
        for col, intersting_values in self.categories.items():
            for value in intersting_values:
                X.loc[:, col + '_' + value] = X[col] == value
        X = X.drop(self.columns, axis=1)
        print (self.columns, 'transformed', len(self.columns),self.categories.items())
        return X
 
 def data_prepration(data_path, selected_columns=[], ignore_columns=[], target_column=''):
    '''
    selects Features and target column and build X and y for training purpose
    - treat each row as an independent training sample, no time series, no features based on other rows
    data: a system path to where csv file is stored 
    selected_columns: if empty list model will use all columns to figure out which one is important otherwise
                      user selected features only
    '''
    data = pd.read_csv(data_path)
    target = target_column if target_column != '' else data.columns[-1]
    if len(selected_columns):
        features = [col for col in selected_columns if (col != target) and (col not in ignore_columns)]
    else:
        features = [col for col in data.columns if (col != target) and (col not in ignore_columns)]
    y = data.loc[:, target]
    X = data.loc[:, features]
    return X, y
  
 
