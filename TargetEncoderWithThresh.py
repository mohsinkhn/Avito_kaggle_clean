import pandas as pd
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoderWithThresh(BaseEstimator, TransformerMixin):
    """
    A utlity class to help encode categorical variables using different methods.
    
    Inputs:
    cols: (List or str) Can be either a string or list of strings with column names
    targetcol: (str) Target column to encode column/group of columns with
    thresh: (int) Minimum count of grouping to encode (Acts as smoothing). Currently not implemented TODO
    func: (str or callable) Function to be applied on column/ group of columns to encode. 
          If str is provided, it should be a attribute of pandas Series
    cname: (str) Column name for new string
    func_kwargs: (dict) Additional arguments to be passed to function 
    add_to_orig: (bool) Whether to return dataframe with added feature or just the feature as series
    
    Output:
    pandas DataFrame/Series
    
    """
    def __init__(self, cols=None, targetcol=None, cname=None, thresh=0, func=np.mean,  add_to_orig=False, func_kwargs={}):                                                 
        self.cols = cols #Can be either a string or list of strings with column names
        self.targetcol = targetcol #Target column to encode column/group of columns with
        self.thresh = thresh  #Minimum count of grouping to encode (Acts as smoothing)
        self.func = func #Function to be applied on column/ group of columns to encode 
        self.add_to_orig = add_to_orig #Whether return a dataframe with added feature or just a series of feature
        self.cname = cname #Column to new feature generated
        self.func_kwargs = func_kwargs  #Additional key word arguments to be applied to func
    
    #@numba.jit        
    def fit(self, X, y=None):
            
        if isinstance(self.func, str):
            if hasattr(pd.Series, self.func):
                #print("here")
                vals = getattr(X.groupby(self.cols)[self.targetcol], self.func)
                self.dictmap = vals(**self.func_kwargs)
                
        else:
            self.dictmap = X.groupby(self.cols)[self.targetcol].apply(lambda x: self.func(x, **self.func_kwargs))
        
        self.counts = Counter(zip(*[X[col].tolist() for col in self.cols]))
        if len(self.cols) == 1:
            counts_greater_than_thresh = [k[0] for k, v in self.counts.items() if v >= self.thresh ]
        else:
            counts_greater_than_thresh = [k for k, v in self.counts.items() if v >= self.thresh ]
            
        self.dictmap = self.dictmap.loc[self.dictmap.index.isin(counts_greater_than_thresh)]
        
        if self.cname:
            self.dictmap.name = self.cname
        else:
            cname = ''
            cname = [cname + '_' +str(col) for col in self.cols]
            self.cname = '_'.join(cname) + "_" + str(self.func)
            self.dictmap.name = self.cname
            
        #print(self.cname)
        #self.dictmap = self.dictmap
        return self
    
    #@numba.jit
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_transformed = X[self.cols]
            
            X_transformed = X_transformed.join(self.dictmap, on=self.cols, how='left')[self.cname]

            if self.add_to_orig:
                return pd.concat([X, X_transformed], axis=1, copy=False)
            else:
                return X_transformed.values

        else:
            raise TypeError("Input should be a pandas DataFrame")