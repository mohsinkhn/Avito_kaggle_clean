from collections import Counter
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LabelEncodeWithThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, thresh = 2, logger = None):
        self.thresh = thresh
        self.encode_dict = None
        self.logger = logger 
        self.name = None
        
    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            cnt = Counter(X.astype(str).tolist())
        else:
            raise TypeError("Need pandas Series as input")
        
        self.name = X.name
        if self.logger:
            self.logger.info("Fitting dictionary with label to number mapping with threshold {}".format(self.thresh))
        self.encode_dict= {k:i+1 for i, (k, v) in enumerate(cnt.items()) if v >= self.thresh}
        return self
        
    def transform(self, X, y=None):
        if not(self.encode_dict):
            raise ValueError("Transformer not fitted yet")
        
        if not(isinstance(X, pd.Series)):
            raise TypeError("Need pandas Series as input")

        if self.logger:
            self.logger.info("Numerical encoding categorical data for {}".format(X.name))
            
        return X.astype(str).map(self.encode_dict).fillna(0).astype(int).values
    
    def get_feature_name(self):
        if self.name:
            return "{}_lbenc_{}".format(self.name, self.thresh)