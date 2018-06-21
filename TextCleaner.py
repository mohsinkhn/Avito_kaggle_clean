import re
from multiprocessing import cpu_count

import dask.dataframe as dd
import pandas as pd
import pymorphy2 #https://github.com/kmike/pymorphy2
from dask.multiprocessing import get
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from translate import translator

tqdm.pandas()

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self,
                 drop_punctuation = True,
                 drop_stopwords = True,
                 drop_number = True,
                 drop_newline = True,
                 drop_multispaces = True,
                 drop_vowels = False,
                 do_polymorph = False,
                 do_translate = False,
                 fill_na_with = "",
                 all_lower_case = True,
                 stopwords = stopwords.words("russian"),
                 use_multiprocessing = True):
        self.drop_punctuation = drop_punctuation
        self.drop_stopwords = drop_stopwords
        self.drop_number = drop_number
        self.drop_newline = drop_newline
        self.drop_multispaces = drop_multispaces
        self.drop_vowels = drop_vowels
        self.do_polymorph = do_polymorph
        self.do_translate = do_translate
        self.all_lower_case = all_lower_case
        self.fill_na_with = fill_na_with
        self.stopwords = stopwords
        self.use_multiprocessing = use_multiprocessing

    def fit(self, X, y = None):
        self.morph = pymorphy2.MorphAnalyzer()
        return self

    def transform(self, X, y = None):
        if not(isinstance(X, pd.Series)):
            X = pd.Series(X)
        if self.use_multiprocessing:
            xdd = dd.from_pandas(X, npartitions = cpu_count())
            x_transformed = xdd.map_partitions(lambda df: df.progress_apply(lambda x: self._transform(x))).compute(get=get)
        else:
            x_transformed = X.apply(self._transform)
        if self.fill_na_with:
            x_transformed = x_transformed.fillna(self.fill_na_with)
        return x_transformed

    def _transform(self, x):
        if self.all_lower_case:
            x = self._lower(x)
        if self.drop_punctuation:
            x = self._remove_punctuation(x)
        if self.drop_stopwords:
            x = self._remove_stopwords(x, self.stopwords)
        if self.drop_number:
            x = self._remove_number(x)
        if self.drop_newline:
            x = self._remove_newline(x)
        if self.drop_multispaces:
            x = self._substitute_multiple_spaces(x)
        if self.do_polymorph:
            x = ' '.join([self.morph.parse(word)[0].normal_form for word in x.split()])
        if self.drop_vowels:
            x = self._remove_vowels(x)
        if self.do_translate:
            x = translator('en', 'ru', x)

        return x

    def _remove_stopwords(self, x, stopwords):
        words = word_tokenize(x)
        words = [w for w in words if not w in stopwords]
        x = " ".join(words)
        return x

    def _remove_number(self, x):
        x = re.sub("\s\d{1,5}", " ", x)
        return x

    def _remove_vowels(self, x):
        x = x.translate(str.maketrans(dict.fromkeys('аэыуояеёюи', None)))
        return x

    def _lower(self, x):
        return x.lower()

    def _remove_punctuation(self, x):
        return re.sub(r'[^\w\s]', ' ', x)

    def _remove_newline(self, x):
        x = x.replace('\n', ' ')
        x = x.replace('\n\n', ' ')
        return x

    def _substitute_multiple_spaces(self, x):
        return ' '.join(x.split())

