"""
@author - Mohsin
"""
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd
import string
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

tqdm.pandas(tqdm)

from utils import *
from TextStatsTransformer import TextStatsTransformer
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    LOGGER_FILE = "prepTextStats1.log"

    IMAGE_FILE_1 =  "../utility/df_image_feats3.csv"

    ######################   Logger   #########################################
    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    ###################### Read data ##########################################
    logger.info("Reading data")
    train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'])
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'])
    test['deal_probability'] = -1

    # City correction
    for df in train, test:
        df['city'] = df['region'].astype(str) + "_" + df["city"].astype(str)
        df = df.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    #del train, test

    ################### Read Image data #######################################
    logger.info("Getting punctuations and emojis")
    punct = set(string.punctuation)

    emoji = set()
    for s in train['title'].fillna('').astype(str):
        for c in s:
            if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                continue
            emoji.add(c)

    for s in train['description'].fillna('').astype(str):
        for c in str(s):
            if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                continue
            emoji.add(c)
    logger.info(''.join(emoji))

    txtstats_title = TextStatsTransformer(ratios=True, puncts=punct, emojis=emoji, logger=logger)
    train_title = txtstats_title.fit_transform(train["title"].astype(str))
    test_title = txtstats_title.transform(test["title"].astype(str))

    title_feats = ["title_" + col for col in txtstats_title.get_feature_names()]
    train_title = pd.DataFrame(train_title, columns = title_feats)
    test_title = pd.DataFrame(test_title, columns = title_feats)
    logger.info("Feature in title text stats transformer are {}".format(train_title.columns))

    txtstats_desc = TextStatsTransformer(ratios=True, puncts=punct, emojis=emoji, logger=logger)
    train_desc = txtstats_desc.fit_transform(train["description"].astype(str))
    test_desc = txtstats_desc.transform(test["description"].astype(str))

    desc_feats = ["desc_" + col for col in txtstats_desc.get_feature_names()]
    train_desc = pd.DataFrame(train_desc, columns = desc_feats)
    test_desc = pd.DataFrame(test_desc, columns = desc_feats)
    logger.info("Feature in desc text stats transformer are {}".format(train_desc.columns))

    train_stats = pd.concat([train_title, train_desc], axis=1)
    test_stats = pd.concat([test_title, test_desc], axis=1)
    stat_feats = title_feats + desc_feats

    train_stats["title_desc_word_ratio"] = train_stats["title_num_words"] + (1 + train_stats["desc_num_words"])
    test_stats["title_desc_word_ratio"] = test_stats["title_num_words"] + (1 + test_stats["desc_num_words"])

    ################### Greedy forward feature selection ######################
    #
    TEXT_STATS = ['title_num_emojis',
                      'title_word_len_ratio', 'title_digits_len_ratio',
                      'title_caps_len_ratio', 'title_punct_len_ratio', 'desc_num_words',
                      'desc_unq_words', 'desc_num_digits', 'desc_num_emojis',
                      'desc_word_len_ratio',
                      'desc_digits_len_ratio', 'desc_caps_len_ratio',
                      'title_num_words', 'title_unq_words', 'title_desc_word_ratio']

    for col in TEXT_STATS:
        median = train_stats[col].median()
        train_stats[col] = train_stats[col].fillna(median)
        test_stats[col] = test_stats[col].fillna(median)

    minmax = MinMaxScaler((-1, 1))
    train_stats[TEXT_STATS] = minmax.fit_transform(train_stats[TEXT_STATS])
    test_stats[TEXT_STATS] = minmax.transform(test_stats[TEXT_STATS])


    logger.info("Saving image features 1")
    for col in TEXT_STATS:
        np.save("../utility/X_train_{}.npy".format(col), train_stats[col].values)
        np.save("../utility/X_test_{}.npy".format(col), test_stats[col].values)
    handler.close()
    logger.removeHandler(handler)
