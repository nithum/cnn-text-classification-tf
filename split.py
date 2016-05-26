import pandas as pd
import numpy as np
from baselines import tidy_labels, load_cf_data, plurality
from ngram import get_labeled_comments
from sklearn.cross_validation import train_test_split

d_b, d_r = load_cf_data()
task = 'recipient'
l_b = plurality(d_b[task].dropna())
l_r = plurality(d_r[task].dropna())
data_b = get_labeled_comments(d_b, l_b)
data_r = get_labeled_comments(d_r, l_r)
b_train, b_test = train_test_split(data_b, test_size = 0.2, random_state=0)
r_train, r_test = train_test_split(data_r, test_size = 0.2, random_state=0)
br_train = pd.concat([b_train, r_train])
br_test = pd.concat([b_test, r_test])

b_train.to_csv('data/b_train.csv')
b_test.to_csv('data/b_test.csv')
r_train.to_csv('data/r_train.csv')
r_test.to_csv('data/r_test.csv')
br_train.to_csv('data/br_train.csv')
br_test.to_csv('data/br_test.csv')