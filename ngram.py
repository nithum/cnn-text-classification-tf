import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split, ShuffleSplit


def get_labeled_comments(d, labels):
    """
    Get comments corresponding to rev_id labels
    """
    c = d[['rev_id', 'clean_diff']].drop_duplicates(subset = 'rev_id')
    c.index = c.rev_id
    c = c['clean_diff']
    c.name = 'x'
    data = pd.concat([c, labels], axis = 1)

    # shuffle
    m = data.shape[0]
    np.random.seed(seed=0)
    shuffled_indices = np.random.permutation(np.arange(m))
    return data.iloc[shuffled_indices].dropna()

