import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pprint import pprint

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


def get_binary_classifier_metrics(prob_pos, y_test):
    """
    http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    """

    ts = [np.percentile(prob_pos, p) for p in np.arange(0, 101, 1)]
    f1s = []
    ps = []
    rs = []

    for t in ts:
        y_pred_t = prob_pos >=t
        f1s.append(f1_score(y_test, y_pred_t))
        ps.append(precision_score(y_test, y_pred_t))
        rs.append(recall_score(y_test, y_pred_t))

    """
    plt.plot(ts, f1s, label = 'F1')
    plt.plot(ts, ps, label = 'precision')
    plt.plot(ts, rs, label = 'recal')
    plt.legend()
    """

    ix = np.argmax(f1s)
    
    # Note: slight change from original ngram.py 
    num_correct = (np.array(prob_pos >= ts[ix]) == np.array(y_test))
    accuracy = np.mean(num_correct)


    scores = {
                'optimal F1': f1s[ix],
                'precision @ optimal F1': rs[ix],
                'recall @ optimal F1': rs[ix],
                'roc': roc_auc_score(y_test, prob_pos),
                'accuracy': accuracy
    }

    
    print('threshold @ optimal F1:', ts[ix])
    pprint({k: '%0.3f' % v for k,v in scores.items()})

    # Note: slight change from original ngram.py    
    return ts[ix], scores