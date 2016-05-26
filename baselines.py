import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from scipy.stats import entropy as kl
from sklearn.metrics import roc_auc_score, f1_score



def get_baseline_matrix(labels, k, agg_function, eval_function):

    """      
    Say we have 2k human scores for each comment. For each comment, for
    all (i,j) <= (k+1),     spit human scores into 2 non-overlapping sets of I and J
    of size i and j respectively. Compute     correlation between the mean of scores
    in set I and set J for all comments.     Intuitively,     correlation i,j tells
    us how good i humans are are predicting the labels of another group of j
    humans.

    As i increases, we expect to get better predictions and as j increases, we
    expect to get more predictable labels.

    To figure out how many humans we need to label each question, we should
    examine the diagonal of the matrix (where i=j) and pick a value of i=j where
    there are diminishing returns to going further down the diagonal.

    To figure out how hard we should try at building a machine learning model
    for labels that we got from aggregating j_0 human labels we can check the
    correlations for different values of i. We can interpret correlation (i,
    j_0) as how good an "ensemble" of i humans is at predicting the labels.

    So a model that can achieve correlation (1, j_0) is as good as a single
    human. Also, we would expect that a model should not beat correlation (j_0,
    j_0). If it does, then it overfit to the group and you should increase j0.

    """
    n = k-1
    m = int(np.ceil(k/2))

    labels = labels.dropna()
    groups = labels.groupby(labels.index)
    groups = [e[1] for e in groups if e[1].shape[0]>=k]
    
    print('Num comments with k labels', len(groups))
    
    r = pd.DataFrame(np.zeros((m, n)))
    r.index = r.index +1
    r.columns = r.columns +1

    for i in range(1, m+1):
        for j in range(i, n+1):
            if (i+j) > k:
                continue

            dis = []
            djs = []
            for g in groups:
                if g.shape[0] >= i+j:
                    g = g.iloc[np.random.permutation(len(g))]
                    dis.append(g[0:i])
                    djs.append(g[i:(i+j)])
                else:
                    print(i,j, g, "WARNING: Comment had less than k labels")

            di =  pd.concat(dis)
            dj = pd.concat(djs)

            scores_i = agg_function (di)
            scores_j = agg_function (dj)

            r.ix[i,j] = "%0.3f" % eval_function(scores_i,scores_j)
    return r


# Aggregation Functions

def average(l):
    """
    Average all labels with the same rev_id
    """
    s = l.groupby(l.index).mean()
    s.name = 'y'
    return s

def plurality(l):
    """
    Take the most common label from all labels with the same rev_id
    """
    s = l.groupby(l.index).apply(lambda x:x.value_counts().index[0])
    s.name = 'y'
    return s

def empirical_dist(l, w = 0.5, index = None):

    """
    Compute empirical distribution over all classes
    using all labels with the same rev_id
    """
    if not index:
        index = sorted(list(set(l.dropna().values)))
    data = {}
    for k, g in l.groupby(l.index):
        data[k] = g.value_counts().reindex(index).fillna(0) + w

    labels = pd.DataFrame(data).T
    labels = labels.fillna(0)
    labels = labels.div(labels.sum(axis=1), axis=0)
    return labels


# Evaluation Metrics

def pearson(x,y):
    return pearsonr(x,y)[0]

def roc_auc(pred, true):
    true = (true > 0.5).astype(float)
    return roc_auc_score(true, pred)

def optimal_f1(pred, true, step = 1):
    binary_true = (true > 0.5).astype(float)
    ts = [np.percentile(pred, p) for p in np.arange(0, 101, step)]
    f1s = []
    for t in ts:
        y_pred_t = pred >= t
        f1 = f1_score(binary_true, y_pred_t)
        # Note F1 should have a parabolic shape, so no need to continue when the score starts falling
        if len(f1s) > 0 and f1 < f1s[-1] :
            return f1s[-1]
        else:
            f1s.append(f1)

    return f1s[-1]



def cross_entropy(x, y):
    logy =  np.log(y)
    logy[np.isinf(logy)] = 0
    return - np.multiply(x,logy).sum(axis=1).mean()    

def kl_divergence(x, y):
    return kl(x.T, y.T).mean()

def tidy_labels(d):
    classes = ['not_attack', 'other', 'quoting', 'recipient', 'third_party']
    for e in classes:
        d[e] = d.is_harassment_or_attack.str.contains(e).astype(float)
    d['attack'] = d.is_harassment_or_attack.str.contains('|'.join(classes[1:])).astype(float)
    return d

def load_cf_data():
    blocked = [
                'annotated_onion_layer_5_rows_0_to_10000',              #annotated 7 times
                'annotated_onion_layer_5_rows_0_to_10000_raters_3',      #annotated 3 times
                'annotated_onion_layer_5_rows_0_to_5000_raters_20',     #annotated 20 times
                'annotated_onion_layer_5_rows_10000_to_50526_raters_10',#annotated 10 times
                'annotated_onion_layer_10_rows_0_to_1000',              #annotated 7 times
                'annotated_onion_layer_20_rows_0_to_1000',              #annotated 7 times
                'annotated_onion_layer_30_rows_0_to_1000',              #annotated 7 times
    ]

    random = [
                'annotated_random_data_rows_5000_to_10000',
                'annotated_random_data_rows_0_to_5000_raters_20',
    ]

    blocked_dfs = []
    for f in blocked:
        d = pd.read_csv('data/v4_annotated/%s.csv' % f)
        d = d.query('_golden == False')
        d.index = d.rev_id
        d['src'] = f
        blocked_dfs.append(d)

    random_dfs = []
    for f in random:
        d = pd.read_csv('data/v4_annotated/%s.csv' % f)
        d = d.query('_golden == False')
        d.index = d.rev_id
        d['src'] = f
        random_dfs.append(d)


    return tidy_labels(pd.concat(blocked_dfs)), tidy_labels(pd.concat(random_dfs))


