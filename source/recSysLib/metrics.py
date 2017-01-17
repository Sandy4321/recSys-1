import numpy as np

def roc_auc(ranked_list, pos_items):
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    ranks = np.arange(len(ranked_list))
    pos_ranks = ranks[is_relevant]
    neg_ranks = ranks[~is_relevant]
    auc_score = 0.0
    if len(neg_ranks) == 0:
        return 1.0
    if len(pos_ranks) > 0:
        for pos_pred in pos_ranks:
            auc_score += np.sum(pos_pred < neg_ranks, dtype=np.float32)
        auc_score /= (pos_ranks.shape[0] * neg_ranks.shape[0])
    assert 0 <= auc_score <= 1
    return auc_score


def precision(ranked_list, pos_items, at=None):
    ranked_list = ranked_list[:at]
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(ranked_list)
    assert 0 <= precision_score <= 1
    return precision_score


def recall(ranked_list, pos_items, at=None):
    ranked_list = ranked_list[:at]
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items.shape[0]
    assert 0 <= recall_score <= 1
    return recall_score
