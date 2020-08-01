import math


def evaluate(rank_list, test_id, K):
    # rank_list: [#users, #candidate_items]
    # test_id: ID of the ground-truth item
    # K: rank list is truncated at K
    hit_ratio = 0
    ndcg = 0
    for line in rank_list:
        rec_list = line[:K]
        if test_id in rec_list:
            hit_ratio += 1
            ndcg += math.log(2) / math.log(rec_list.index(test_id) + 2)
    hit_ratio = hit_ratio / len(rank_list)
    ndcg = ndcg / len(rank_list)
    return hit_ratio, ndcg
