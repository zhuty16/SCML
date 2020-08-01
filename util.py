import numpy as np
import scipy.sparse as sp


def generate_sparse_rating_matrix(training_dict, num_user, num_item):
    row = []
    col = []
    for u in training_dict:
        for i in training_dict[u]:
            row.append(u)
            col.append(i)
    rating_matrix_sparse = sp.csr_matrix(([1 for _ in range(len(row))], (row, col)), (num_user, num_item)).astype(np.float32)
    return rating_matrix_sparse


def generate_sparse_social_matrix(social_dict, num_user):
    row = []
    col = []
    for u in social_dict:
        for v in social_dict[u]:
            row.append(u)
            col.append(v)
    social_matrix_sparse = sp.csr_matrix(([1 for _ in range(len(row))], (row, col)), (num_user, num_user)).astype(np.float32) + sp.eye(num_user)
    return social_matrix_sparse


def generate_user_batch(num_user, batch_size):
    user_batch = []
    user_list = list(range(num_user))
    np.random.shuffle(user_list)
    i = 0
    while i < len(user_list):
        user_batch.append(np.asarray(user_list[i:i+batch_size]))
        i += batch_size
    return user_batch


def generate_test_data(test_dict, negative_dict):
    test_data = []
    for u in test_dict:
        test_data.append([u, test_dict[u]] + negative_dict[u])
    test_data = np.asarray(test_data)
    return test_data


def feed_dict_training(model, user_batch, rating_matrix_sparse, social_matrix_sparse, keep_prob):
    feed_dict = dict()
    feed_dict[model.social_u] = np.array(social_matrix_sparse[list(user_batch)].todense())
    feed_dict[model.rating_u] = np.array(rating_matrix_sparse[list(user_batch)].todense())
    feed_dict[model.keep_prob] = keep_prob
    return feed_dict


def feed_dict_test(model, test_data, rating_matrix_sparse, social_matrix_sparse, start, end):
    feed_dict = dict()
    feed_dict[model.social_u] = np.array(social_matrix_sparse[list(test_data[start:end, 0])].todense())
    feed_dict[model.rating_u] = np.array(rating_matrix_sparse[list(test_data[start:end, 0])].todense())
    feed_dict[model.i] = test_data[start:end, 1:]
    feed_dict[model.keep_prob] = 1.0
    return feed_dict
