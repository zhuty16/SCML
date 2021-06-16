import tensorflow as tf
import time
import random
import argparse
import numpy as np
from util import *
from SCML import SCML
from evaluate import evaluate


parser = argparse.ArgumentParser()
# hyperparameters for Ciao
parser.add_argument('--dataset', type=str, default='Ciao')
parser.add_argument('--num_factor', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=2020)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--dae_coef', type=float, default=1.0)
parser.add_argument('--ste_coef', type=float, default=1.0)
parser.add_argument('--emb_reg', type=float, default=1.0)
parser.add_argument('--output_reg', type=float, default=1.0)

# hyperparameters for Epinions
'''
parser.add_argument('--dataset', type=str, default='Epinions')
parser.add_argument('--num_factor', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--l2_reg', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=2020)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--dae_coef', type=float, default=1.0)
parser.add_argument('--ste_coef', type=float, default=1.0)
parser.add_argument('--emb_reg', type=float, default=1.0)
parser.add_argument('--output_reg', type=float, default=1.0)
'''
args = parser.parse_args()
print(vars(args))

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    [training_dict, test_dict, validation_dict, negative_dict, social_dict, num_user, num_item] = np.load('data/{dataset}/{dataset}.npy'.format(dataset=args.dataset), allow_pickle=True)
    print('num_user:%d, num_item:%d' % (num_user, num_item))

    rating_matrix_sparse = generate_sparse_rating_matrix(training_dict, num_user, num_item)
    social_matrix_sparse = generate_sparse_social_matrix(social_dict, num_user)
    validation_data = generate_test_data(validation_dict, negative_dict)
    test_data = generate_test_data(test_dict, negative_dict)

    print("Model preparing...")
    model = SCML(num_user, num_item, args)
    sess.run(tf.global_variables_initializer())

    print("Model training...")
    for epoch in range(1, args.num_epoch+1):
        print("epoch: %d" % epoch)
        time_start = time.time()
        user_batch = generate_user_batch(num_user, args.batch_size)
        training_dae_loss = list()
        training_ste_loss = list()
        for batch in range(len(user_batch)):
            dae_loss, ste_loss, _ = sess.run([model.dae_loss, model.ste_loss, model.train_op], feed_dict=feed_dict_training(model, user_batch[batch], rating_matrix_sparse, social_matrix_sparse, args.keep_prob))
            training_dae_loss.append(dae_loss)
            training_ste_loss.append(ste_loss)
        training_dae_loss = np.mean(training_dae_loss)
        training_ste_loss = np.mean(training_ste_loss)
        print("time: %.2fs" % (time.time() - time_start))
        print("training dae loss: %.2f, training ste loss: %.2f" % (training_dae_loss, training_ste_loss))

        batch_size_test = 1000
        rank_list_r = list()
        rank_list_s = list()
        for start in range(0, num_user, batch_size_test):
            r_hat_r, r_hat_s = sess.run([model.r_hat_ui_rating, model.r_hat_ui_social], feed_dict=feed_dict_test(model, validation_data, rating_matrix_sparse, social_matrix_sparse, start, start+batch_size_test))
            rank_list_r.extend(r_hat_r.argsort()[:, ::-1].tolist())
            rank_list_s.extend(r_hat_s.argsort()[:, ::-1].tolist())
        validation_hr_r, validation_ndcg_r = evaluate(rank_list_r, 0, 10)
        validation_hr_s, validation_ndcg_s = evaluate(rank_list_s, 0, 10)
        print("validation HR@10 for Mult-DAE: %.4f, validation NDCG@10 for Mult-DAE: %.4f" % (validation_hr_r, validation_ndcg_r))
        print("validation HR@10 for Mult-STE: %.4f, validation NDCG@10 for Mult-STE: %.4f" % (validation_hr_s, validation_ndcg_s))

    print("Model testing...")
    batch_size_test = 1000
    rank_list_r = list()
    rank_list_s = list()
    for start in range(0, num_user, batch_size_test):
        r_hat_r, r_hat_s = sess.run([model.r_hat_ui_rating, model.r_hat_ui_social], feed_dict=feed_dict_test(model, test_data, rating_matrix_sparse, social_matrix_sparse, start, start+batch_size_test))
        rank_list_r.extend(r_hat_r.argsort()[:, ::-1].tolist())
        rank_list_s.extend(r_hat_s.argsort()[:, ::-1].tolist())
    test_hr_r, test_ndcg_r = evaluate(rank_list_r, 0, 10)
    test_hr_s, test_ndcg_s = evaluate(rank_list_s, 0, 10)
    print("test HR@10 for Mult-DAE: %.4f, test NDCG@10 for Mult-DAE: %.4f" % (test_hr_r, test_ndcg_r))
    print("test HR@10 for Mult-STE: %.4f, test NDCG@10 for Mult-STE: %.4f" % (test_hr_s, test_ndcg_s))
