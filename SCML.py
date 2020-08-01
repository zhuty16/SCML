'''
Social Collaborative Mutual Learning
@author: Tianyu Zhu
@created: 2020/8/1
'''
import tensorflow as tf


class SCML(object):
    def __init__(self, num_user, num_item, args):
        self.num_user = num_user
        self.num_item = num_item

        self.num_factor = args.num_factor
        self.l2_reg = args.l2_reg
        self.lr = args.lr
        self.dae_coef = args.dae_coef
        self.ste_coef = args.ste_coef
        self.emb_reg = args.emb_reg
        self.output_reg = args.output_reg

        with tf.name_scope('Mult-DAE'):
            # input
            self.rating_u = tf.placeholder(tf.float32, [None, self.num_item], name="rating_u") # user's rating vector
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # parameters
            # item embeddings
            self.W_in_rating = tf.Variable(tf.random_normal([self.num_item, self.num_factor], stddev=0.01), name="W_in_rating")
            self.W_out_rating = tf.Variable(tf.random_normal([self.num_factor, self.num_item], stddev=0.01), name="W_out_rating")

            # bias
            self.b_in_rating = tf.Variable(tf.zeros([1, self.num_factor]), name="b_in_rating")
            self.b_out_rating = tf.Variable(tf.zeros([1, self.num_item]), name="b_out_rating")

            # model
            # l2 normalize
            self.rating_u_normalized = tf.nn.l2_normalize(self.rating_u, 1)

            # user rating embedding
            self.hidden_layer_rating = tf.nn.tanh(tf.matmul(tf.nn.dropout(self.rating_u_normalized, self.keep_prob), self.W_in_rating) + self.b_in_rating)

            # reconstruction
            self.rating_u_hat_rating = tf.matmul(self.hidden_layer_rating, self.W_out_rating) + self.b_out_rating

            # softmax
            self.rating_u_hat_rating_normalized = tf.nn.log_softmax(self.rating_u_hat_rating)

            # loss
            self.dae_loss = -tf.reduce_mean(tf.reduce_sum(self.rating_u * self.rating_u_hat_rating_normalized, 1))

        with tf.name_scope('Mult-STE'):
            # input
            self.social_u = tf.placeholder(tf.float32, [None, self.num_user], name="social_u") # users' adjacency vectors

            # parameters
            # user embedding
            self.W_in_social = tf.Variable(tf.random_normal([self.num_user, self.num_factor], stddev=0.01), name="W_in_social")

            # item embedding
            self.W_out_social = tf.Variable(tf.random_normal([self.num_factor, self.num_item], stddev=0.01), name="W_out_social")

            # bias
            self.b_in_social = tf.Variable(tf.zeros([1, self.num_factor]), name="b_in_social")
            self.b_out_social = tf.Variable(tf.zeros([1, self.num_item]), name="b_out_social")

            # model
            # l2 normalize
            self.social_u_normalized = tf.nn.l2_normalize(self.social_u, 1)

            # user social embedding
            self.hidden_layer_social = tf.nn.tanh(tf.matmul(tf.nn.dropout(self.social_u_normalized, self.keep_prob), self.W_in_social) + self.b_in_social)

            # prediction
            self.rating_u_hat_social = tf.matmul(self.hidden_layer_social, self.W_out_social) + self.b_out_social

            # softmax
            self.rating_u_hat_social_normalized = tf.nn.log_softmax(self.rating_u_hat_social)

            # loss
            self.ste_loss = -tf.reduce_mean(tf.reduce_sum(self.rating_u * self.rating_u_hat_social_normalized, 1))

        with tf.name_scope('mutual_learning'):
            # embedding layer mutual regularization
            self.emb_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.hidden_layer_rating - self.hidden_layer_social), 1))

            # output layer mutual regularization
            self.output_loss = -tf.reduce_mean(tf.reduce_sum(tf.exp(self.rating_u_hat_social_normalized) * self.rating_u_hat_rating_normalized, 1))

        with tf.name_scope('training'):
            # l2 reg
            self.regularization = tf.nn.l2_loss(self.W_in_rating) + tf.nn.l2_loss(self.W_in_social) + tf.nn.l2_loss(self.W_out_social) + tf.nn.l2_loss(self.W_out_rating)

            # total loss
            self.loss = self.dae_coef * self.dae_loss + self.ste_coef * self.ste_loss + self.output_reg * self.output_loss + self.emb_reg * self.emb_loss + self.l2_reg * self.regularization

            # optimizer
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('test'):
            # candidate items
            self.i = tf.placeholder(tf.int32, [None, 100], name="i")

            # Mult-DAE prediction
            self.i_emb_rating = tf.nn.embedding_lookup(tf.transpose(self.W_out_rating), self.i)
            self.r_hat_ui_rating = tf.reduce_sum(tf.expand_dims(self.hidden_layer_rating, 1) * self.i_emb_rating, 2)

            # Mult-STE prediction
            self.i_emb_social = tf.nn.embedding_lookup(tf.transpose(self.W_out_social), self.i)
            self.r_hat_ui_social = tf.reduce_sum(tf.expand_dims(self.hidden_layer_social, 1) * self.i_emb_social, 2)
