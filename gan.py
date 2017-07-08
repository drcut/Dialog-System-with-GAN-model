# encoding: utf-8  
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"

# 总迭代次数500
max_epoch = 500

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256

# generate (model 1)
'''
question is a tensor of shape [batch_size * max_sequence_len]
'''
def build_generator(question):
    with tf.variable_scope("generator"):
        tf.contrib.rnn.BasicLSTMCell(size)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)
        def seq2seq_f(encoder_inputs, decoder_inputs):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs, 
                decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols, 
                embedding_size, output_projection=None, feed_previous=True)
        '''
        input:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        returns:
        outputs
        A list of the same length as decoder_inputs of 2D Tensors 
        with shape [batch_size x num_decoder_symbols] containing the generated outputs
        so it seems like a one-hot coding,and although we don't need to use decoder_inputs
        as input,we should let it as long as possible to get enough result
        '''
        outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(encoder_inputs, decoder_inputs, targets, 
            weights, buckets, seq2seq, 
            softmax_loss_function=None, 
            per_example_loss=False, name='model_with_buckets')
    #[decoder_inputs_len x batch_size x num_decoder_symbols]
    return outputs

# discriminator (model 2)
def build_discriminator(true_ans, generated_ans, keep_prob):
    with tf.variable_scope("discriminator"):
        state_size = 512
        def sentence2state(sentence):
            #sentence:[max_time, batch_size, num_decoder_symbols]
            #sequence_length:[batch_size]
            with tf.variable_scope("rnn_encoder"):
                single_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
                if num_layers > 1:
                    cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)
                outputs, state = tf.nn.dynamic_rnn(cell, sentence, 
                    sequence_length=None, initial_state=None, 
                    time_major=True)
                return state #[batch_size, cell.state_size]
        def state2sigmoid(state):
            h1_size = 512
            h2_size = 512
            h3_size = 128
            with tf.variable_scope("state2sigmoid"):
                w1 = tf.Variable(tf.truncated_normal([state_size, h1_size], stddev=0.1),name="d_w1", dtype=tf.float32)
                b1 = tf.Variable(tf.zeros([h1_size]), name="d_b1", dtype=tf.float32)
                h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
                w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
                b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
                h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
                w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
                b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
                h3 = tf.matmul(h2, w3) + b3
                return tf.nn.sigmoid(h3)

        with tf.variable_scope("twinsNN") as scope:
            true_state = sentence2state(true_ans)
            scope.reuse_variables()
            fake_state = sentence2state(generated_ans)
            true_pos = state2sigmoid(true_state)
            fake_pos = state2sigmoid(fake_state)
    return true_pos, fake_pos

def train():
    
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)
    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    # 创建生成模型
    generated_ans = build_generator(question)
    # 创建判别模型
    y_data, y_generated = build_discriminator(true_ans, generated_ans, keep_prob)

    # 损失函数的设置
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_params = tf.get_collection(TRAINABLE_VARIABLES,scope = "discriminator")
    g_params = tf.get_collection(TRAINABLE_VARIABLES,scope = "generator")
    # 两个模型的优化函数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

    steps = 60000 / batch_size
    for i in range(sess.run(global_step), max_epoch):
        for j in np.arange(steps):
            print("epoch:%s, iter:%s" % (i, j))
            # 每一步迭代，我们都会加载256个训练样本，然后执行一次train_step
            question, ans = mnist.train.next_batch(batch_size)
            # 执行生成
            sess.run(d_trainer,
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            # 执行判别
            if j % 1 == 0:
                sess.run(g_trainer,
                         feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, "output/sample{0}.jpg".format(i))
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)

def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, "output/test_result.jpg")

if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()