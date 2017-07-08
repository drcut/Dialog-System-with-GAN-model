# encoding: utf-8  
import tensorflow as tf
import numpy as np
import os
import shutil

batch_size = 2
embedding_size = 16
max_len = 5
num_layers = 3
num_encoder_symbols = 20
num_decoder_symbols = 20
# generate (model 1)
'''
question is a tensor of shape [batch_size * max_sequence_len]
'''
def build_generator(question):
    with tf.variable_scope("generator"):
        single_cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * 3)
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
        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        for l in xrange(5):
        	encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="encoder{0}".format(l)))
        for l in xrange(5):
        	decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="decoder{0}".format(l)))
          	target_weights.append(tf.placeholder(tf.float32, shape=[batch_size],
                                              name="weight{0}".format(l)))

        targets = decoder_inputs
          
        outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(
        	encoder_inputs, decoder_inputs, targets, 
            target_weights, [[5,5]], seq2seq_f, 
            softmax_loss_function=None, 
            per_example_loss=False, name='model_with_buckets')
    #[decoder_inputs_len x batch_size x num_decoder_symbols]
    return outputs[0]

# discriminator (model 2)
def build_discriminator(true_ans, generated_ans, keep_prob):
    with tf.variable_scope("discriminator"):
        state_size = 512
        def sentence2state(sentence):
            print("sentence")
            print(sentence)#Tensor("true_ans:0", shape=(5, 2), dtype=int32)
            
            #sentence:[max_time, batch_size, num_decoder_symbols]
            #sequence_length:[batch_size]
            with tf.variable_scope("rnn_encoder"):
                single_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
                if num_layers > 1:
                    cell = tf.contrib.rnn.MultiRNNCell([single_cell] * 3)
                outputs, state = tf.nn.dynamic_rnn(cell, sentence, 
                    sequence_length=None, initial_state=None,dtype=tf.float32,
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
            scope.reuse_variables()
            fake_pos = state2sigmoid(fake_state)
    return true_pos, fake_pos

def train():
    ini_question = [[1,3],[2,3],[4,1],[5,3],[6,6]]
    ini_ans = [[2,2],[1,9],[7,1],[2,1],[3,3]]
    keep_prob = tf.constant(0.5,tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)
    question = tf.placeholder(tf.int32, [max_len ,batch_size], name="question")
    true_ans = tf.placeholder(tf.int32, [max_len ,batch_size], name = "true_ans")
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

#    saver = tf.train.Saver()
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)
    steps = 5
    max_epoch = 5
    for i in range(sess.run(global_step), max_epoch):
        for j in np.arange(steps):
            print("epoch:%s, iter:%s" % (i, j))
            batch_question, batch_ans = ini_question, ini_ans
            input_feed = {}
            for l in xrange(5):#[encoder_size*batch_size]
		        input_feed[self.encoder_inputs[l].name] = ini_question[l]
            for l in xrange(5):
		    	input_feed[self.decoder_inputs[l].name] = ini_ans[l]
		    	input_feed[self.target_weights[l].name] = 1.0
            sess.run(d_trainer,feed_dict=input_feed)
            sess.run(g_trainer,feed_dict=input_feed)
        gen_val = sess.run(generated_ans, feed_dict=input_feed)
        print(gen_val)
        sess.run(tf.assign(global_step, i + 1))

if __name__ == '__main__':
    train()