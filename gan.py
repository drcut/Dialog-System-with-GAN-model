# encoding: utf-8  
import tensorflow as tf
import numpy as np
import os
import shutil

batch_size = 2
embedding_size = 16
max_len = 5
num_layers = 3
num_symbols = 20
state_size = 512
'''
test data
'''
ini_question = [[1,3],[2,3],[4,1],[5,3],[6,6]]
ini_ans = [[2,2],[1,9],[7,1],[2,1],[3,3]]
keep_prob = tf.constant(0.5,tf.float32, name="keep_prob")
# generate (model 1)
'''
question is a tensor of shape [batch_size * max_sequence_len]
'''
encoder_inputs = []
decoder_inputs = []
target_weights = []
def build_generator(encoder_inputs,decoder_inputs,target_weights):
    with tf.variable_scope("generator"):
        cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        def seq2seq_f(encoder_inputs, decoder_inputs):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs, 
                decoder_inputs, cell, num_symbols, num_symbols, 
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
        targets = decoder_inputs
          
        outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(
        	encoder_inputs, decoder_inputs, targets, 
            target_weights, [[5,5]], seq2seq_f, 
            softmax_loss_function=None, 
            per_example_loss=False, name='model_with_buckets')
    #[decoder_inputs_len x batch_size x num_decoder_symbols]

    # [batch_size * num_symbol]
    return outputs[0]

# discriminator (model 2)
def build_discriminator(true_ans, generated_ans, keep_prob):
    '''
    true_ans, generated_ans:[max_len,batch_size,num_symbol]
    '''
    with tf.variable_scope("discriminator"):
        
        def sentence2state(sentence):
            #sentence:[max_time, batch_size, num_decoder_symbols]
            #sequence_length:[batch_size]
            with tf.variable_scope("rnn_encoder"):
                cell = tf.contrib.rnn.BasicLSTMCell(state_size)
                if num_layers > 1:
                    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
                '''
                cell.state_size:[num_layer * 2] (for c and h)
                (LSTMStateTuple(c=512, h=512), 
                LSTMStateTuple(c=512, h=512), 
                LSTMStateTuple(c=512, h=512))
                '''
                outputs, state = tf.nn.dynamic_rnn(cell, sentence, 
                    sequence_length=None, initial_state=None,dtype=tf.float32,
                    time_major=True)
                #accumulate the state of each RNN layer
                return (tf.reduce_sum(state,axis = 0))
        def state2sigmoid(state):
            h1_size = 512
            h2_size = 256
            res = tf.reshape(state,[batch_size,-1])
            with tf.variable_scope("state2sigmoid"):
                #state * 2 stand for c and h in lstm_state
                w1 = tf.Variable(tf.truncated_normal([state_size*2, h1_size], stddev=0.1),name="d_w1", dtype=tf.float32)
                b1 = tf.Variable(tf.zeros([h1_size]), name="d_b1", dtype=tf.float32)
                h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(res, w1) + b1), keep_prob)
                w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="d_w2", dtype=tf.float32)
                b2 = tf.Variable(tf.zeros([h2_size]), name="d_b2", dtype=tf.float32)
                h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
                w3 = tf.Variable(tf.truncated_normal([h2_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
                b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
                h3 = tf.matmul(h2, w3) + b3
                return tf.nn.sigmoid(h3)

        with tf.variable_scope("twinsNN") as scope:
        #true_ans, generated_ans: [max_len,batch_size,num_symbol]
            true_state = sentence2state(true_ans)
            scope.reuse_variables()
            fake_state = sentence2state(generated_ans)
            true_pos = state2sigmoid(true_state)
            scope.reuse_variables()
            fake_pos = state2sigmoid(fake_state)
    return true_pos, fake_pos

def train():
    for l in xrange(5):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="encoder{0}".format(l)))
    for l in xrange(5):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="decoder{0}".format(l)))
        target_weights.append(tf.placeholder(tf.float32, shape=[batch_size],
                                              name="weight{0}".format(l)))

    global_step = tf.Variable(0, name="global_step", trainable=False)
    true_ans = tf.placeholder(tf.int32, [max_len ,batch_size], name = "true_ans")
    # 创建生成模型
    generated_ans = build_generator(encoder_inputs,decoder_inputs,target_weights)
    # 创建判别模型
    y_data, y_generated = build_discriminator(tf.one_hot(true_ans,num_symbols,
                                                            on_value=1.0,off_value=0.0,axis=-1,
                                                            dtype=tf.float32,name="onehot"), 
                                            tf.convert_to_tensor(generated_ans), keep_prob)
    # 损失函数的设置
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss =  - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "discriminator")
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "generator")

    gard = optimizer.compute_gradients(d_loss,var_list=d_params)
    print("gard ok")
    # 两个模型的优化函数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()
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
		        input_feed[encoder_inputs[l].name] = ini_question[l]
            for l in xrange(5):
		    	input_feed[decoder_inputs[l].name] = ini_ans[l]
		    	input_feed[target_weights[l].name] = [1.0,1.0]
            input_feed[true_ans.name] = ini_ans
            sess.run(d_trainer,feed_dict=input_feed)
            sess.run(g_trainer,feed_dict=input_feed)
        gen_val = sess.run(generated_ans, feed_dict=input_feed)
        print(gen_val)
        sess.run(tf.assign(global_step, i + 1))

if __name__ == '__main__':
    train()