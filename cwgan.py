# encoding: utf-8  
import tensorflow as tf
import numpy as np
import os
import shutil
import logging
import dataset
from utils import Translator, seq2seq_onehot2label
'''
model size:
embedding matrix: num_symbols * embedding_size
generator:
    output_project = embedding_size * num_symbols
    cell: ??? relate to LSTM struct
discriminator:
    seq2state:
        cell: ??? may be should use the cell of generator
    state2logit:
        state_size * h1_size + h1_size + h1_size * h2_size + h2_size +h2_size * 1 + 1
'''
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log_file/emb_wgan.log',
                    filemode='w')
output_path = "./ckpt"
res_path = "./res"
'''
network config
'''
batch_size = 12
embedding_size = 96
num_layers = 2
num_symbols = 20000
state_size = 256
buckets = [(10,10),(20,20),(40,40)]
to_restore = False
max_len = buckets[-1][1]
learning_rate_dis = 5e-3#0.01
learning_rate_gen = 5e-4#0.01
CLIP_RANGE =[-0.1,0.1]
CRITIC = 10
gen_critic = 1
max_epoch = 100000
keep_prob = tf.constant(1.0,tf.float32, name="keep_prob")
encoder_inputs = []
decoder_inputs = []
target_weights = []

def generator(encoder_inputs,decoder_inputs,target_weights,bucket_id,seq_len):
    def seq2seq_f(encoder,decoder):
        cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        w = tf.get_variable("proj_w", [embedding_size, num_symbols])
        b = tf.get_variable("proj_b", [num_symbols])
        output_projection = (w, b)
        outputs, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder,
                decoder,cell,num_symbols,num_symbols,embedding_size,output_projection=output_projection,
                feed_previous = True)
        trans_output = []
        for output in outputs:
            trans_output.append(tf.matmul(output,w) + b)
        return trans_output, state

    targets = decoder_inputs
    outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(
        	encoder_inputs, decoder_inputs, targets, 
            target_weights, buckets, seq2seq_f, 
            softmax_loss_function=None, 
            per_example_loss=False, name='model_with_buckets')
    patch = tf.convert_to_tensor([[0.0]*num_symbols] * batch_size)
    def f0(): 
        for _ in range(0,max_len-buckets[0][1]):
            outputs[0].append(patch)
        return tf.convert_to_tensor(outputs[0],dtype = tf.float32)
    def f1(): 
        for _ in range(0,max_len-buckets[1][1]):
            outputs[1].append(patch)
        return tf.convert_to_tensor(outputs[1],dtype = tf.float32)
    def f2(): 
        for _ in range(0,max_len-buckets[2][1]):
            outputs[2].append(patch)
        return tf.convert_to_tensor(outputs[2],dtype = tf.float32)
    r = tf.case({tf.equal(bucket_id, 0): f0,
                 tf.equal(bucket_id, 1): f1},
                default=f2, exclusive=True)
    return tf.nn.softmax(tf.reshape(r,[max_len,batch_size,num_symbols]))

def discriminator(seq_raw, keep_prob,seq_len,reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        cell = tf.contrib.rnn.BasicLSTMCell(state_size)
        emb_matrix = tf.get_variable(name='emb_matrix',
                                     shape=[num_symbols, embedding_size],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        emb_ans = tf.reduce_mean(tf.multiply(
            tf.reshape(seq_raw,[max_len, batch_size, num_symbols, 1]), emb_matrix), axis=2)
        print(emb_ans)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

        _,state = tf.nn.dynamic_rnn(cell, emb_ans,sequence_length=seq_len, initial_state=None, dtype=tf.float32,
                                               time_major=True)
        tmp_state = tf.convert_to_tensor(state[-1])  # 2*batch_size*emb_size
        h_state = tf.slice(tmp_state, [1, 0, 0], [1, batch_size, state_size])
        state = tf.reshape(h_state,[batch_size,-1])
        h1_size = 32
        w1 = tf.get_variable("w1", [state_size, h1_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable("b1", h1_size, initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(state, w1) + b1), keep_prob)
        w3 = tf.get_variable("w3", [h1_size, 1], initializer=tf.truncated_normal_initializer())
        b3 = tf.get_variable("b3", [1], initializer=tf.constant_initializer(0.0))
        h3 = tf.matmul(h1, w3) + b3
        return h3

def train():
    for l in xrange(buckets[-1][0]):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="encoder{0}".format(l)))
    for l in xrange(buckets[-1][1]):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],
                                                    name="decoder{0}".format(l)))
        target_weights.append(tf.placeholder(tf.float32, shape=[batch_size],
                                              name="weight{0}".format(l)))

    global_step = tf.Variable(0, name="global_step", trainable=False)
    true_ans = tf.placeholder(tf.int32, [max_len ,batch_size], name = "true_ans")
    seq_len = tf.placeholder(tf.int32, name="seq_len")
    bucket_id = tf.placeholder(tf.int32, name="bucket_id")

    #[seq_len * batch_size]
    with tf.variable_scope('generator'):
        fake_ans = generator(encoder_inputs,decoder_inputs,target_weights,
                                        bucket_id,seq_len)
    # 创建判别模型
    true_ans_one_hot = tf.one_hot(true_ans,num_symbols,on_value=1.0,off_value=0.0,axis=-1)

    y_data = discriminator(true_ans_one_hot,keep_prob=keep_prob,seq_len=seq_len)
    y_generated = discriminator(fake_ans,keep_prob=keep_prob,seq_len=seq_len, reuse=True)
    
    # 损失函数的设置
    d_loss_real = tf.reduce_mean(y_data)
    d_loss_fake = tf.reduce_mean(y_generated)
    d_loss = d_loss_fake - d_loss_real
    g_loss =  tf.reduce_mean(-y_generated)

    optimizer_dis = tf.train.RMSPropOptimizer(learning_rate_dis,name='RMSProp_dis')
    optimizer_gen = tf.train.RMSPropOptimizer(learning_rate_gen, name='RMSProp_gen')

    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "discriminator")
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "generator")

    d_trainer = optimizer_dis.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer_gen.minimize(g_loss, var_list=g_params)

    #clip discrim weights
    d_clip = [tf.assign(v,tf.clip_by_value(v, CLIP_RANGE[0], CLIP_RANGE[1])) for v in d_params]

    init = tf.global_variables_initializer()
    # Create a saver.
    saver = tf.train.Saver(var_list = None,max_to_keep = 5)

    sess = tf.Session()

    sess.run(init)
    sess.run(d_clip)
    #load previous variables
    if to_restore == True:
        print("reloading variables...")
        logging.debug("reloading variables...")
        ckpt = tf.train.get_checkpoint_state(output_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    if os.path.exists(output_path) == False:
            os.mkdir(output_path)

    get_data = dataset.DataProvider(pkl_path='./bdwm_data_token.pkl',
                            buckets_size=buckets,batch_size=batch_size)
    translator = Translator('./dict.txt')
    print("save ckpt")
    saver.save(sess, os.path.join(output_path, 'refine_model.ckpt'), global_step=global_step)
    for i in range(sess.run(global_step), max_epoch):
        data_iterator = get_data.get_batch()
        if i < 25 or i % 500 == 0:
            citers = 100
        else:
            citers = CRITIC
        for j in np.arange(citers):
            print("epoch:%s, dis iter:%s" % (i, j))
            logging.debug("epoch:%s, dis iter:%s" % (i, j))
            try:
                feed_dict, BUCKET_ID = data_iterator.next()
            except StopIteration:
                get_data = dataset.DataProvider(pkl_path='./bdwm_data_token.pkl',
                                                buckets_size=buckets, batch_size=batch_size)
                data_iterator = get_data.get_batch()
                feed_dict, BUCKET_ID = data_iterator.next()
            _,dis_loss,fake_value,true_value = sess.run([d_trainer,d_loss,d_loss_fake,d_loss_real],feed_dict=feed_dict)
            sess.run(d_clip)
            print("d_loss:{}".format(dis_loss))
            print("fake:{} true:{}".format(fake_value,true_value))
            logging.debug("d_loss:{}".format(dis_loss))
            logging.debug("fake:{} true:{}".format(fake_value,true_value))

        for j in np.arange(gen_critic):
            print("epoch:%s, gen iter:%s" % (i, j))
            logging.debug("epoch:%s, gen iter:%s" % (i, j))
            try:
                feed_dict, BUCKET_ID = data_iterator.next()
            except StopIteration:
                get_data = dataset.DataProvider(pkl_path='./bdwm_data_token.pkl',
                                            buckets_size=buckets, batch_size=batch_size)
                data_iterator = get_data.get_batch()
                feed_dict, BUCKET_ID = data_iterator.next()

            g_loss_val, _, d_loss_val = sess.run([g_loss, g_trainer, d_loss], feed_dict=feed_dict)
            logging.debug("g_loss:{} d_loss:{}".format(g_loss_val, d_loss_val))
            print("g_loss:{} d_loss:{}".format(g_loss_val, d_loss_val))

        #get gen val for the true bucket
        gen_val = sess.run([fake_ans], feed_dict=feed_dict)
        print(gen_val)
        #translator.translate_and_print(seq2seq_onehot2label(gen_val),logger = logging)
        print("save ckpt")
        logging.debug("save ckpt")
        saver.save(sess,os.path.join(output_path,'model.ckpt'),global_step=global_step)
        
if __name__ == '__main__':
    train()