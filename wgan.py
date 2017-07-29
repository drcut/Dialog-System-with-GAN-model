# encoding: utf-8  
import tensorflow as tf
import numpy as np
import os
import shutil
import dataset
from utils import Translator, seq2seq_onehot2label
#save path
output_path = "./ckpt"
res_path = "./res"
batch_size = 64
embedding_size = 128
max_len = 80
num_layers = 3
num_symbols = 20000
state_size = 512
buckets = [(5,5),(10,10),(20,20),(40,40),(80,80)]
to_restore = True
max_len = buckets[-1][1]
learning_rate = 0.0001
CLIP_RANGE =[-0.01,0.01]
CRITIC = 25
max_epoch = 5
'''
test data
'''
keep_prob = tf.constant(0.8,tf.float32, name="keep_prob")
# generate (model 1)
'''
question is a tensor of shape [batch_size * max_sequence_len]
'''
encoder_inputs = []
decoder_inputs = []
target_weights = []
BUCKET_ID = 0
def build_generator(encoder_inputs,decoder_inputs,target_weights,bucket_id,seq_len):
    global BUCKET_ID
    with tf.variable_scope("generator"):
        
        def seq2seq_f(encoder,decoder):
            cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

            # Sampled softmax only makes sense if we sample less than vocabulary size.
            w = tf.get_variable("proj_w", [embedding_size, num_symbols])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [num_symbols])
            output_projection = (w, b)
            outputs, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder,
                decoder,cell,num_symbols,num_symbols,embedding_size,output_projection=output_projection,
                feed_previous = True)
            trans_output = []
            for output in outputs:
                trans_output.append(tf.matmul(output,w) + b)
            #print("trans_output")
            #print(tf.argmax(trans_output,axis = 2))
            #output:[seq len * batch_size]
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
    def f3(): 
        for _ in range(0,max_len-buckets[3][1]):
            outputs[3].append(patch)
        return tf.convert_to_tensor(outputs[3],dtype = tf.float32)
    def f4(): 
        for _ in range(0,max_len-buckets[4][1]):
            outputs[4].append(patch)
        return tf.convert_to_tensor(outputs[4],dtype = tf.float32)

    r = tf.case({tf.equal(bucket_id, 0): f0,
                tf.equal(bucket_id, 1): f1,
                tf.equal(bucket_id, 2): f2,
                tf.equal(bucket_id, 3): f3},
                default=f4, exclusive=True)
    return tf.reshape(r,[max_len,batch_size,num_symbols])

# discriminator (model 2)
def build_discriminator(true_ans, generated_ans, keep_prob ,seq_len):
    '''
    true_ans, generated_ans:[max_len,batch_size,num_symbol]
    '''
    h1_size = 512
    h2_size = 256
    cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
    def seq2seq(sentence):
        outputs, state = tf.nn.dynamic_rnn(cell, sentence, 
                sequence_length=seq_len, initial_state=None,dtype=tf.float32,
                time_major=True)
        return state
    def state2logit(state):
        res = tf.reshape(state,[batch_size,-1])
        w1 = tf.get_variable("w1", [state_size*2, h1_size],initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b1", h1_size,initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(res, w1) + b1), keep_prob)
        w2 = tf.get_variable("w2", [h1_size, h2_size],initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("b2", [h2_size],initializer=tf.constant_initializer(0.0))
        h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
        w3 = tf.get_variable("w3", [h2_size, 1],initializer=tf.random_normal_initializer())
        b3 = tf.get_variable("b3", [1],initializer=tf.constant_initializer(0.0))
        h3 = tf.matmul(h2, w3) + b3
        #print(b3.name)
        return h3
    with tf.variable_scope("discriminator"):
        def sentence2state(sentence):
            #with tf.variable_scope("rnn_encoder"): 
            state = seq2seq(sentence)
            return (tf.reduce_sum(state,axis = 0))
        def state2sigmoid(state):
            return state2logit(state)

        with tf.variable_scope("twinsNN") as scope:
            true_state = sentence2state(true_ans)
            true_pos = state2sigmoid(true_state)
            scope.reuse_variables()
            fake_state = sentence2state(generated_ans)
            fake_pos = state2sigmoid(fake_state)
    return true_pos, fake_pos

def train():
    global BUCKET_ID
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
    
    # return a list of different bucket,but only one bucket it what we need
    #[seq_len * batch_size]
    #just to feed fake_ans
    fake_ans = build_generator(encoder_inputs,decoder_inputs,target_weights,
                                        bucket_id,seq_len)
    # 创建判别模型
    #true_ans, generated_ans:[max_len,batch_size,num_symbol]
    
    y_data, y_generated = build_discriminator(tf.one_hot(true_ans,num_symbols,
                                                            on_value=1.0,off_value=0.0,axis=-1,
                                                            dtype=tf.float32,name="onehot"), 
                                            fake_ans, 
                                            keep_prob ,seq_len)
    
    # 损失函数的设置
    d_loss_real = tf.reduce_mean(tf.scalar_mul(-1,y_data))
    d_loss_fake = tf.reduce_mean(y_generated)
    d_loss = d_loss_fake + d_loss_real
    g_loss =  tf.reduce_mean(tf.scalar_mul(-1,y_generated))

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "discriminator")
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "generator")

    gard = optimizer.compute_gradients(d_loss,var_list=d_params)
    # 两个模型的优化函数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    #clip discrim weights
    d_clip = [v.assign(tf.clip_by_value(v, CLIP_RANGE[0], CLIP_RANGE[1])) for v in d_params]

    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver(var_list = None,max_to_keep = 5)
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)
    sess.run(d_clip)
    #load previous variables
    if to_restore:
        print("reloading variables...")
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    if os.path.exists(output_path) == False:
            os.mkdir(output_path)

    
    get_data = dataset.DataProvider(pkl_path='./bdwm_data_token.pkl',
                            buckets_size=buckets,batch_size=batch_size)
    translator = Translator('./dict.txt')
    print("save ckpt")
    saver.save(sess,output_path,global_step=global_step)
    for i in range(sess.run(global_step), max_epoch):
        data_iterator = get_data.get_batch()
        for j in np.arange(CRITIC):
            print("epoch:%s, iter:%s" % (i, j))
            try:
                feed_dict, BUCKET_ID = data_iterator.next()
            except:
                pass
            sess.run(d_trainer,feed_dict=feed_dict)
            sess.run(d_clip)
        sess.run(g_trainer,feed_dict=feed_dict)
        try:
            feed_dict, BUCKET_ID = data_iterator.next()
        except:
            pass
        #get gen val for the true bucket
        gen_val = sess.run(fake_ans, feed_dict=feed_dict)
        translator.translate_and_print(seq2seq_onehot2label(gen_val))
        print("save ckpt")
        saver.save(sess,output_path,global_step=global_step)
        
if __name__ == '__main__':
    train()