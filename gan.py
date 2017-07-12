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
to_restore = False
max_len = buckets[-1][1]
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
    #print("patch")
    patch = tf.convert_to_tensor([[0.0]*num_symbols] * batch_size)
    #print("filled")
    #print(filled)
    #Tensor("mul:0", shape=(64, 20000), dtype=float32)
    #outputs[0].append([patch]*(max_len-buckets[0][1])) #a list of max_len's elements,
    #each with[batch_size * num_symbols]
    #print(outputs[0][70])
    #print("padding")
    #print(outputs[0].append([patch]*(max_len-buckets[0][1])))
    #print(tf.argmax(tf.convert_to_tensor(outputs[0],dtype = tf.float32),axis = 2))
    #print("after test")
    def f0(): 
        for _ in range(0,max_len-buckets[0][1]):
            outputs[0].append(patch)
        #fake_ans = tf.argmax(tf.convert_to_tensor(outputs[0],dtype = tf.float32),axis = 2)
        #return tf.argmax(tf.convert_to_tensor(outputs[0],dtype = tf.float32),axis = 2)
        return tf.convert_to_tensor(outputs[0],dtype = tf.float32)
    def f1(): 
        for _ in range(0,max_len-buckets[1][1]):
            outputs[1].append(patch)
        #fake_ans = tf.argmax(tf.convert_to_tensor(outputs[1],dtype = tf.float32),axis = 2)
        #return tf.argmax(tf.convert_to_tensor(outputs[1],dtype = tf.float32),axis = 2)
        return tf.convert_to_tensor(outputs[1],dtype = tf.float32)
    def f2(): 
        for _ in range(0,max_len-buckets[2][1]):
            outputs[2].append(patch)
        #fake_ans = tf.argmax(tf.convert_to_tensor(outputs[2],dtype = tf.float32),axis = 2)
        #return tf.argmax(tf.convert_to_tensor(outputs[2],dtype = tf.float32),axis = 2)
        return tf.convert_to_tensor(outputs[2],dtype = tf.float32)
    def f3(): 
        for _ in range(0,max_len-buckets[3][1]):
            outputs[3].append(patch)
        #fake_ans = tf.argmax(tf.convert_to_tensor(outputs[3],dtype = tf.float32),axis = 2)
        #return tf.argmax(tf.convert_to_tensor(outputs[3],dtype = tf.float32),axis = 2)
        return tf.convert_to_tensor(outputs[3],dtype = tf.float32)
    def f4(): 
        for _ in range(0,max_len-buckets[4][1]):
            outputs[4].append(patch)
        #fake_ans = tf.argmax(tf.convert_to_tensor(outputs[4],dtype = tf.float32),axis = 2)
        #return tf.argmax(tf.convert_to_tensor(outputs[4],dtype = tf.float32),axis = 2)
        return tf.convert_to_tensor(outputs[4],dtype = tf.float32)

    r = tf.case({tf.equal(bucket_id, 0): f0,
                tf.equal(bucket_id, 1): f1,
                tf.equal(bucket_id, 2): f2,
                tf.equal(bucket_id, 3): f3},
                default=f4, exclusive=True)

    print("r")
    print(r)
    return tf.reshape(r,[max_len,batch_size,num_symbols])

# discriminator (model 2)
def build_discriminator(true_ans, generated_ans, keep_prob ,seq_len):
    '''
    true_ans, generated_ans:[max_len,batch_size,num_symbol]
    '''
    #print("fake_ans")
    #print(generated_ans)
    with tf.variable_scope("discriminator"):
        def sentence2state(sentence):
            #sentence:[max_time, batch_size, num_decoder_symbols]
            #sequence_length:[batch_size]
            with tf.variable_scope("rnn_encoder"):
                cell = tf.contrib.rnn.BasicLSTMCell(state_size)
                if num_layers > 1:
                    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
                outputs, state = tf.nn.dynamic_rnn(cell, sentence, 
                    sequence_length=seq_len, initial_state=None,dtype=tf.float32,
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
    d_loss = - (tf.log(y_data) - tf.log(1 - y_generated))
    g_loss =  - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "discriminator")
    g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "generator")

    gard = optimizer.compute_gradients(d_loss,var_list=d_params)
    #print("gard ok")
    # 两个模型的优化函数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()

    # Create a saver.
    saver = tf.train.Saver(var_list = None,max_to_keep = 5)
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)
    #load previous variables
    if to_restore:
        print("reloading variables...")
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    if os.path.exists(output_path) == False:
            os.mkdir(output_path)

    steps = 5
    max_epoch = 5
    get_data = dataset.DataProvider(pkl_path='./bdwm_data_token.pkl',
                            buckets_size=buckets,batch_size=batch_size)
    translator = Translator('./dict.txt')

    for i in range(sess.run(global_step), max_epoch):
        data_iterator = get_data.get_batch()
        for j in np.arange(steps):
            print("epoch:%s, iter:%s" % (i, j))
            feed_dict, BUCKET_ID = data_iterator.next()
            sess.run(d_trainer,feed_dict=feed_dict)
        sess.run(g_trainer,feed_dict=feed_dict)
        feed_dict, BUCKET_ID = data_iterator.next()
        #get gen val for the true bucket
        #gen_val = sess.run([mul_generated_ans[BUCKET_ID]], feed_dict=feed_dict)
        translator.translate_and_print(seq2seq_onehot2label())

        '''
        file_object = open(os.path.join(res_path,"epoch:%s.txt" % (i)), 'w')
        print("true data possible")
        print(true_p)
        print("fake data possible")
        print(fake_p)
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "gan_model"), global_step=global_step)
        '''
        
if __name__ == '__main__':
    train()