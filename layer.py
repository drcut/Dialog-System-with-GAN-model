#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import time
import numpy as np
from six.moves import xrange
import random
import warnings
from gensim.models import KeyedVectors
set_keep = globals()
set_keep['_layers_name_list'] =[]
set_keep['name_reuse'] = False

try:  # For TF12 and later
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
except:  # For TF11 and before
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.VARIABLES

def print_all_variables(train_only=False):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)
    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)
    """
    print("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars

def get_layers_with_name(network=None, name="", printable=False):
    """Get layer list in a network by a given name scope.
    >>> layers = tl.layers.get_layers_with_name(network, "CNN", True)
    """
    assert network is not None
    print("  [*] geting layers with %s" % name)

    layers = []
    i = 0
    for layer in network.all_layers:
        # print(type(layer.name))
        if name in layer.name:
            layers.append(layer)
            if printable:
                # print(layer.name)
                print("  got {:3}: {:15}   {}".format(i, layer.name, str(layer.get_shape())))
                i = i + 1
    return layers

def initialize_global_variables(sess=None):
    """Excute ``sess.run(tf.global_variables_initializer())`` for TF12+ or
    sess.run(tf.initialize_all_variables()) for TF11.

    Parameters
    ----------
    sess : a Session
    """
    assert sess is not None
    try:    # TF12
        sess.run(tf.global_variables_initializer())
    except: # TF11
        sess.run(tf.initialize_all_variables())

## Basic layer
class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.
    """
    def __init__(
        self,
        inputs = None,
        name ='layer'
    ):
        self.inputs = inputs
        if (name in set_keep['_layers_name_list']) and name_reuse == False:
            raise Exception("Layer '%s' already exists, please choice other 'name' or reuse this layer\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)" % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)
    def print_params(self, details=True):
        ''' Print all info of parameters in the network'''
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    print("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                except Exception as e:
                    print(str(e))
                    raise Exception("Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")
            else:
                print("  param {:3}: {:15}    {}".format(i, str(p.get_shape()), p.name))
        print("  num of params: %d" % self.count_params())
    def print_layers(self):
        ''' Print all info of layers in the network '''
        for i, p in enumerate(self.all_layers):
            print("  layer %d: %s" % (i, str(p)))
    def count_params(self):
        ''' Return the number of parameters in the network '''
        n_params = 0
        for i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params
    def __str__(self):
        print("\nIt is a Layer class")
        self.print_params(False)
        self.print_layers()
        return "  Last layer is: %s" % self.__class__.__name__
class Seq2seqWrapper(Layer):
  """Sequence-to-sequence model with attention and for multiple buckets.
    Parameters
    ----------
    source_vocab_size : size of the source vocabulary.
    target_vocab_size : size of the target vocabulary.
    buckets : a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
    size : number of units in each layer of the model.
    num_layers : number of layers in the model.
    max_gradient_norm : gradients will be clipped to maximally this norm.
    batch_size : the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
    learning_rate : learning rate to start with.
    learning_rate_decay_factor : decay learning rate by this much when needed.
    use_lstm : if true, we use LSTM cells instead of GRU cells.
    num_samples : number of samples for sampled softmax.
    forward_only : if set, we do not construct the backward pass in the model.
    name : a string or None
        An optional name to attach to this layer.
  """
  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               vec_file,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               name='wrapper'):
    Layer.__init__(self)#, name=name)

    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.size = size
    # =========== Load Vector File ======
    self.vec_model = KeyedVectors.load_word2vec_format(vec_file, binary=True)

    # =========== Fake output Layer for compute cost ======
    # If we use sampled softmax, we need an output projection.
    with tf.variable_scope(name) as vs:
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
          w = tf.get_variable("proj_w", [size, self.target_vocab_size])
          w_t = tf.transpose(w)
          b = tf.get_variable("proj_b", [self.target_vocab_size])
          output_projection = (w, b)

          def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                    self.target_vocab_size)
            #return tf.nn.sampled_softmax_loss(w_t, b, labels, num_samples,
              #      self.target_vocab_size)
          softmax_loss_function = sampled_loss

        # ============ Seq Encode Layer =============
        # Create the internal multi-layer cell for our RNN.
        try: # TF1.0
          single_cell = tf.contrib.rnn.GRUCell(size)
        except:
          single_cell = tf.nn.rnn_cell.GRUCell(size)

        if use_lstm:
          try: # TF1.0
            single_cell = tf.contrib.rnn.BasicLSTMCell(size)
          except:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)

        cell = single_cell
        if num_layers > 1:
          try: # TF1.0
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)
          except:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          #loop_function:
          #If not None, this function will be applied to i-th output in order to generate i+1-th input,
          #and decoder_inputs will be ignored
          if(do_decode==True):
            #loop_function = lambda prev,i: prev
            loop_function=None
          else:
            loop_function = None
          return tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(
            encoder_inputs, decoder_inputs, cell,
            loop_function=loop_function, dtype=tf.float32, scope=None)
          '''
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              encoder_inputs, decoder_inputs, cell,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode)
        '''
        #=============================================================
        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        #each step for an loop
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,size],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,size],
                                                    name="decoder{0}".format(i)))
          #[decoder_size*batch_size]
          self.target_weights.append(tf.placeholder(tf.float32, shape=[batch_size],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]
        self.targets = targets  # DH add for debug
        # Training outputs and losses.
        if forward_only:
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
              softmax_loss_function=lambda x,y:tf.losses.mean_squared_error(x, y))
              #softmax_loss_function=softmax_loss_function)
          # If we use output projection, we need to project outputs for decoding.
          if output_projection is not None:
            for b in xrange(len(buckets)):
              self.outputs[b] = [
                  tf.matmul(output, output_projection[0]) + output_projection[1]
                  for output in self.outputs[b]
              ]
        else:
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets,
              lambda x, y: seq2seq_f(x, y, False),
              softmax_loss_function=lambda x,y:tf.losses.mean_squared_error(x, y))
              #softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
          self.gradient_norms = []
          self.updates = []
          opt = tf.train.GradientDescentOptimizer(self.learning_rate)
          for b in xrange(len(buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))
        # if save into npz
        self.all_params = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Parameters
    ----------
    session : tensorflow session to use.
    encoder_inputs : list of numpy int vectors to feed as encoder inputs.
    decoder_inputs : list of numpy int vectors to feed as decoder inputs.
    target_weights : list of numpy float vectors to feed as target weights.
    bucket_id : which bucket of the model to use.
    forward_only : whether to do the backward step or only forward.

    Returns
    --------
    A triple consisting of gradient norm (or None if we did not do backward),
    average perplexity, and the outputs.

    Raises
    --------
    ValueError : if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
    # print('in model.step()')
    # print('a',bucket_id, encoder_size, decoder_size)

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
        #[encoder_size*batch_size]
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      #[decoder_size*batch_size]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = decoder_inputs[decoder_size-1]#should be padding,and I hope it is

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def id2vec(self,id):
    '''should return the word embedding(shape:a list of length [size])
    Parameters
    ------------
    id:the token id should be trans
    dictionary:like a map
    '''
    #print("id2vec")
    #print("id=")
    #print(id)
    try:
        ret_vec = self.vec_model[str(id)]
    except KeyError:
        ret_vec = [0.0]*self.size  # Later, this should be substituted as vec_model['3'], i.e. UNK_ID
    #print("res=")
    #print(ret_vec)
    return ret_vec

  def get_batch(self, data, bucket_id, PAD_ID=0, GO_ID=1, EOS_ID=2, UNK_ID=3):
    """Get a random batch of data from the specified bucket, prepare for step.
    Parameters
    ----------
    data : a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
    bucket_id : integer, which bucket to get the batch for.
    Returns
    -------
    The triple (encoder_inputs, decoder_inputs, target_weights) for
    the constructed batch that has the proper format to call step(...) later.
    """
    #print ("get batch")
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
        #encoder_input and decoder_input is a single sentence
      encoder_input, decoder_input = random.choice(data[bucket_id])
      #print ("encoder input")
      #print(encoder_input)
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      #encoder_inputs is a list(batch_size elements) of ask sentence(which is list)
      #reversed the order of input
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    #batch_encoder_inputs:shape[encoder_size*batch_size],each element is a size
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          list(#[encoder_inputs[batch_idx][length_idx]
                [ self.id2vec(encoder_inputs[batch_idx][length_idx])
                    for batch_idx in xrange(self.batch_size)]))
    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    #print ("finish encoder")
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          list(#[decoder_inputs[batch_idx][length_idx]
                            [ self.id2vec(decoder_inputs[batch_idx][length_idx]) #[2.0]*self.size stand for word_vec of word decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)]))
      #print ("finish decoder")
      batch_weight = self.batch_size*[1.0]
      for batch_idx in xrange(self.batch_size):
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
      #batch_weight:[decoder_size*batch_size]
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
