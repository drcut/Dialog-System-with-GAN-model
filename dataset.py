import numpy as np
import pickle

_PAD = 0


class DataProvider(object):
    def __init__(self, pkl_path, buckets_size, batch_size):
        with open(pkl_path, 'rb') as fin:
            self.data = pickle.load(fin)

        self.buckets_size = buckets_size
        self.batch_size = batch_size
        self.all_qa = [(tid, pid) for tid in self.data for pid in self.data[tid]]
        self.buckets = [[] for i in range(len(self.buckets_size))]

    def get_batch(self):
        np.random.shuffle(self.all_qa)

        for tid, pid in self.all_qa:
            bucket_id = self.put_into_bucket(tid, pid)
            if bucket_id != -1:
                self.buckets[bucket_id].append((tid, pid))
                if len(self.buckets[bucket_id]) == self.batch_size:
                    yield self.build_feed_dict(bucket_id)
                    self.buckets[bucket_id] = []    # empty the bucket

    def build_feed_dict(self, bucket_id):
        if len(self.buckets[bucket_id]) != self.batch_size:
            raise Exception('Instances in bucket {} not equal to batch_size'.format(bucket_id))

        q_len, a_len = self.buckets_size[bucket_id]
        q_batch = []
        a_batch = []
        for tid, pid in self.buckets[bucket_id]:
            pair = self.data[tid][pid]
            q_batch.append(np.pad(pair[0],
                                  (0, q_len - len(pair[0])),
                                   'constant',
                                  constant_values=_PAD))
            a_batch.append(np.pad(pair[1],
                                  (0, a_len - len(pair[1])),
                                   'constant',
                                  constant_values=_PAD))

        q_batch = np.transpose(q_batch)
        a_batch = np.transpose(a_batch)
        feed_dict = {}
        for i, q in enumerate(q_batch):
            feed_dict['encoder{}'.format(i)] = q.astype('int32')
        for i, a in enumerate(a_batch):
            feed_dict['decoder{}'.format(i)] = a.astype('int32')
            feed_dict['weight{}'.format(i)] = np.ones(self.batch_size, 'float32')

        return feed_dict

    def put_into_bucket(self, tid, pid):
        pair = self.data[tid][pid]
        q_len, a_len = len(pair[0]), len(pair[1])
        for i, size in enumerate(self.buckets_size):
            if q_len <= size[0] and a_len <= size[1]:
                return i
        return -1
