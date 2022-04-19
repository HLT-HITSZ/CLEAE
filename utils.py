import hashlib
import numpy as np
import torch
from collections import defaultdict
from sklearn import metrics
import pickle

class Stack(object):
    def __init__(self, axis = 0, dtype = None):
        self._axis = axis
        self._dtype = dtype
    
    def __call__(self, data):
        data = np.stack(data, axis = self._axis).astype(self._dtype) if self._dtype else np.stack(data, axis = self._axis)
        return data

class Pad(object):
    def __init__(self,
                 pad_val = 0,
                 axis = 0,
                 ret_length = None,
                 dtype = None,
                 pad_right = True):
        self._pad_val = pad_val
        self._axis = axis
        self._ret_length = ret_length
        self._dtype = dtype
        self._pad_right = pad_right
    
    def __call__(self, data):
        arrs = [np.asarray(ele) for ele in data]
        original_length = [ele.shape[self._axis] for ele in arrs]
        max_size = max(original_length)
        ret_shape = list(arrs[0].shape)
        ret_shape[self._axis] = max_size
        ret_shape = (len(arrs), ) + tuple(ret_shape)
        ret = np.full(
            shape = ret_shape,
            fill_value = self._pad_val,
            dtype = arrs[0].dtype if self._dtype is None else self._dtype)
        for i, arr in enumerate(arrs):
            if arr.shape[self._axis] == max_size:
                ret[i] = arr
            else:
                slices = [slice(None) for _ in range(arr.ndim)]
                if self._pad_right:
                    slices[self._axis] = slice(0, arr.shape[self._axis])
                else:
                    slices[self._axis] = slice(max_size - arr.shape[self._axis], max_size)
                
                if slices[self._axis].start != slices[self._axis].stop:
                    slices = [slice(i, i + 1)] + slices
                    ret[tuple(slices)] = arr
        if self._ret_length:
            return ret, np.asarray(
                original_length,
                dtype="int32") if self._ret_length == True else np.asarray(
                    original_length, self._ret_length)
        else:
            return ret

class Tuple(object):
    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                i, str(type(ele_fn)))

    def __call__(self, data):
        assert len(data[0]) == len(self._fn), \
            'The number of attributes in each data sample should contain' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            result = ele_fn([ele[i] for ele in data])
            if isinstance(result, (tuple, list)):
                ret.extend(result)
            else:
                ret.append(result)
        return tuple(ret)


class ChunkEvaluator(object):
    def __init__(self, label_list, type="micro",no_relation = None):
        self.id2label_dict = dict(enumerate(label_list))
        self.label_list = label_list
        self.type = type
        self.preds = []
        self.labels = []
        self.no_relation = no_relation
    
    def compute(self):
        
        labels = self.labels
        preds = self.preds
        golden_cnt = 0
        guess_cnt = 0
        correct_cnt = 0
        for gold,guess in zip(labels,preds):
            if gold == self.no_relation and guess == self.no_relation:
                continue
            elif gold == self.no_relation and guess != self.no_relation:
                guess_cnt += 1
            elif gold != self.no_relation and guess == self.no_relation:
                golden_cnt += 1
            elif gold != self.no_relation and guess != self.no_relation:
                guess_cnt += 1
                golden_cnt += 1
                if gold == guess:
                    correct_cnt += 1
        
        if guess_cnt > 0:
            precision = correct_cnt/guess_cnt
        else:
            precision = 0
        if golden_cnt > 0:
            recall = correct_cnt/golden_cnt
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision+recall)
        else:
            f1 = 0.0
        acc = 0
        matrix = ""


        return acc,f1,precision,recall,matrix
    
    def update(self, preds, labels):
        self.preds = self.preds + preds.tolist()
        self.labels = self.labels + labels.tolist()
    
    def reset(self):
        self.preds = []
        self.labels = []


def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w") as outfile:
        [outfile.write(d + "\n") for d in data]


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab

def read_from_pickle(filename):
    with open(filename,"rb") as fin:
        return pickle.load(fin)

def to_device(inputs,device):

    return [torch.tensor(x).to(device, dtype = torch.int64) for x in inputs]