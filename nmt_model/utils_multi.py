import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import py_utils as ut
import pdb

def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    failed_ids = []
    lines = open(file_path, 'r').readlines()
    for i, line in enumerate(lines):
        try:
#             sent = list(line)
            sent = line.strip().split(' ')
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
#                 sent = list(line)
                sent = ut.tokenize_code(line.strip(), mode='canonicalize')
                sent = ['<s>'] + sent + ['</s>']
                data.append(sent)
            elif source == 'src_code': #ocde2code
                 
                # data.append(sent)
                 sent = ut.tokenize_code(line.strip(), mode='canonicalize')
                 data.append(sent)
            elif source == 'src_nl':
                data.append(sent)
            else:
                print("WHAT")
                #data.append(sent)
            #data.append(sent)
        except:
            data.append('DUMMY')
            failed_ids.append(i)
    return data, failed_ids


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:

        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples_code = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        examples_nl = sorted(examples, key=lambda e: len(e[1]), reverse=True)

        examples_code = sorted(((i,e) for i,e in enumerate(examples)), key=lambda e: len(e[1][0]), reverse=True)
        examples_nl = sorted(((i,e) for i,e in enumerate(examples)), key=lambda e: len(e[1][1]), reverse=True)

       #code = [e[0] for e in examples_code]
       # nl = [e[1] for e in examples_nl]
        temp = [e[1] for e in examples_code]

        tgt_sents = [e[2] for e in temp]

        yield examples_code, examples_nl, tgt_sents

def batch_iter_beam(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:

        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples_code = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        examples_nl = sorted(examples, key=lambda e: len(e[1]), reverse=True)

        examples_code = sorted(((i,e) for i,e in enumerate(examples)), key=lambda e: len(e[1][0]), reverse=True)
        examples_nl = sorted(((i,e) for i,e in enumerate(examples)), key=lambda e: len(e[1][1]), reverse=True)

       #code = [e[0] for e in examples_code]
       # nl = [e[1] for e in examples_nl]


        yield examples_code, examples_nl

class LabelSmoothingLoss(nn.Module):
    """
    label smoothing

    Code adapted from OpenNMT-py
    """
    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)

        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss
