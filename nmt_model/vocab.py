#!/usr/bin/env python
"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 2500]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle
import torch
import pdb

from utils import read_corpus, input_transpose, article2ids, abstract2ids


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_t = input_transpose(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)

        return sents_var
    
    def to_input_tensor_tgt_extend(self, sents: List[List[str]], article_oovs: List[List[str]], device: torch.device) -> torch.Tensor:

        sents_var = []
        for i, sent in enumerate(sents):
            ids = abstract2ids(sent, self, article_oovs[i])
            sents_var.append(ids)

        sents_var = input_transpose(sents_var, self['<pad>'])
        sents_var = torch.tensor(sents_var, dtype=torch.long, device=device)

        return sents_var

    def to_input_tensor_extend(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:

        sents_var = []
        oov_list = []

        for sent in sents:
            ids, oovs = article2ids(sent, self)
            sents_var.append(ids)
            oov_list.append(oovs)

        sents_var = input_transpose(sents_var, self['<pad>'])
        sents_var = torch.tensor(sents_var, dtype=torch.long, device=device)

        return sents_var, oov_list

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, src_sents, tgt_sents, vocab_size, freq_cutoff):
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        self.src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        self.tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents, src_f_ids = read_corpus(args['--train-src'], source='src')
    tgt_sents, tgt_f_ids = read_corpus(args['--train-tgt'], source='tgt')

   # vocab = Vocab.build(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))
    
    #src_sents, src_f_ids = read_corpus('/home/anushap/Code-Generation/data/nl_train.txt', source='src')
    #tgt_sents, tgt_f_ids  = read_corpus('/home/anushap/Code-Generation/data/code_train.txt', source='tgt')
#     pdb.set_trace()

    total_failed_ids = set(src_f_ids).union(tgt_f_ids)
    src_sents = [src_sents[i] for i in range(len(src_sents)) if i not in total_failed_ids]
    tgt_sents = [tgt_sents[i] for i in range(len(tgt_sents)) if i not in total_failed_ids]
    vocab = Vocab(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))    
    #vocab = Vocab(src_sents, tgt_sents, 7000, freq_cutoff=2)
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    pickle.dump(vocab, open(args['VOCAB_FILE'], 'wb'))
    print('vocabulary saved to vocab.bin')
    
#     args = docopt(__doc__)

#     print('read in source sentences: %s' % args['--train-src'])
#     print('read in target sentences: %s' % args['--train-tgt'])

#     src_sents = read_corpus(args['--train-src'], source='src')
#     tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

#     vocab = Vocab(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))
#     print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

#     pickle.dump(vocab, open(args['VOCAB_FILE'], 'wb'))
#     print('vocabulary saved to %s' % args['VOCAB_FILE'])
