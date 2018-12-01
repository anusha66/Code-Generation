#!/usr/bin/env python
"""
Usage:
    vocab_multi.py --train-src-code=<file> --train-src-nl=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src-code=<file>    File of training source code sentences
    --train-src-nl=<file>      File of training source nl sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 10000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle
import torch
import pdb

from utils_multi import read_corpus, input_transpose


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
    def __init__(self, src_sents_code, src_sents_nl, tgt_sents, vocab_size, freq_cutoff):
        
        assert len(src_sents_code) == len(tgt_sents) == len(src_sents_nl)
        
        print('initialize source code vocabulary ..')
        self.src_code = VocabEntry.from_corpus(src_sents_code, vocab_size, freq_cutoff)

        print('initialize source nl vocabulary ..')
        self.src_nl = VocabEntry.from_corpus(src_sents_nl, vocab_size, freq_cutoff)
        
        print('initialize target vocabulary ..')
        self.tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

    def __repr__(self):
        return 'Vocab(source code %d words, source nl %d words, target %d words)' % (len(self.src_code), len(self.src_nl), len(self.tgt))


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source code sentences: %s' % args['--train-src-code'])
    print('read in source nl sentences: %s' % args['--train-src-nl'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents_code, src_f_ids_code = read_corpus(args['--train-src-code'], source='src_code')
    tgt_sents, tgt_f_ids = read_corpus(args['--train-tgt'], source='tgt')
    src_sents_nl, src_f_ids_nl = read_corpus(args['--train-src-nl'], source='src_nl')

    total_failed_ids = set(src_f_ids_code).union(set(tgt_f_ids)).union(set(src_f_ids_nl))

    src_sents_code = [src_sents_code[i] for i in range(len(src_sents_code)) if i not in total_failed_ids]
    src_sents_nl = [src_sents_nl[i] for i in range(len(src_sents_nl)) if i not in total_failed_ids]
    tgt_sents = [tgt_sents[i] for i in range(len(tgt_sents)) if i not in total_failed_ids]
    
    vocab = Vocab(src_sents_code, src_sents_nl, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))    
    print('generated vocabulary, source code %d words, source nl %d words, target %d words' % (len(vocab.src_code), len(vocab.src_nl), len(vocab.tgt)))

    pickle.dump(vocab, open(args['VOCAB_FILE'], 'wb'))
    print('vocabulary saved to vocab.bin')
