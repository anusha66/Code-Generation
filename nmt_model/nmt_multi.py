# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src-code=<file> --train-src-nl=<file> --train-tgt=<file> --dev-src-code=<file> --dev-src-nl=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py train-mcmc-raml --proposal-model=<file> --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_CODE_FILE TEST_SOURCE_NL_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --proposal-model=<file>                 proposal model in MCMC-RAML
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --label-smoothing=<float>                  use label smoothing [default: 0.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import pdb
import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import raml_utils
from utils_multi import batch_iter_beam, read_corpus, batch_iter, LabelSmoothingLoss
from vocab_multi import Vocab, VocabEntry


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, input_feed=True, label_smoothing=0.):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.input_feed = True

        self.src_code_embed = nn.Embedding(len(vocab.src_code), embed_size, padding_idx=vocab.src_code['<pad>'])
        self.src_nl_embed = nn.Embedding(len(vocab.src_nl), embed_size, padding_idx=vocab.src_nl['<pad>'])

        self.tgt_embed = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt['<pad>'])

        self.code_encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.nl_encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)

        decoder_lstm_input = embed_size + hidden_size*2 + hidden_size*2 if self.input_feed else embed_size

        self.decoder_lstm = nn.LSTMCell(decoder_lstm_input, hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's state spacexxxx
        self.att_src_code_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_src_nl_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(hidden_size * 2 * 2+ hidden_size, hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(hidden_size * 2, hidden_size)

        self.label_smoothing = label_smoothing
        if label_smoothing > 0.:
            self.label_smoothing_loss = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=len(vocab.tgt),
                                                           padding_idx=vocab.tgt['<pad>'])

    @property
    def device(self) -> torch.device:
        return self.src_code_embed.weight.device

    def forward(self, src_code_sents: List[List[str]], src_nl_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        
        src_code_sents_str = [e[1][0] for e in src_code_sents]
        src_code_sents_order = [e[0] for e in src_code_sents]
        
        src_nl_sents_str = [e[1][1] for e in src_nl_sents]
        src_nl_sents_order = [e[0] for e in src_nl_sents]

        src_code_sents_var = self.vocab.src_code.to_input_tensor(src_code_sents_str, device=self.device)
        src_nl_sents_var = self.vocab.src_nl.to_input_tensor(src_nl_sents_str, device=self.device)

        tgt_sents_var = self.vocab.tgt.to_input_tensor(tgt_sents, device=self.device)
        
        src_code_sents_len = [len(s) for s in src_code_sents_str]
        src_nl_sents_len = [len(s) for s in src_nl_sents_str]

        src_code_encodings, decoder_init_vec_code = self.encode_code(src_code_sents_var, src_code_sents_len)
        src_nl_encodings, src_nl_sents_len, decoder_init_vec_nl = self.encode_nl(src_nl_sents_var, src_code_sents_order, src_nl_sents_order, src_nl_sents_len)
        
        src_code_sent_masks = self.get_attention_mask(src_code_encodings, src_code_sents_len)
        src_nl_sent_masks = self.get_attention_mask(src_nl_encodings, src_nl_sents_len)
        
        att_vecs = self.decode(src_code_encodings, src_code_sent_masks, src_nl_encodings, src_nl_sent_masks, decoder_init_vec_code, decoder_init_vec_nl, tgt_sents_var[:-1])

        tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)
            
        tgt_words_mask = (tgt_sents_var != self.vocab.tgt['<pad>']).float()

        tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]

        scores = tgt_gold_words_log_prob.sum(dim=0)

        return scores

    def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float,
                                     device=self.device)
        
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1
        
        return src_sent_masks

    def encode_code(self, src_code_sents_var: torch.Tensor, src_code_sent_lens: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        src_code_word_embeds = self.src_code_embed(src_code_sents_var)

        packed_src_code_embed = pack_padded_sequence(src_code_word_embeds, src_code_sent_lens)
        src_code_encodings, (last_state, last_cell) = self.code_encoder_lstm(packed_src_code_embed)
        src_code_encodings, _ = pad_packed_sequence(src_code_encodings)

        src_code_encodings = src_code_encodings.permute(1, 0, 2)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], dim=1))
        dec_init_state = torch.tanh(dec_init_cell)

        return src_code_encodings, (dec_init_state, dec_init_cell)

    def encode_nl(self, src_nl_sents_var: torch.Tensor, src_code_sents_order: List[int], src_nl_sents_order: List[int], src_nl_sent_lens: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        src_nl_word_embeds = self.src_nl_embed(src_nl_sents_var)
        
        packed_src_nl_embed = pack_padded_sequence(src_nl_word_embeds, src_nl_sent_lens)
        src_nl_encodings, (last_state, last_cell) = self.nl_encoder_lstm(packed_src_nl_embed)
        src_nl_encodings, _ = pad_packed_sequence(src_nl_encodings)
        
        src_nl_encodings = src_nl_encodings.permute(1, 0, 2)

        src_nl_encodings_orig = sorted(list(zip(src_nl_encodings, src_nl_sents_order)), key=lambda e: e[1])
        temp = []

        for each in range(len(src_nl_encodings_orig)):
                 temp.append(src_nl_encodings_orig[each][0])

        src_nl_encodings_new1 = [temp[e] for e in src_code_sents_order]

        src_nl_encodings_new2 = torch.stack(src_nl_encodings_new1)

        src_nl_sent_lens_new = [src_nl_sent_lens[e] for e in src_code_sents_order]

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], dim=1))
        dec_init_state = torch.tanh(dec_init_cell)

        return src_nl_encodings_new2, src_nl_sent_lens_new, (dec_init_state, dec_init_cell)

    def decode(self, src_code_encodings: torch.Tensor, src_code_sent_masks: torch.Tensor, src_nl_encodings: torch.Tensor, src_nl_sent_masks: torch.Tensor,
               decoder_init_vec_code: Tuple[torch.Tensor, torch.Tensor], decoder_init_vec_nl: Tuple[torch.Tensor, torch.Tensor], tgt_sents_var: torch.Tensor) -> torch.Tensor:
        
        src_code_encoding_att_linear = self.att_src_code_linear(src_code_encodings)
        src_nl_encoding_att_linear = self.att_src_nl_linear(src_nl_encodings)

        batch_size = src_code_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = (torch.zeros(batch_size, self.hidden_size*2, device=self.device), torch.zeros(batch_size, self.hidden_size*2, device=self.device))

        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding
        tgt_word_embeds = self.tgt_embed(tgt_sents_var)

        h_tm1 = ((decoder_init_vec_code[0] + decoder_init_vec_nl[0])/2, (decoder_init_vec_code[1] + decoder_init_vec_nl[1])/2)

        att_ves = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector

                x = torch.cat([y_tm1_embed, att_tm1[0], att_tm1[1]], dim=-1)
            else:
                x = y_tm1_embed

            (cell_t, h_t), att_t, ctx_t_code, ctx_t_nl = self.step(x, h_tm1, src_code_encodings, src_code_encoding_att_linear, src_nl_encodings, src_nl_encoding_att_linear, src_code_sent_masks, src_nl_sent_masks)

            att_tm1 = (ctx_t_code, ctx_t_nl)
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        att_ves = torch.stack(att_ves)

        return att_ves

    def step(self, x: torch.Tensor,
             h_tm1: Tuple[torch.Tensor, torch.Tensor],
             src_code_encodings: torch.Tensor, src_code_encoding_att_linear: torch.Tensor, src_nl_encodings: torch.Tensor, src_nl_encoding_att_linear: torch.Tensor, src_code_sent_masks: torch.Tensor, src_nl_sent_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        
        h_t, cell_t = self.decoder_lstm(x, h_tm1)
        
        ctx_t_code, alpha_t_code = self.dot_prod_attention(h_t, src_code_encodings, src_code_encoding_att_linear, src_code_sent_masks)
        ctx_t_nl, alpha_t_nl = self.dot_prod_attention(h_t, src_nl_encodings, src_nl_encoding_att_linear, src_nl_sent_masks)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t_code, ctx_t_nl], 1))) 
        att_t = self.dropout(att_t)
        
        return (cell_t, h_t), att_t, ctx_t_code, ctx_t_nl

    def dot_prod_attention(self, h_t: torch.Tensor, src_encoding: torch.Tensor, src_encoding_att_linear: torch.Tensor,
                           mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        
        if mask is not None:
            att_weight.data.masked_fill_(mask.byte(), -float('inf'))
        softmaxed_att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)
        
        return ctx_vec, softmaxed_att_weight

    def beam_search(self, src_code_sent: List[Tuple[List[int], Tuple[List[str],List[str]]]], src_nl_sent: List[Tuple[List[int], Tuple[List[str],List[str]]]], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        
        src_code_sent_str = src_code_sent[0][1][0]
        src_code_sent_order = src_code_sent[0][0]

        src_nl_sent_str = src_nl_sent[0][1][1]
        src_nl_sent_order = src_nl_sent[0][0]

        src_code_sents_var = self.vocab.src_code.to_input_tensor([src_code_sent_str], self.device)
        src_nl_sents_var = self.vocab.src_nl.to_input_tensor([src_nl_sent_str], self.device)

        src_code_encodings, dec_init_vec_code = self.encode_code(src_code_sents_var, [len(src_code_sents_var)])
        src_nl_encodings, src_nl_len, dec_init_vec_nl = self.encode_nl(src_nl_sents_var, [src_code_sent_order], [src_nl_sent_order], [len(src_nl_sents_var)])

        src_code_encodings_att_linear = self.att_src_code_linear(src_code_encodings)
        src_nl_encodings_att_linear = self.att_src_nl_linear(src_nl_encodings)

        h_tm1 = ((dec_init_vec_code[0] + dec_init_vec_nl[0])/2, (dec_init_vec_code[1] + dec_init_vec_nl[1])/2)

        att_tm1 = (torch.zeros(1, self.hidden_size*2, device=self.device), torch.zeros(1, self.hidden_size*2, device=self.device))

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []
        t = 0
        
        while len(completed_hypotheses) < beam_size  and t < max_decoding_time_step:
        
            t += 1
            hyp_num = len(hypotheses)

            exp_src_code_encodings = src_code_encodings.expand(hyp_num,
                                                     src_code_encodings.size(1),
                                                     src_code_encodings.size(2))
            exp_src_nl_encodings = src_nl_encodings.expand(hyp_num,
                                                     src_nl_encodings.size(1),
                                                     src_nl_encodings.size(2))

            exp_src_code_encodings_att_linear = src_code_encodings_att_linear.expand(hyp_num,
                                                                           src_code_encodings_att_linear.size(1),
                                                                           src_code_encodings_att_linear.size(2))

            exp_src_nl_encodings_att_linear = src_nl_encodings_att_linear.expand(hyp_num,
                                                                           src_nl_encodings_att_linear.size(1),
                                                                           src_nl_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1[0], att_tm1[1]], dim=-1)
            else:
                x = y_tm1_embed

            (cell_t, h_t), att_t, ctx_t_code, ctx_t_nl = self.step(x, h_tm1,
                                                      exp_src_code_encodings, exp_src_code_encodings_att_linear, exp_src_nl_encodings, exp_src_nl_encodings_att_linear, src_code_sent_masks=None, src_nl_sent_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.readout(att_t), dim=-1)
            
            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
    
                    if len(new_hyp_sent[1:-1])!=0:
                        completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score/len(new_hyp_sent[1:-1])))
                    else:
                        completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                   
                else:

                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = (ctx_t_code[live_hyp_ids], ctx_t_nl[live_hyp_ids])

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            
            if len((hypotheses[0][1:])) == 0:
               completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
            else:
               completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()/len(hypotheses[0][1:])))
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         input_feed=self.input_feed, label_smoothing=self.label_smoothing),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


def evaluate_ppl(model, dev_data, batch_size=32):
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.
    with torch.no_grad():
        for src_code_sents, src_nl_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_code_sents, src_nl_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    # compute BLEU
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict):
    
    train_data_src_code, failed_train_src_code_ids = read_corpus(args['--train-src-code'], source='src_code')
    train_data_src_nl, failed_train_src_nl_ids = read_corpus(args['--train-src-nl'], source='src_nl')
    train_data_tgt, failed_train_tgt_ids = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src_code, failed_dev_src_code_ids = read_corpus(args['--dev-src-code'], source='src_code')
    dev_data_src_nl, failed_dev_src_nl_ids = read_corpus(args['--dev-src-nl'], source='src_nl')
    dev_data_tgt, failed_dev_tgt_ids = read_corpus(args['--dev-tgt'], source='tgt') 

    total_failed_ids = set(failed_train_src_nl_ids).union(failed_train_tgt_ids).union(failed_train_src_code_ids)

    train_data_src_code = [train_data_src_code[i] for i in range(len(train_data_src_code)) if i not in total_failed_ids]
    train_data_src_nl = [train_data_src_nl[i] for i in range(len(train_data_src_nl)) if i not in total_failed_ids]
    train_data_tgt = [train_data_tgt[i] for i in range(len(train_data_tgt)) if i not in total_failed_ids]

    total_failed_ids = set(failed_dev_src_nl_ids).union(failed_dev_tgt_ids).union(failed_dev_src_code_ids)

    dev_data_src_code = [dev_data_src_code[i] for i in range(len(dev_data_src_code)) if i not in total_failed_ids]
    dev_data_src_nl = [dev_data_src_nl[i] for i in range(len(dev_data_src_nl)) if i not in total_failed_ids]
    dev_data_tgt = [dev_data_tgt[i] for i in range(len(dev_data_tgt)) if i not in total_failed_ids]

    train_data = list(zip(train_data_src_code, train_data_src_nl, train_data_tgt))
    dev_data = list(zip(dev_data_src_code, dev_data_src_nl,dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                input_feed=args['--input-feed'],
                label_smoothing=float(args['--label-smoothing']),
                vocab=vocab)
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:3" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        
        for src_code_sents, src_nl_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_code_sents)
            
            example_losses = -model(src_code_sents, src_nl_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=16)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                exit(0)


def beam_search(model: NMT, test_data_src_code: List[List[str]], test_data_src_nl: List[List[str]],  beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    
    was_training = model.training
    model.eval()

    hypotheses = []

    test_data = list(zip(test_data_src_code, test_data_src_nl))
 
    with torch.no_grad():
        for src_code_sent, src_nl_sent in batch_iter_beam(test_data,  batch_size=1, shuffle=False):
            example_hyps = model.beam_search(src_code_sent, src_nl_sent,  beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def decode(args: Dict[str, str]):
    
    test_data_src_code, failed_ids_src_code = read_corpus(args['TEST_SOURCE_CODE_FILE'], source='src_code')
    test_data_src_nl, failed_ids_src_nl = read_corpus(args['TEST_SOURCE_NL_FILE'], source='src_nl')
     
    test_data_tgt, failed_ids_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')
    
    total_failed_ids = set(failed_ids_src_code).union(failed_ids_tgt).union(failed_ids_src_nl)
    
    test_data_src_code = [test_data_src_code[i] for i in range(len(test_data_src_code)) if i not in total_failed_ids]

    test_data_src_nl = [test_data_src_nl[i] for i in range(len(test_data_src_nl)) if i not in total_failed_ids]

    test_data_tgt = [test_data_tgt[i] for i in range(len(test_data_tgt)) if i not in total_failed_ids]

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:3"))
    hypotheses = beam_search(model, test_data_src_code, test_data_src_nl,
                              beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_code_sent, src_nl_sent, hyps in zip(test_data_src_code, test_data_src_nl, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the RNG
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        print ('Training Start \n')
        train(args)
    elif args['decode']:
        decode(args)
    elif args['train-mcmc-raml']:
        train_mcmc_raml(args)
    else:
        raise RuntimeError(f'invalid run mode')


if __name__ == '__main__':
    main()
