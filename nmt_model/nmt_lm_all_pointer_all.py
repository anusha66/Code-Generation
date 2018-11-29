# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py train-mcmc-raml --proposal-model=<file> --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

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
    --log-every=<int>                       log every [default: 50]
    --max-epoch=<int>                       max epoch [default: 20]
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
from torch.autograd import *
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
from utils import read_corpus, batch_iter, LabelSmoothingLoss
from vocab import Vocab, VocabEntry


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, input_feed=True, label_smoothing=0., lm=False):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.input_feed = input_feed
        self.lm=lm
        self.use_cuda = True

        self.src_embed = nn.Embedding(len(vocab.src), embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt['<pad>'])
        self.all_embed = nn.Embedding(len(vocab.all), embed_size, padding_idx=vocab.all['<pad>'])

        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        decoder_lstm_input = embed_size + hidden_size if self.input_feed else embed_size
        self.decoder_lstm = nn.LSTMCell(decoder_lstm_input, hidden_size)


        self.sentinel_vector = Variable(torch.zeros(hidden_size, 1), requires_grad=True)
        if self.use_cuda:
            self.sentinel_vector = self.sentinel_vector.cuda(3)
        # attention: dot product attention
        # project source encoding to decoder rnn's state space
        self.att_src_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
#         self.att_vec_linear = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(vocab.all), bias=False)
        #self.readout = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(hidden_size * 2, hidden_size)

        self.label_smoothing = label_smoothing
        if label_smoothing > 0.:
            self.label_smoothing_loss = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=len(vocab.tgt), padding_idx=vocab.tgt['<pad>'])

    @property
    def device(self) -> torch.device:
        return self.src_embed.weight.device

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        
        
        if self.lm:

             # (tgt_sent_len, batch_size)
            tgt_sents_var = self.vocab.all.to_input_tensor(tgt_sents, device=self.device)

            src_sents_var = self.vocab.all.to_input_tensor(src_sents, device=self.device)
            src_word_embeds = self.all_embed(src_sents_var)

            #print(tgt_sents_var.shape[0]) 
            att_vecs = []
            batch_size = len(tgt_sents)
            h_t, cell_t = torch.zeros(batch_size, self.hidden_size, device=self.device), torch.zeros(batch_size, self.hidden_size, device=self.device)
            
            cumulate_matrix_tgt = torch.zeros((tgt_sents_var.size(0), batch_size, len(self.vocab.all)))
            if self.use_cuda:
               cumulate_matrix_tgt = cumulate_matrix_tgt.cuda(3)

            cumulate_matrix_tgt.scatter_(2, tgt_sents_var.unsqueeze(2), 1.0)

            prefix_matrix_src = torch.zeros((src_sents_var.size(0), batch_size, len(self.vocab.all)))
            
            if self.use_cuda:
               prefix_matrix_src = prefix_matrix_src.cuda(3)

            prefix_matrix_src.scatter_(2, src_sents_var.unsqueeze(2), 1.0)
            
            probs = [] 
            hiddens = []
            ptr_scores = []
            
            tgt_word_embeds = self.all_embed(tgt_sents_var[:-1])

            src_len = len(src_word_embeds.split(split_size=1))
            for step, y_tm1_embed in enumerate(src_word_embeds.split(split_size=1)):
                y_tm1_embed = y_tm1_embed.squeeze(0)
                x = y_tm1_embed
                (h_t, cell_t) = self.decoder_lstm(x, (h_t, cell_t))
                hiddens.append(h_t)

            ###

            # start from y_0=`<s>`, iterate until y_{T-1}
            for step, y_tm1_embed in enumerate(tgt_word_embeds.split(split_size=1)):
                
                y_tm1_embed = y_tm1_embed.squeeze(0)
                
                x = y_tm1_embed
                (h_t, cell_t) = self.decoder_lstm(x, (h_t, cell_t))
                query = torch.tanh(self.att_vec_linear(h_t))
                hiddens.append(h_t)
                #z_s = []

                #for j in range(src_len):

                #	z_s.append(torch.sum(hiddens[j] * query, 1).view(-1))
           
                z = []
                #print(step)   
                for j in range(step + 1 + src_len):
                   #if step == 49:
                    #print(j, " j")

                   z.append(torch.sum(hiddens[j] * query, 1).view(-1))


                z.append(torch.mm(query, self.sentinel_vector).view(-1))
                #print(query.cuda(self.sentinel_vector.get_device()).get_device(), " query")
                #print(self.sentinel_vector.get_device(), " sent")
                z = torch.stack(z)

                a = F.softmax(z.transpose(0, 1), dim=-1).transpose(0, 1) #dim
                prefix_matrix_tgt = cumulate_matrix_tgt[:step + 1]
                
                if a[:src_len].shape[0] > 200:
                  for k in range(a[:src_len].shape[0]):
                     if k == 0:
                        p_ptr_src = Variable(prefix_matrix_src[k]) * a[:src_len][k].unsqueeze(1).expand_as(prefix_matrix_src[k])
                     else:
                        p_ptr_src = p_ptr_src + Variable(prefix_matrix_src[k]) * a[:src_len][k].unsqueeze(1).expand_as(prefix_matrix_src[k])

                else:
                  p_ptr_src = torch.sum(Variable(prefix_matrix_src) * a[:src_len].unsqueeze(2).expand_as(prefix_matrix_src), 0).squeeze(0)
                  
                if a[src_len:-1].shape[0] > 200:
                  for k in range(a[src_len:-1].shape[0]):
                     if k == 0:
                        p_ptr_tgt = Variable(prefix_matrix_tgt[k]) * a[src_len:-1][k].unsqueeze(1).expand_as(prefix_matrix_tgt[k])
                     else:
                        p_ptr_tgt = p_ptr_tgt + Variable(prefix_matrix_tgt[k]) * a[src_len:-1][k].unsqueeze(1).expand_as(prefix_matrix_tgt[k])

                else:
                  p_ptr_tgt = torch.sum(Variable(prefix_matrix_tgt) * a[src_len:-1].unsqueeze(2).expand_as(prefix_matrix_tgt), 0).squeeze(0)

                p_vocab = F.softmax(self.readout(h_t), dim=-1) #dim
                p = p_ptr_src + p_ptr_tgt + p_vocab * a[-1].unsqueeze(1).expand_as(p_vocab)
                
                probs.append(p)
                ptr_scores.append(p_ptr_src + p_ptr_tgt + a[-1].unsqueeze(1))
            # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
            if self.label_smoothing:
                # (tgt_sent_len - 1, batch_size)
                tgt_gold_words_log_prob = self.label_smoothing_loss(tgt_words_log_prob.view(-1, tgt_words_log_prob.size(-1)),
                                                                tgt_sents_var[1:].view(-1)).view(-1, len(tgt_sents))
            
            else:
                # (tgt_sent_len, batch_size)
                tgt_words_mask = (tgt_sents_var != self.vocab.all['<pad>']).float()

                # (tgt_sent_len - 1, batch_size)
                tgt_gold_words_log_prob = torch.gather(torch.log(torch.stack(probs)), index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]            
                tgt_gold_ptr_log_prob = torch.gather(torch.log(torch.stack(ptr_scores)), index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]
            
            scores = tgt_gold_words_log_prob.sum(dim=0)
            #temp = torch.log(torch.stack(probs))
            #temp2 = temp * tgt_words_mask[1:].unsqueeze(2)
            #t = temp2.view(-1,  len(self.vocab.tgt))
            ptr_loss = tgt_gold_ptr_log_prob.sum(dim=0)
            #torch.log(torch.cat(ptr_scores).view(-1, self.vocab_size))

            #t = torch.log(torch.cat(probs).view(-1, len(self.vocab.tgt)))
            return scores, ptr_loss
        
        else:
            # (src_sent_len, batch_size)
            src_sents_var = self.vocab.src.to_input_tensor(src_sents, device=self.device)
            # (tgt_sent_len, batch_size)
            tgt_sents_var = self.vocab.tgt.to_input_tensor(tgt_sents, device=self.device)
            src_sents_len = [len(s) for s in src_sents]

            src_encodings, decoder_init_vec = self.encode(src_sents_var, src_sents_len)

            src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

            # (tgt_sent_len - 1, batch_size, hidden_size)
            att_vecs = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])

            # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
            tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)

            if self.label_smoothing:
                # (tgt_sent_len - 1, batch_size)
                tgt_gold_words_log_prob = self.label_smoothing_loss(tgt_words_log_prob.view(-1, tgt_words_log_prob.size(-1)),
                                                                tgt_sents_var[1:].view(-1)).view(-1, len(tgt_sents))
            else:
                # (tgt_sent_len, batch_size)
                tgt_words_mask = (tgt_sents_var != self.vocab.tgt['<pad>']).float()

                # (tgt_sent_len - 1, batch_size)
                tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]

            # (batch_size)
            scores = tgt_gold_words_log_prob.sum(dim=0)

            return scores


    def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float,
                                     device=self.device)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1
        return src_sent_masks

    def encode(self, src_sents_var: torch.Tensor, src_sent_lens: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # (src_sent_len, batch_size, embed_size)
        src_word_embeds = self.src_embed(src_sents_var)
        packed_src_embed = pack_padded_sequence(src_word_embeds, src_sent_lens)

        # src_encodings: (src_sent_len, batch_size, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)

        # (batch_size, src_sent_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], dim=1))
        dec_init_state = torch.tanh(dec_init_cell)

        return src_encodings, (dec_init_state, dec_init_cell)

    def decode(self, src_encodings: torch.Tensor, src_sent_masks: torch.Tensor,
               decoder_init_vec: Tuple[torch.Tensor, torch.Tensor], tgt_sents_var: torch.Tensor) -> torch.Tensor:
        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = self.att_src_linear(src_encodings)

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # (tgt_sent_len, batch_size, embed_size)
        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding
        tgt_word_embeds = self.tgt_embed(tgt_sents_var)

        h_tm1 = decoder_init_vec

        att_ves = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)

                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks)

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        return att_ves

    def step(self, x: torch.Tensor,
             h_tm1: Tuple[torch.Tensor, torch.Tensor],
             src_encodings: torch.Tensor, src_encoding_att_linear: torch.Tensor, src_sent_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encoding_att_linear, src_sent_masks)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        return (cell_t, h_t), att_t, alpha_t

    def dot_prod_attention(self, h_t: torch.Tensor, src_encoding: torch.Tensor, src_encoding_att_linear: torch.Tensor,
                           mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

        if mask is not None:
            att_weight.data.masked_fill_(mask.byte(), -float('inf'))

        softmaxed_att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, softmaxed_att_weight

    def beam_search(self, src_sent: List[str], tgt_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        src_sents_var = self.vocab.all.to_input_tensor([src_sent], self.device)
        src_word_embeds = self.all_embed(src_sents_var)

        h_t, cell_t = torch.zeros(1, self.hidden_size, device=self.device), torch.zeros(1, self.hidden_size, device=self.device)
        

        eos_id = self.vocab.all['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []
        hiddens = []
        t = 0

        src_len = len(src_word_embeds.split(split_size=1))

        cumulate_matrix_tgt = torch.zeros((1, 1, len(self.vocab.all)))
        if self.use_cuda:
               cumulate_matrix_tgt = cumulate_matrix_tgt.cuda(3)

        prefix_matrix_src = torch.zeros((src_sents_var.size(0), 1, len(self.vocab.all)))

        if self.use_cuda:
               prefix_matrix_src = prefix_matrix_src.cuda(3)

        prefix_matrix_src.scatter_(2, src_sents_var.unsqueeze(2), 1.0)

        for step, y_tm1_embed in enumerate(src_word_embeds.split(split_size=1)):
                y_tm1_embed = y_tm1_embed.squeeze(0)
                x = y_tm1_embed
                (h_t, cell_t) = self.decoder_lstm(x, (h_t, cell_t))
                hiddens.append(h_t)
        prefix_matrix_tgt_list = [] 
        
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            y_tm1 = torch.tensor([self.vocab.all[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1_embed = self.all_embed(y_tm1)
            x = y_tm1_embed

            (h_t, cell_t) = self.decoder_lstm(x, (h_t, cell_t))
            query = torch.tanh(self.att_vec_linear(h_t))
            hiddens.append(h_t)
            z = []
            step = t-1
            for j in range(step + 1 + src_len):
                   z.append(torch.sum(hiddens[j] * query, 1).view(-1))
            z.append(torch.mm(query, self.sentinel_vector).view(-1))

            z = torch.stack(z)
            a = F.softmax(z.transpose(0, 1), dim=-1).transpose(0, 1)

            #if t ==2 :
            #   pdb.set_trace()

            prefix_matrix_tgt_list = []
            for k in range(len(hypotheses[0])):
                   y_tm1 = torch.tensor([self.vocab.all[hyp[k]] for hyp in hypotheses], dtype=torch.long, device=self.device)
                   temp = y_tm1.expand_as(torch.zeros(1, cumulate_matrix_tgt.shape[1]))
                   temp2 = cumulate_matrix_tgt.clone()
                   temp2.scatter_(2, temp.unsqueeze(2), 1.0)

                   prefix_matrix_tgt_list.append(temp2)
            '''
            temp = y_tm1.expand_as(torch.zeros(1, cumulate_matrix_tgt.shape[1])) 
            cumulate_matrix_tgt.scatter_(2, temp.unsqueeze(2), 1.0)
            prefix_matrix_tgt_list_new = []

            for ea in prefix_matrix_tgt_list:
              if cumulate_matrix_tgt.shape[1] >= ea.shape[1]:
                prefix_matrix_tgt_list_new.append(ea.expand_as(cumulate_matrix_tgt))
              else:
                prefix_matrix_tgt_list_new.append(ea[:, :cumulate_matrix_tgt.shape[1], :])

            prefix_matrix_tgt_list_new.append(cumulate_matrix_tgt)
            prefix_matrix_tgt_list = prefix_matrix_tgt_list_new
            prefix_matrix_tgt = torch.stack(prefix_matrix_tgt_list_new)
            '''

            prefix_matrix_tgt = torch.stack(prefix_matrix_tgt_list)
            prefix_matrix_tgt = prefix_matrix_tgt.squeeze(1)

            if a[:src_len].shape[0] > 200:
                  for k in range(a[:src_len].shape[0]):
                     if k == 0:
                        p_ptr_src = Variable(prefix_matrix_src[k]) * a[:src_len][k].unsqueeze(1).expand_as(prefix_matrix_src[k])
                     else:
                        p_ptr_src = p_ptr_src + Variable(prefix_matrix_src[k]) * a[:src_len][k].unsqueeze(1).expand_as(prefix_matrix_src[k])

            else:
                  p_ptr_src = torch.sum(Variable(prefix_matrix_src) * a[:src_len].unsqueeze(2).expand_as(prefix_matrix_src), 0).squeeze(0)
            if a[src_len:-1].shape[0] > 200:
                  for k in range(a[src_len:-1].shape[0]):
                     if k == 0:
                        p_ptr_tgt = Variable(prefix_matrix_tgt[k]) * a[src_len:-1][k].unsqueeze(1).expand_as(prefix_matrix_tgt[k])
                     else:
                        p_ptr_tgt = p_ptr_tgt + Variable(prefix_matrix_tgt[k]) * a[src_len:-1][k].unsqueeze(1).expand_as(prefix_matrix_tgt[k])

            else:
                  p_ptr_tgt = torch.sum(Variable(prefix_matrix_tgt) * a[src_len:-1].unsqueeze(2).expand_as(prefix_matrix_tgt), 0).squeeze(0)

            p_vocab = F.softmax(self.readout(h_t), dim=-1) #dim

                        
            p = p_ptr_src + p_ptr_tgt + p_vocab * a[-1].unsqueeze(1).expand_as(p_vocab)
            log_p_t = torch.log(p) 

#             else:
#                 if self.input_feed:
#                     x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
#                 else:
#                     x = y_tm1_embed

#                 (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1,
#                                                       exp_src_encodings, exp_src_encodings_att_linear, src_sent_masks=None)

            # log probabilities over target words
            #log_p_t = F.log_softmax(self.readout(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.all)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.all)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.all.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    if len(new_hyp_sent[1:-1])!=0:
                        completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score/len(new_hyp_sent[1:-1])))
                    else:
                        completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
#                     completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
#                                                            score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break
            #print(t, len(completed_hypotheses))
            
            cumulate_matrix_tgt = torch.zeros((1, beam_size - len(completed_hypotheses), len(self.vocab.all)))
            if self.use_cuda:
                  cumulate_matrix_tgt = cumulate_matrix_tgt.cuda(3)

            
            prefix_matrix_src = torch.zeros((src_sents_var.size(0), beam_size - len(completed_hypotheses), len(self.vocab.all)))

            temp_new = src_sents_var.unsqueeze(2).expand_as(torch.zeros(src_sents_var.shape[0], beam_size - len(completed_hypotheses), 1))

            if self.use_cuda:
               prefix_matrix_src = prefix_matrix_src.cuda(3)
            prefix_matrix_src.scatter_(2, temp_new, 1.0)

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_t, cell_t = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            hidden_new = []
            for each in hiddens:

                hidden_new.append(each[live_hyp_ids])
            hiddens = hidden_new

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

            
        if len(completed_hypotheses) == 0:
            if len((hypotheses[0][1:])) == 0:
                completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
            else:
                completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()/len(hypotheses[0][1:])))
#             completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
#                                                    score=hyp_scores[0].item()))

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
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss, loss_ptr = model(src_sents, tgt_sents)

            cum_loss += -1*loss.sum().item() + -1*loss_ptr.sum().item()
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
    
    train_data_src, failed_train_src_ids = read_corpus(args['--train-src'], source='src')
    train_data_tgt, failed_train_tgt_ids = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src, failed_dev_src_ids = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt, failed_dev_tgt_ids = read_corpus(args['--dev-tgt'], source='tgt')
    
    total_failed_ids = set(failed_train_src_ids).union(failed_train_tgt_ids)
    train_data_src = [train_data_src[i] for i in range(len(train_data_src)) if i not in total_failed_ids]
    train_data_tgt = [train_data_tgt[i] for i in range(len(train_data_tgt)) if i not in total_failed_ids]

    total_failed_ids = set(failed_dev_src_ids).union(failed_dev_tgt_ids)
    dev_data_src = [dev_data_src[i] for i in range(len(dev_data_src)) if i not in total_failed_ids]
    dev_data_tgt = [dev_data_tgt[i] for i in range(len(dev_data_tgt)) if i not in total_failed_ids]

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                input_feed=False,
                label_smoothing=float(args['--label-smoothing']),
                vocab=vocab,
               lm=True)
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:3" if args['--cuda'] else "cpu")
    #print('use device: %s' % device, file=sys.stderr)

    #if torch.cuda.device_count() > 1:
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # model = nn.DataParallel(model)

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

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            # (batch_size)
            example_losses, ptr_loss = model(src_sents, tgt_sents)
            batch_loss = -1*example_losses.sum() + -1*ptr_loss.sum()

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

def beam_search(model: NMT, test_data_src: List[List[str]], test_data_tgt: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training
    model.eval()

    test_data = list(zip(test_data_src, test_data_tgt))
    ii =0 
    hypotheses = []
    with torch.no_grad():
         for src_sent, tgt_sent in test_data:
        #for src_sent, tgt_sent in tqdm(test_data_src, test_data_tgt, desc='Decoding', file=sys.stdout):
            print(ii)
            example_hyps = model.beam_search(src_sent, tgt_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)
            ii = ii + 1
    if was_training: model.train(was_training)

    return hypotheses


def decode(args: Dict[str, str]):
    
    test_data_src, failed_ids_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    
    test_data_tgt, failed_ids_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')
    
    total_failed_ids = set(failed_ids_src).union(failed_ids_tgt)
    test_data_src = [test_data_src[i] for i in range(len(test_data_src)) if i not in total_failed_ids]
    test_data_tgt = [test_data_tgt[i] for i in range(len(test_data_tgt)) if i not in total_failed_ids]
   

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    #if args['--cuda']:
    #    model = model.to(torch.device("cuda:0"))

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.device_count() > 1:
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # model = nn.DataParallel(model)

    model = model.to(device)
    hypotheses = beam_search(model, test_data_src, test_data_tgt,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
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
    else:
        raise RuntimeError(f'invalid run mode')


if __name__ == '__main__':
    main()