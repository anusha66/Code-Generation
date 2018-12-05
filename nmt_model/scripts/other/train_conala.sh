#!/bin/sh

vocab="data/conala/vocab.bin"
train_src="/home/anushap/Code-Generation/nmt_model/data/conala/conala-trainnodev.intent"
train_tgt="/home/anushap/Code-Generation/nmt_model/data/conala/conala-trainnodev.snippet"
# train_src="data/conala/valid.de-en.de"
# train_tgt="data/conala/valid.de-en.en"
dev_src="/home/anushap/Code-Generation/nmt_model/data/conala/conala-dev.intent"
dev_tgt="/home/anushap/Code-Generation/nmt_model/data/conala/conala-dev.snippet"
test_src="/home/anushap/Code-Generation/nmt_model/data/conala/conala-test.intent"
test_tgt="/home/anushap/Code-Generation/nmt_model/data/conala/conala-test.snippet"
#test_tgt_bleu="/home/anushap/Code-Generation/nmt_model/data/conala/code_test_bleu.txt"
#dev_tgt_bleu="/home/anushap/Code-Generation/nmt_model/data/conala/code_dev_bleu.txt"

work_dir="work_dir_conala"
'''
mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model.bin \
    --valid-niter 200 \
    --batch-size 16 \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 2>${work_dir}/err.log
'''

python nmt.py \
   decode \
   --cuda \
   --beam-size 10 \
   --max-decoding-time-step 50 \
   ${work_dir}/model.bin \
   ${test_src} \
   ${test_tgt} \
   ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
