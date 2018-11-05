#!/bin/sh

vocab="data/2code/vocab.bin"
train_src_code="/home/anushap/Code-Generation/nmt_model/data/2code/src_code_train.txt"
train_src_nl="/home/anushap/Code-Generation/nmt_model/data/2code/nl_train.txt"
train_tgt="/home/anushap/Code-Generation/nmt_model/data/2code/code_train.txt"
# train_src="data/2code/valid.de-en.de"
# train_tgt="data/2code/valid.de-en.en"

dev_src_code="/home/anushap/Code-Generation/nmt_model/data/2code/src_code_dev.txt"
dev_src_nl="/home/anushap/Code-Generation/nmt_model/data/2code/nl_dev.txt"
dev_tgt="/home/anushap/Code-Generation/nmt_model/data/2code/code_dev.txt"

test_src_code="/home/anushap/Code-Generation/nmt_model/data/2code/src_code_test.txt"
test_src_nl="/home/anushap/Code-Generation/nmt_model/data/2code/nl_test.txt"
test_tgt="/home/anushap/Code-Generation/nmt_model/data/2code/code_test.txt"

test_tgt_bleu="/home/anushap/Code-Generation/nmt_model/data/2code/code_test_bleu.txt"
dev_tgt_bleu="/home/anushap/Code-Generation/nmt_model/data/2code/code_dev_bleu.txt"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt_multi.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src-code ${train_src_code} \
    --train-src-nl ${train_src_nl} \
    --train-tgt ${train_tgt} \
    --dev-src-code ${dev_src_code} \
    --dev-src-nl ${dev_src_nl} \
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

python nmt_multi.py \
   decode \
   --cuda \
   --beam-size 10 \
   --max-decoding-time-step 20 \
   ${work_dir}/model.bin \
   ${test_src_code} \
   ${test_src_nl} \
   ${test_tgt} \
   ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt_bleu} < ${work_dir}/decode.txt
