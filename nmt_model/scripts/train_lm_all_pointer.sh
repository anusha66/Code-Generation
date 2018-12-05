#!/bin/sh

# vocab="data/vocab.bin"
# train_src="data/nl_train.txt"
# train_tgt="data/code_train.txt"
# dev_src="data/nl_dev.txt"
# dev_tgt="data/code_dev.txt"
# test_src="data/nl_test.txt"
# test_tgt="data/code_test.txt"
# test_tgt_bleu="data/code_test_bleu.txt"
# dev_tgt_bleu="data/code_dev_bleu.txt"

vocab="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/vocab.bin"
train_src="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/nl_train.txt"
train_tgt="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/code_train.txt"
dev_src="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/nl_dev.txt"
dev_tgt="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/code_dev.txt"
test_src="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/nl_test.txt"
test_tgt="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/code_test.txt"
test_tgt_bleu="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/code_test_bleu.txt"
dev_tgt_bleu="/home/anushap/Code-Generation-lm/nmt_model/data/code2code/code_dev_bleu.txt"


# vocab="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/vocab.bin"
# train_src="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/nl_train.txt"
# train_tgt="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/code_train.txt"
# dev_src="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/nl_dev.txt"
# dev_tgt="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/code_dev.txt"
# test_src="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/nl_test.txt"
# test_tgt="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/code_test.txt"
# test_tgt_bleu="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/code_test_bleu.txt"
# dev_tgt_bleu="/home/anushap/Code-Generation-lm/nmt_model/data/nl2code/code_dev_bleu.txt"


work_dir="work_dir_lm_all_pointer_tgt"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt_lm_all_pointer_tgt.py \
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

python nmt_lm_all_pointer_tgt.py \
   decode \
   --cuda \
   --beam-size 5 \
   --max-decoding-time-step 50 \
   ${work_dir}/model.bin \
   ${test_src} \
   ${test_tgt} \
   ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt_bleu} < ${work_dir}/decode.txt
