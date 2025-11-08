#!/bin/bash

# IWSLT2017 德语到英语翻译训练命令
# torchrun --standalone --nproc_per_node=2 ../src/train.py \
#     --dataset iwslt \
#     --data_dir ../data \
#     --language_pair de-en \
#     --batch_size 256 \
#     --epochs 100 --ddp --dist_backend nccl --max_len 120 --d_model 128 --n_layers 3 --n_heads 8 --d_ff 512 --dropout 0.3 --label_smoothing 0.10 --weight_decay 0.02 --patience 5 --save_dir ../results \
#     --use_adamw \
#     --lr 0.0005 \
#     --scheduler cosine \

torchrun --standalone --nproc_per_node=2 ../src/train.py \
    --dataset iwslt \
    --data_dir ../data \
    --language_pair de-en \
    --batch_size 256 \
    --epochs 100 --ddp --dist_backend nccl --max_len 100 --d_model 256 --n_layers 3 --n_heads 8 --d_ff 1024 --dropout 0.3 --label_smoothing 0.10 --weight_decay 0.02 --patience 5 --save_dir ../results \
    --use_adamw \
    --lr 0.0005 \
    --scheduler cosine \
    --is_pos_encoding False