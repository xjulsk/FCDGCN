#!/bin/bash
# * laptop

# * FCDGCN
python ./train.py --model_name FCDGCN --dataset laptop --seed 1000 --num_epoch 40 --vocab_dir ./dataset/Laptops_corenlp --cuda 3  

# * FCDGCN with Bert
# python ./train.py --model_name FCDGCN_bert --dataset laptop --seed 1000 --bert_lr 2e-5 --num_epoch 10 --hidden_dim 768 --max_length 100 --cuda 0  


# * restaurant

# * FCDGCN
#python ./train.py --model_name FCDGCN --dataset restaurant --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Restaurants_corenlp --cuda 0  
# * FCDGCN with Bert
# python ./train.py --model_name FCDGCN_bert --dataset restaurant --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 

# * twitter

# * FCDGCN
# python ./train.py --model_name FCDGCN --dataset twitter --seed 1000 --num_epoch 40 --vocab_dir ./dataset/Tweets_corenlp --cuda 0 

# * FCDGCN with Bert
# python ./train.py --model_name FCDGCN_bert --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 
