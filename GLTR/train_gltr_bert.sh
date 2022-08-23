export CUDA_VISIBLE_DEVICES=1

nohup python3 -u gltr_train_bert.py \
--train_dataset='./processed_GLTR_gpt2xl_TRAIN_k40_temp07_4k_mixed.jsonl' \
--bert_large_gltr_ckpt='./ckpts/xx.sav' \
--output_metrics='./metrics/xx.jsonl' \
--bert_model='bert-large-cased' \
>./logs/xx.txt &



