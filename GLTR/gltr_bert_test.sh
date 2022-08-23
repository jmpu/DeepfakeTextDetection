export CUDA_VISIBLE_DEVICES=0

nohup python3 -u gltr_test_bert.py \
--test_dataset='xx.jsonl' \
--bert_large_gltr_ckpt='./ckpts/GLTR_bert_TRAIN_k40_temp07_mix_512.sav' \
--bert_model='bert-large-cased' \
--return_stat_file="./hist_stats/xx.jsonl" \
--output_metrics="./metrics/xx.jsonl" \
>./logs/xx.txt &

