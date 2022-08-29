export CUDA_VISIBLE_DEVICES=1

nohup python3 -u gltr_test_gpt2xl.py \
--test_dataset='xx.jsonl' \
--gpt2_xl_gltr_ckpt='./ckpts/GLTR_gpt2xl_TRAIN_k40_temp07_mix_512.sav' \
--gpt2_model='gpt2-xl' \
--return_stat_file='./hist_stats/xx.jsonl' \
--output_metrics='./metrics/xx.jsonl' \
>./logs/xx.txt &
