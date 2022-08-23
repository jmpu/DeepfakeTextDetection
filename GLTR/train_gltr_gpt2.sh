export CUDA_VISIBLE_DEVICES=1

nohup python3 -u gltr_train_gpt2xl.py \
--train_dataset='./processed_GLTR_gpt2xl_TRAIN_k40_temp07_4k_mixed.jsonl' \
--gpt2_xl_gltr_ckpt='./xx.sav' \
--output_metrics='./xx.jsonl' \
--gpt2_model='gpt2-xl' \
>./xx.txt &



