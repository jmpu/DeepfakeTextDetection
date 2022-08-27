# install virtual env and then activate virtual env
source venv_path/bin/activate

# RoBERTa-Defense evaluation script
# Important parameter description can be found by ``python xx.py -h''
TEST_DATA="./grover_mega_4k_p096_prime1_mix.jsonl"
BASENAME="grover_mega_4k_p096_prime1_mix"

export CUDA_VISIBLE_DEVICES=1
nohup python3 -u ./roberta_defense_eval.py \
--cache_dir='./models' \
--test_dir="${TEST_DATA}" \
--prediction_output="./metrics/${BASENAME}.jsonl" \
--output_dir='./model/' \
--logging_file="./logging/${BASENAME}_logging.txt" \
--tensor_logging_dir='./tf_logs' \
--train_batch_size=1 \
--val_batch_size=32 \
--model_ckpt_path='./checkpoint-940' \
--num_train_epochs=6 \
--save_steps=100000 \
>./logs/${BASENAME}.txt &
