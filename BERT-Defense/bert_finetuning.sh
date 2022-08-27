TRAIN_DATA="./xx.jsonl"
VAL_DATA="./xx.jsonl"
TEST_DATA="./xx.jsonl"
OUTPUT_DIR="./ckpts/xx"
SAVE_NAME="xx"


# BERT-Defense finetuning script
# Important parameter description can be found by ``python xx.py -h''
export CUDA_VISIBLE_DEVICES=1
nohup python3 -u bert_fine_tune.py \
--cache_dir='./models' \
--train_dir=${TRAIN_DATA} \
--val_dir=${VAL_DATA} \
--test_dir=${TEST_DATA} \
--prediction_output="./metrics/${SAVE_NAME}.jsonl" \
--output_dir=${OUTPUT_DIR} \
--logging_file="./logs/${SAVE_NAME}.txt" \
--tensor_logging_dir='./tf_logs' \
--train_batch_size=4 \
--val_batch_size=4 \
--token_len=512 \
--model_ckpt_path='./checkpoint-10000' \
--num_train_epochs=8 \
--save_steps=125 &
