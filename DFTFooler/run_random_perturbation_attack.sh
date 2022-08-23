function run_random_perturb_attack {
    MAX_ITER=$1 # number of word perturbations
    SIM_THRE=$2 # sentence semantic similarity threshold, 0.7 by default
    ATTACK_DATASET=$3
    NUM_SAMPLE=$4
    CSV1="./csv1/xx.csv"
    CSV2="./csv2/xx.csv"
    OUTPUT_TEXT="./output_jsonl/xx.jsonl"


    export CUDA_VISIBLE_DEVICES=0
    nohup python3 -u random_perturbation_attack.py \
    --max_iter=${MAX_ITER} \
    --sim_thre=${SIM_THRE} \
    --attack_dataset=${ATTACK_DATASET} \
    --num_samples_to_attack=${NUM_SAMPLE} \
    --attack_stat_csv1=${CSV1} \
    --attack_stat_csv2=${CSV2} \
    --output_new_file=${OUTPUT_TEXT} \
    >./logs/xx.txt &
}

# The following cmd will run random perturbation attack on 1000 samples from dataset './df_1k_correct_512truncated.jsonl' (apply 10 word perturbations each document)
run_random_perturb_attack 10 0.7 './df_1k_correct_512truncated.jsonl' 1000













