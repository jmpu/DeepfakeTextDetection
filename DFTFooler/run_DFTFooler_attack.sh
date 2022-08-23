function run_dftfooler_attack {
    LOW_PROB_THRE=$1 # 0.01 by default
    MAX_ITER=$2 # number of word perturbations
    SIM_THRE=$3 # 0.7 by default
    ATTACK_DATASET=$4 # specify the dataset path to be attacked
    BACKEND_MODEL=$5 # choose para from ['bert', 'gpt2xl']
    NUM_SAMPLE=$6 # numbers of samples to be attacked
    CSV1="/csv1/xx.csv"
    CSV2="/csv2/xx.csv"
    OUTPUT_TEXT="./output_jsonl/xx.jsonl"

    export CUDA_VISIBLE_DEVICES=1
    nohup python3 -u DFTFooler_attack.py \
    --low_prob_thre=${LOW_PROB_THRE} \
    --max_iter=${MAX_ITER} \
    --sim_thre=${SIM_THRE} \
    --attack_dataset_path=${ATTACK_DATASET} \
    --backend_model=${BACKEND_MODEL} \
    --num_samples_to_attack=${NUM_SAMPLE} \
    --attack_stat_csv1=${CSV1} \
    --attack_stat_csv2=${CSV2} \
    --output_new_file=${OUTPUT_TEXT} \
    >./logs/xx.txt &
}


# The following cmd will run dftfooler attack on 1000 samples from dataset './df_1k_correct_512truncated.jsonl' usin bert LM backend (apply 10 word perturbations each document)
run_dftfooler_attack 0.01 10 0.7 './df_1k_correct_512truncated.jsonl' 'bert' 1000


