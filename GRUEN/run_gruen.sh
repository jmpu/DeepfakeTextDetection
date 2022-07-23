# Preprocessing script for your dataset

python3 /rdata/zainsarwar865/Scripts/Scripts/dataset_split_gruen.py \
--input_dir='/rdata/zainsarwar865/outputs/attack_datasets/processed/GRUEN/old/gbrain/gen/*' \
--output_dir='/rdata/zainsarwar865/outpaauts/attack_datasets/processed/GRUEN/new/gbrain/gen/' \
--grover_ds=0 \
--len_disc=0  


# Change to path where GRUEN is installed

cd /rdata/zainsarwar865/venv_2/lib/python3.6/site-packages/GRUEN/
python3 Main_l.py \
--input_dir='/rdata/zainsarwar865/outputs/attack_datasets/processed/GRUEN/new/gbrain/gen/*' \
--cola_dir='/rdata/zainsarwar865/models/cola_model/bert-base-cased/' \
--cache_dir='/rdata/zainsarwar865/models' \
--label='machine' \
--discriminator_type='gltr' \ 
--output_dir='/rdata/zainsarwar865/outputs/attack_results/GRUEN_NEW/machine/gbrain/'     