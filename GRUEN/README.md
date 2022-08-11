# GRUEN
These scripts will evaluate a dataset using the GRUEN metric but in a customized way to allow for longer sentences/documents to be evaluated

## Run the scripts using the following steps

### 1 - Install the original GRUEN module

https://github.com/WanzhengZhu/GRUEN

### 2 - Preprocess your dataset by using the script ``dataset_split_gruen.py``
Use the following arguments

```
python3 dataset_split_gruen.py \
--input_dir='your/data/directory/*' \
--output_dir='output/directory/' \
--grover_ds=0 \ # Whether the dataset is Grover-styled or not
--len_disc=0  \ # Truncates dataset to the maximum context window size of the model you want to compute the GRUEN score for.
```


### 3 -  Replace the ```Main.py``` file with ```Main_l.py``` file

Use the following arguments
```
python3 Main_l.py \
--input_dir='input_directory_with_files/*' \
--cola_dir='path/to/bert/model/' \
--cache_dir='cache/directory/for/bert' \
--label='machine' \ # If the dataset has mixed files, choose which label to compute scores for 'machine' or 'human'
--discriminator_type='gltr' \ # 'gltr' or 'Grover' styled dataset being used
--output_dir='path/to/output/directory'     
```
