## Code for DFTFooler

1. Install virtual environment, env requirements can be found in requirements.txt

2. Run experiments:

- Run DFTFooler attack: ```run_dftfooler_attack.sh```
- Run random perturbation attack: ```run_random_perturbation_attack.sh```

Note that description of important parameters can be found in the corresponding python script.

For the argument ```--counter_fitting_embeddings_path```, you can find the needed txt file ''counter-fitted-vectors.txt'' in the Google drive [folder](https://drive.google.com/drive/folders/1BD6i7MWYYPPFr5SK2EhdKBWx0W8SJO4L).

For the argument ```--counter_fitting_cos_sim_path```, we did not upload the numpy array used by us to the drive folder because its size is huge. We computed it using the [script](https://github.com/jind11/TextFooler/blob/master/comp_cos_sim_mat.py) provided by [TextFooler](https://github.com/jind11/TextFooler). Please pre-compute it by yourself to avoid the extra computation cost everytime you run the script.