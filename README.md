# DeepfakeTextDetection
Code and datasets for the paper "Deepfake Text Detection: Limitations and Opportunities"

In this repository, we release code scripts, datasets and model for main evaluation experiments of paper --- "Deepfake Text Detection: Limitations and Opportunities". 

1. We evaluated 6 deepfake text detection schemes (BERT-Defense, GLTR-BERT, GLTR-GPT2, RoBERTa-Defense, GROVER, FAST) in total. The code and dataset of GROVER can be found in the original code [repository](https://github.com/rowanz/grover). The code scripts for BERT-Defense, RoBERTa-Defense, GLTR-BERT and GLTR-GPT2 can be found in each corresponding folder.
Datasets used for training and evaluating the models can be found [here](https://drive.google.com/drive/folders/1lFxw23DaGm3UMoSVVR2zT3dPQqKCHFi7). Checkpoints for the pretrained models can be found [here](https://drive.google.com/drive/folders/1BD6i7MWYYPPFr5SK2EhdKBWx0W8SJO4L). Names of datasets and pretrained models are self-explanatory. 
We have not shared the code or pretrained models for FAST/DistilFAST as the code for FAST was not publicly released. Please contact the authors for the code.


2. We proposed a novel block-box adversarial perturbation attack called **DFTFooler**. We evaluated DFTFooler along with two other baseline methods, i.e., TextFooler and Random Perturbation. We release our code scripts for DFTFooler and Random Perturbations, and the code of TextFooler can be found [here](https://github.com/jind11/TextFooler). 



The In-The-Wild datasets can be accessed using this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdgbiK97hnBWL1_98xIYjqWQpjeg9tzX49r0t7xGCrPkKLP-w/viewform?usp=sf_link).