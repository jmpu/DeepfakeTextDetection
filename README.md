# DeepfakeTextDetection
Code and datasets for the paper "Deepfake Text Detection: Limitations and Opportunities"

In this repository, we release code scripts, datasets and model for main evaluation experiments of paper --- XXX. 

1. In this paper, we evaluated 6 deepfake text detection schemes (BERT-Defense, GLTR-BERT, GLTR-GPT2, RoBERTa-Defense, GROVER, FAST) in total. Of the 6 defenses, we are not able to release code for FAST (as well as DistilFAST discussed in the paper) because we do not have the original author's permission. The code of GROVER can be found in the original code repository. The code scripts for BERT-Defense, RoBERTa-Defense, GLTR-BERT and GLTR-GPT2 can be found in each folder. 

2. In this paper, we propose a novel adversarial attack called **DFTFooler**. We evaluated DFTFooler along with two other baseline methods, i.e., TextFooler and Random Perturbation. We release our code scripts for DFTFooler and Random Perturbations, and the code of TextFooler can be found here. 


Checkpoints for the pretrained models can be found <a href="https://drive.google.com/drive/folders/1BD6i7MWYYPPFr5SK2EhdKBWx0W8SJO4L">here </a>

Datasets used for training and evaluating the models can be found <a href="https://drive.google.com/drive/folders/1lFxw23DaGm3UMoSVVR2zT3dPQqKCHFi7"> here </a> 

We have not shared the code or pretrained models for FAST/DistilFAST as the code for FAST was not publicly released. Please contact the authors for the code.

The In-The-Wild datasets can be accessed using this <a href="https://docs.google.com/forms/d/1MPEuRO_RUPZR1jrSXC-VCXJviw8_62UOafGjWjxTNro/edit"> Google Form</a>
