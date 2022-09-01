# Deepfake Text Detection

In this repository, we release code, datasets and model for the paper --- [Deepfake Text Detection: Limitations and Opportunities](https://jmpu.github.io/files/Deepfake%20Text%20Detection%20Limitations%20and%20Opportunities_CR.pdf) accepted by IEEE S&P 2023.

## Paper Abstract

*Recent advances in generative models for language have enabled the creation of convincing synthetic text or deepfake text. Prior work has demonstrated the potential for misuse of deepfake text to mislead content consumers. Therefore, deepfake text detection, the task of discriminating between human and machine-generated text, is becoming increasingly critical. Several defenses have been proposed for deepfake text detection. However, we lack a thorough understanding of their real-world applicability. In this paper, we collect deepfake text from 4 online services powered by Transformer-based tools to evaluate the generalization ability of the defenses on content in the wild. We develop several low-cost adversarial attacks, and investigate the robustness of existing defenses against an adaptive attacker. We find that many defenses show significant degradation in performance under our evaluation scenarios compared to their original claimed performance. Our evaluation shows that tapping into the semantic information in the text content is a promising approach for improving the robustness and generalization performance of deepfake text detection schemes.*

## Request In-The-Wild Synthetic Text Datasets

We collect 4 In-the-wild datasets from the web containing both synthetic and real articles from matching semantic categories. This includes synthetic text posted by Internet users, and text from synthetic text-generation-as-a-service platforms, geared towards the SEO community. While we could not verify the text generators used by the services we study, they claim to use customized versions of Transformer-based LMs. This again highlights the need to understand real-world performance of defenses, because text generators used in the wild can be different from those used by the research community.

#### Dataset Statistics

| Datasets     |#Document per class      | Document topics |
| ------------- | ------------- | -------- |
| AI-Writer     | 1000| News  |
| ArticleForge|1000|News|
|Kafkai|1000|Cyber Security, SEO, Marketing|
|RedditBot|887|Reddit Comments|

The In-The-Wild datasets can be requested using this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdgbiK97hnBWL1_98xIYjqWQpjeg9tzX49r0t7xGCrPkKLP-w/viewform?usp=sf_link).

## Evaluating Existing Deepfake Text Defenses

1. We evaluated 6 existing deepfake text detection schemes (BERT-Defense, GLTR-BERT, GLTR-GPT2, RoBERTa-Defense, GROVER, FAST) in the paper. The code and datasets of GROVER can be found in the original code [repository](https://github.com/rowanz/grover). The code for **BERT-Defense**, **RoBERTa-Defense**, **GLTR-BERT** and **GLTR-GPT2** can be found in each corresponding folder, i.e., ```./BERT-Defense```, ```./RoBERTa-Defense``` and ```./GLTR``` respectively. 

2. Datasets and (pretrained) models used by these defenses in the paper can be found [here](https://drive.google.com/drive/folders/1BD6i7MWYYPPFr5SK2EhdKBWx0W8SJO4L). To be more specific, these datasets and models were used to obtain evaluation results in Table I, III, V and VI of the paper.
File names of datasets and (pretrained) models are self-explanatory. 

3. We have not shared the code or pretrained models for [FAST](https://arxiv.org/abs/2010.07475)/DistilFAST as the code for FAST was not publicly released. Please contact the authors for the code.


## Our Proposed Adversarial Perturbation Attack -- DFTFooler
*Given a synthetic sample, our approach called DFTFooler, aims to misclassify it as real by adding adversarial perturbations to it. Unlike existing work on adversarial inputs in the text domain, DFTFooler requires no queries to the victim model, or a surrogate/shadow classifier to craft the perturbations. DFTFooler only requires a pre-trained LM, and several versions are publicly available today.*

1. We evaluated DFTFooler along with two other baseline methods, i.e., TextFooler and Random Perturbation. 

2. We release our code scripts for DFTFooler and Random Perturbations in the ```./DFTFooler``` subfolder, and the code of TextFooler can be found [here](https://github.com/jind11/TextFooler). 

## Evaluating Text Quality via GRUEN

We measure linguistic quality using the state-of-the-art [GRUEN](https://arxiv.org/abs/2010.02498) metric. The GRUEN metric was originally designed to evaluate the quality of short paragraphs of synthetic text. Due to the cumulative penalty function imposed by some of the subscore metrics of GRUEN, longer articles often get assigned a score of zero by it. Therefore, we contacted the authors of the GRUEN paper and implemented a methodology suggested by them to circumvent this issue. This involved splitting an article into multiple articles with equal number of sentences such that each article had an approximate average length of 120 words. We chose this configuration as the GRUEN scores for several datasets had a zero score below 1.5\%.

The code of GRUEN score adapted by us for longer documents evaluation can be found in ```./GRUEN``` subfolder.