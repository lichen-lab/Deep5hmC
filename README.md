# Deep5hmC: Predicting genome-wide 5-Hydroxymethylcytosine landscape via multimodal deep learning model

We develop __Deep5hmC__, which is a multimodal deep learning framework that integrates both the DNA sequence and the histone modification, to predict genome-wide 5hmC modification. Moreover, __Deep5hmC__ demonstrates its practical usage by accurately predicting gene expression and differentially hydroxymethylated regions in a case-control Alzheimer’s disease study.

<br/>

![Deep5hmC.pdf](https://github.com/XinBiostats/Deep5hmC/files/14386695/figure2.pdf)


## Installation

1. Download __Deep5hmC__:
```bash
git clone https://github.com/XinBiostats/Deep5hmC
```
2. Requirements: __Deep5hmC__ is implemented in Python. To install requirements
```bash
conda env create -f environment.yml
```
## Usage
1. Download data and pretrained models from [Dropbox](https://www.dropbox.com/scl/fo/m1p1i6d4goigafokadfxb/h?rlkey=apjt44fxmqcwj56wienw76w8w&dl=1).
2. Run Deep5hmC-binary model in the terminal.
```bash
python Deep5hmC_binary.py
```
3. Run Deep5hmC-binary model in Jupyter Notebook.<br/>
We created a demo ([main.ipynb](https://github.com/XinBiostats/Deep5hmC/blob/main/source/main.ipynb)) to demonstrate how to use __Deep5hmC__. The results will be displayed inline or saved by users.
