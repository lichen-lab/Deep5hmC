# Deep5hmC: Predicting genome-wide 5-Hydroxymethylcytosine landscape via multimodal deep learning model

We develop __Deep5hmC__, which is a multimodal deep learning framework that integrates both the DNA sequence and the histone modification, to predict genome-wide 5hmC modification. Moreover, __Deep5hmC__ demonstrates its practical usage by accurately predicting gene expression and differentially hydroxymethylated regions in a case-control Alzheimer’s disease study.

<br/>

![Deep5hmC.pdf](https://github.com/XinBiostats/Deep5hmC/files/14386695/figure2.pdf)


## Installation

1. Download Deep5hmC:
```bash
git clone https://github.com/XinBiostats/Deep5hmC
```
2. Requirements: Deep5hmC is implemented in Python. To install requirements
```bash
conda env create -f environment.yml
```
## Usage
1. Download data from [Dropbox](https://www.dropbox.com/scl/fi/5lc4sjudy1eby0ns80imq/3d_lipid.csv?rlkey=3e8yzh7gva8kzliwnc3xa2mt3&dl=0) and put downloaded data into data folder.
2. We created a demo ([demo.ipynb](https://github.com/XinBiostats/MetaVision3D/blob/main/demo.ipynb)) to demonstrate how to use __Deep5hmC__. The results will be displayed inline or saved by users.
