# Deep5hmC: Predicting genome-wide 5-Hydroxymethylcytosine landscape via multimodal deep learning model

We develop __Deep5hmC__, which is a multimodal deep learning framework that integrates both the DNA sequence and the epigenetic features, to predict genome-wide 5hmC modification. Moreover, __Deep5hmC__ demonstrates its practical usage by accurately predicting gene expression and differentially hydroxymethylated regions in a case-control Alzheimer’s disease study.

<br/>

![Deep5hmC.pdf](figure1.png)


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
1. Download preprocessed "Forebrain Orgainoid" data at EB stage from [Dropbox](https://www.dropbox.com/scl/fo/zbht290yp67yfd1bcvofy/AGG8rFnKJxGTbmrv6uR_RSA?rlkey=3va94r26om8muivxho72wpzpp&dl=0) to './Deep5hmC/data'.
2. setup "./source/config.json" file.
### In terminal
- Activate your conda environment 'Deep5hmC' in the terminal.
```bash
conda activate Deep5hmC
```
- Run Deep5hmC models using sample data in the terminal.
```bash
python ./source/Deep5hmC_binary.py

python ./source/Deep5hmC_cont.py

python ./source/Deep5hmC_diff.py

python ./source/Deep5hmC_gene.py
```

### In Jupyter
We created a demo ([main.ipynb](https://github.com/XinBiostats/Deep5hmC/blob/main/source/main.ipynb)) to demonstrate how to use __Deep5hmC__. The results will be displayed inline or saved by users.
