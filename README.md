# Deep5hmC: Predicting genome-wide 5-Hydroxymethylcytosine landscape via multimodal deep learning model

We introduce __MetaVision3D__, a novel pipeline driven by computer vision techniques for the transformation of serial 2D MALDI mass spectrometry imaging sections into a high-resolution 3D spatial metabolome. Our framework employs advanced algorithms for image registration, normalization, and interpolation to enable the integration of serial 2D tissue sections, thereby generating a comprehensive 3D model of unique diverse metabolites across host tissues at mesoscale.
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
2. We created a demo ([demo.ipynb](https://github.com/XinBiostats/MetaVision3D/blob/main/demo.ipynb)) to demonstrate how to use MetaVision3D. The results will be displayed inline or saved by users.
