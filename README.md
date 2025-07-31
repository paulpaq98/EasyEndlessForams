# Endless Forams - Pytorch Implementation

📚 [Paper](https://hal.science/hal-02975093v1) | 🌐 [Website](http://endlessforams.org) | ✏️ [Simple Training Loop](https://https://github.com/paulpaq98/EasyEndlessForams/blob/main/train_model_classification.py)

<img src="assets/palo20770-fig-0006-m.jpg" alt="Forams Exemples" width="100%" />


This repository contains a pytorch implementation for a training loop based on the images from the endless forams dataset as described in:

```
Hsiang AY, Brombacher A, Rilo MC, Mleneck-Vautravers MJ, Conn S, Lordsmith S, Jentzen A, Henehan MJ, Metcalfe B, Fenton I, Wade B, Fox L, Meilland J, Davis CV, Baranowski U, Groeneveld J, Edgar KM, Movellan A, Aze T, Dowsett H, Miller G, Rios N & Hull PM. Endless Forams: >34,000 modern planktonic foraminiferal images for taxonomic training and automated species recognition using convolutional neural networks. Paleoceanography and Paleoclimatology. 34(7):1157-1177. (https://doi.org/10.1029/2019PA003612)
```

Training images and a taxonomic training module can be found at Endless Forams.


## Setup

To set up the environment using Conda (recommended):
```
conda create -n formas python=3.10
conda activate formas
pip install -r requirements.txt
```
## Running Code

Start by running a training classification script


##### Run example minimalistic training for an efficientNetV2

```
python .\train_model_classification.py 
```

## Repo Structure

```
├───assets
├───data
│   ├───img
│   └───masks
├───models
│   ├───classification  
│   └───detection
├───outputs
│   ├───classification
│   │   ├───accuracy_curve
│   │   └───loss_curve
│   └───detection
└───utils
```
