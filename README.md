# Land Cover Mapping on Sattelite Images

## About
This project implements a semantic segmentation pipeline using a U-Net architecture with a ResNet18 encoder pretrained on Sentinel-2 data. It includes preprocessing, patch creation, model training, fine-tuning, and evaluation for classifying land cover types from multi-spectral satellite imagery.

## Getting Started
### Setup conda enviroment:
 ```bash
conda create -n conda_env python=3.8
conda activate conda_env
``` 
### Intall dependencies:
```bash
pip install -r requirements.txt
``` 
## Project Structure
 ```bash

model/                        # Model architecture definitions
│   └──unet.py                # U-Net with pretrained ResNet18 (Sentinel-2)
│   
├── output/                    # Saved model weights and checkpoints
│   └── fine_tuning.pt         # Best model weights after fine-tuning
│
├── utils/                     # Utility functions and training components
│   ├── train.py               # Training and validation loops
│   ├── load_dataset.py        # Dataset loading and preprocessing
│   ├── utils.py               # Metrics, visualization, helpers
│   └── __init__.py
│
├── data/                      # Folder for storing satellite data
│
├── create_dataset.py          # preprocessing code of raw satelite images
├── pipeline.ipynb             # preprocessing to training pipeline
├── predict.ipynb              # prediction of model in unseen data
│
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
``` 
