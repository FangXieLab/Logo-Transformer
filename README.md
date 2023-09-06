# Logo-Transformer for Generating Fantastic Logos by Brand Names

This repository provides code corresponding to the report *Logo-Transformer for Generating Fantastic Logos by Brand Names* by Xiaolin Zhu and Fang Xie.

## Code Files

Python files include:  
```Logo_Transformer.py``` to build our proposed Logo-Transformer network;   
```train_logo_transformer.py``` to train Logo-Transformer based on the training set and visualize the training process;   
```test_logo_transformer.py``` to test model performance by presenting losses and accuracy and analysing the attention distribution;   
```generate_logo_transformer.py``` to implement logo generation inference given the required brand name and the desired style;   
```resize_images.py``` to resize logo images to different sizes;  
```process_logo_data_logo2k.py``` to cluster Logo-2K+ images into different clusters and split into training, validation and testing sets for the following experiment;  
```cluster.py``` to run ```process_logo_data_logo2k.py``` based on a pre-defined cluster number.

## Dataset

The dataset involved in this study consists of [Famous Brand Logos](https://www.kaggle.com/datasets/linkanjarad/famous-brand-logos) and [Logo-2K+](https://paperswithcode.com/dataset/logo-2k).

## Environmental Setup

Packages required: python(3.8.11), pytorch(2.0.0), torchvision(0.15.1), numpy, pandas, sklearn, math, matplotlib, seaborn, time, random, pickle, PIL, kneed

## Acknowledgements

The file ```Logo_Transformer.py``` is based on codes of [Image Transformer](https://github.com/sahajgarg/image\_transformer) and [Transformer for language translation](https://github.com/tunz/transformer-pytorch/tree/master).
