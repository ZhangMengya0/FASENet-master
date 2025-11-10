FASENet: a frequency aware and shape enhanced deep network for mapping farmland ditch from high-resolution remote sensing imagery

## Introduction
Accurately mapping farmland ditches from high-resolution remote sensing imagery is of great significance for improving agricultural water resource management, optimizing farmland planning, and advancing smart agriculture. However, farmland ditches exhibit diverse scales, long linear shapes, frequent vegetation coverage, and are easily confused with other linear objects, which poses challenges to precise mapping. To address this issue, a Frequency Aware and Shape Enhanced Deep Network (FASENet) was proposed. A Frequency Aware Attention Module is developed to enhance the model’s ability to distinguish farmland ditches from similar objects. A Multi-Scale Global Feature Aggregation Module was designed to reduce information loss caused by the long span and scale differences of farmland ditches. Additionally, we proposed a Dual-Branch Parallel Decoder that integrates shape enhancement and gating mechanisms to ensure the continuity of the farmland ditches. To validate our proposed model, we constructed manually annotated farmland ditch datasets, which were derived from GF-2 imagery spanning paddy fields and irrigated fields in Hubei and Shandong provinces, China. The results showed that FASENet achieved high accuracy on both datasets, with F1-scores of 91.09% and 89.07%, respectively. Compared with the different models, FASENet achieved the best accuracy, while ablation experiments confirmed that the proposed modules effectively boosted the model's accuracy. In particular, the Frequency Aware Attention can accurately distinguish farmland ditches from similar objects by combining frequency features, thereby improving the accuracy of the model. Furthermore, the transfer of the model across regions and images demonstrated its excellent generalization ability.

## License

This project is licensed under the **CC0 1.0 Universal Public Domain Dedication** (CC0 1.0).

[![CC0 1.0 Universal](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

## Requirements

The code is built with the following dependencies:

- Python 3.11 or higher
- CUDA 11.8 or higher
- [PyTorch](https://pytorch.org/) 2.2.2 or higher
- pytorch-wavelets
- pillow
- numpy

## Training
python train.py

## Evaluation
python test.py

