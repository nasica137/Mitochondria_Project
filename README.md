# Mitochondria Segmentation Project

This project focuses on automating the segmentation of mitochondria using deep learning models, specifically through transfer learning techniques. The primary goal is to accurately distinguish between untreated and treated mitochondria. The project comprises two main approaches: a model directly trained on the Uniklinik dataset and a model pretrained on the MitoEM dataset, followed by fine-tuning on the Uniklinik dataset.

## Overview

- **Objective:** Accurate segmentation of mitochondria in untreated and treated conditions.
- **Approaches:**
  - Model trained on the Uniklinik dataset.
  - Model pretrained on the MitoEM dataset and fine-tuned on Uniklinik.

## Methodology

The project includes adaptations for each model involving various encoders and loss functions. Evaluation metrics, such as Intersection over Union (IoU), are used to assess model performance. The study also explores the optimization of data augmentation for multiclass segmentation.

## Key Results

- Average IoU:
  - Non-pretrained model: 0.8089
  - Pretrained model: 0.9439
- Class-wise IoU scores for the pretrained model:
  - Background: 0.9664
  - Untreated: 0.9271
  - Treated: 0.9383
- Class-wise IoU scores for the non-pretrained model:
  - Background: 0.9284
  - Untreated: 0.6942
  - Treated: 0.8039
 
## Usage




## Usage

The proper way to run the a prediction of the best model is by invoking the cross_validation_predict_large_image.py script with python3.8 or later:

Not Pretrained:
```bash
python cross_validation_predict_large_image.py --encoder==mobilenet_v2 --loss==DiceLoss
```
Pretrained:
```bash
python cross_validation_predict_large_image.py --encoder==vgg16 --loss==DiceFocalLoss
```
