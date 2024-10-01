# VisDist-Net

## Overview

The VisDist-Net project aims to classify fruits as either fresh or rotten using a novel deep learning architecture. This repository provides scripts and models to train, evaluate, and utilize the VisDist-Net for fruit classification tasks.

## Installation

Ensure you have Python 3.6 or higher. Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Original Dataset Structure

The dataset used for training and testing the models can be found on Kaggle: [Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification). 
The original dataset is structured as follows:

```plaintext
├── fruit_dataset/
│   ├── fresh apple/        
│   ├── fresh banana/       
│   ├── fresh orange/
│   ├── rotten apple/
│   ├── rotten banana/
│   └── rotten orange/
```

## Dataset Preparation

### 1. split_dataset_into_folds.py

**Description**: This script splits the dataset into multiple folds for cross-validation.

**Usage**: 
```bash
python split_dataset_into_folds.py --input-dir <path_to_dataset> --output-dir <output_directory>
```
### 2. create_validating.py

**Description**: Creates a validation dataset after splitting the folds.

**Usage**: 
```bash
python create_validating.py --fold-dir <path_to_fold_dataset>
```
After splitting the dataset into multiple folds, the dataset structure should be as follows:

```plaintext
├── fold_dataset/
│   ├── Fold1                 # Fold1
│   │   ├── Train/             # Training data
│   │   │   ├── fresh apple/        
│   │   │   ├── fresh banana/       
│   │   │   ├── fresh orange/
│   │   │   ├── rotten apple/
│   │   │   ├── rotten banana/
│   │   │   └── rotten orange/
│   │   ├── Val/               # Validating data
│   │   │   ├── fresh apple/        
│   │   │   ├── fresh banana/       
│   │   │   ├── fresh orange/
│   │   │   ├── rotten apple/
│   │   │   ├── rotten banana/
│   │   │   └── rotten orange/  
│   │   └── Test/              # Testing data
│   │       ├── fresh apple/        
│   │       ├── fresh banana/       
│   │       ├── fresh orange/
│   │       ├── rotten apple/
│   │       ├── rotten banana/
│   │       └── rotten orange/      
│   ├── Fold2                 # Organize it as Fold1
│   ├── Fold3                 # Organize it as Fold1
│   ├── Fold4                 # Organize it as Fold1
│   ├── Fold5                 # Organize it as Fold1
```

## Training

### 1. train.py

**Description**: Trains the `mobilenetv1`, `resnet18`, `vision transformer` and `hybrid-cnn` models independently.

**Usage**: 
```bash
python train.py
```

### 2. train_knowledge_distillation.py

**Description**: Trains the `hybrid-cnn` model with knowledge distillation enabled.

**Usage**: 
```bash
python train_knowledge_distillation.py 
```

## Testing

### 1. test.py

**Description**: Evaluates the trained model's performances. To test the models the structure should be as follows: 

The training, testing and validating data can be downloaded using the following URL: [Fold Dataset](https://drive.google.com/file/d/1KPtA88ITmXXn26jYl8xnIt-QN8ySzJt4/view?usp=sharing)

The checkpoints can be downloaded using the following URL: [Checkpoints](https://drive.google.com/file/d/1SppjdjHgpbh7dBBSZY_Ur5gHQaHbOGyf/view?usp=sharing)

**Usage**: 
```bash
python test.py 
```
