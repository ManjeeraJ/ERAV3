# Advanced Neural Network Architectures - CIFAR10 Classification

This folder contains implementation of advanced neural network architectures for image classification using the CIFAR10 dataset.

## Contents

- `Session_8.ipynb`: Jupyter notebook containing the main implementation
- `model_AC.py`: Model architecture definition
- `utils.py`: Utility functions for data loading and training

## Model Architecture

The implementation uses a Convolutional Neural Network (CNN) optimized for CIFAR10 classification. The model includes:
- Multiple convolutional layers
- Batch normalization
- Dropout for regularization
- Global average pooling
- Dense layers for classification

## Dataset

CIFAR10 dataset consists of 60,000 32x32 color images in 10 classes:
- Airplane
- Car 
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is split into 50,000 training images and 10,000 test images.

## Training Configuration

- Batch Size: 128 (GPU) / 64 (CPU)
- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.01
- Number of Epochs: 40
- Loss Function: Negative Log Likelihood
- Data Augmentation: Random crop, horizontal flip

## Results

The model achieves:
- Training Accuracy: ~75%
- Test Accuracy: ~82%

## Visualizations

The notebook provides several visualizations:
1. Training and test loss curves
2. Training and test accuracy curves
3. Sample images from the dataset
4. Grid of misclassified images with predicted vs actual labels

## Requirements

- PyTorch
- torchvision
- torchsummary
- matplotlib
- albumentations
- CUDA-capable GPU (recommended)

## Usage

1. Ensure all requirements are installed
2. Open `Session_8.ipynb` in Jupyter Notebook or Google Colab
3. If using Colab, mount your Google Drive
4. Run all cells sequentially
5. View training progress and final results

## Key Features

- GPU acceleration support
- Comprehensive data augmentation
- Detailed training metrics
- Visual analysis of model performance
- Error analysis through misclassified image visualization

## Model Performance Analysis

The notebook includes:
- Loss and accuracy tracking
- Learning rate analysis
- Model behavior visualization
- Misclassification analysis

## Training logs

Epoch 1
Train: Loss=1.4490 Batch_id=390 Accuracy=36.27: 100%|██████████| 391/391 [00:28<00:00, 13.69it/s]
Test set: Average loss: 1.4530, Accuracy: 4659/10000 (46.59%)

Epoch 2
Train: Loss=1.1896 Batch_id=390 Accuracy=46.99: 100%|██████████| 391/391 [00:28<00:00, 13.94it/s]
Test set: Average loss: 1.2543, Accuracy: 5496/10000 (54.96%)

Epoch 3
Train: Loss=1.3528 Batch_id=390 Accuracy=51.94: 100%|██████████| 391/391 [00:29<00:00, 13.27it/s]
Test set: Average loss: 1.1428, Accuracy: 5915/10000 (59.15%)

Epoch 4
Train: Loss=1.3560 Batch_id=390 Accuracy=55.48: 100%|██████████| 391/391 [00:28<00:00, 13.68it/s]
Test set: Average loss: 1.0306, Accuracy: 6346/10000 (63.46%)

Epoch 5
Train: Loss=1.1870 Batch_id=390 Accuracy=58.04: 100%|██████████| 391/391 [00:28<00:00, 13.91it/s]
Test set: Average loss: 0.9236, Accuracy: 6727/10000 (67.27%)

Epoch 6
Train: Loss=1.2844 Batch_id=390 Accuracy=60.44: 100%|██████████| 391/391 [00:28<00:00, 13.88it/s]
Test set: Average loss: 0.8782, Accuracy: 6938/10000 (69.38%)

Epoch 7
Train: Loss=1.0046 Batch_id=390 Accuracy=62.23: 100%|██████████| 391/391 [00:31<00:00, 12.55it/s]
Test set: Average loss: 0.8605, Accuracy: 6939/10000 (69.39%)

Epoch 8
Train: Loss=0.8588 Batch_id=390 Accuracy=63.77: 100%|██████████| 391/391 [00:29<00:00, 13.18it/s]
Test set: Average loss: 0.7820, Accuracy: 7279/10000 (72.79%)

Epoch 9
Train: Loss=1.0004 Batch_id=390 Accuracy=64.71: 100%|██████████| 391/391 [00:29<00:00, 13.13it/s]
Test set: Average loss: 0.7657, Accuracy: 7328/10000 (73.28%)

Epoch 10
Train: Loss=0.7905 Batch_id=390 Accuracy=65.51: 100%|██████████| 391/391 [00:29<00:00, 13.09it/s]
Test set: Average loss: 0.7466, Accuracy: 7434/10000 (74.34%)

Epoch 11
Train: Loss=1.0087 Batch_id=390 Accuracy=66.89: 100%|██████████| 391/391 [00:29<00:00, 13.12it/s]
Test set: Average loss: 0.7860, Accuracy: 7345/10000 (73.45%)

Epoch 12
Train: Loss=0.9659 Batch_id=390 Accuracy=67.60: 100%|██████████| 391/391 [00:29<00:00, 13.07it/s]
Test set: Average loss: 0.7274, Accuracy: 7467/10000 (74.67%)

Epoch 13
Train: Loss=0.9191 Batch_id=390 Accuracy=68.19: 100%|██████████| 391/391 [00:29<00:00, 13.08it/s]
Test set: Average loss: 0.6920, Accuracy: 7643/10000 (76.43%)

Epoch 14
Train: Loss=0.8678 Batch_id=390 Accuracy=68.57: 100%|██████████| 391/391 [00:32<00:00, 12.05it/s]
Test set: Average loss: 0.6819, Accuracy: 7649/10000 (76.49%)

Epoch 15
Train: Loss=0.8467 Batch_id=390 Accuracy=69.06: 100%|██████████| 391/391 [00:29<00:00, 13.25it/s]
Test set: Average loss: 0.6808, Accuracy: 7691/10000 (76.91%)

Epoch 16
Train: Loss=0.9716 Batch_id=390 Accuracy=69.50: 100%|██████████| 391/391 [00:29<00:00, 13.22it/s]
Test set: Average loss: 0.6585, Accuracy: 7717/10000 (77.17%)

Epoch 17
Train: Loss=0.8218 Batch_id=390 Accuracy=69.99: 100%|██████████| 391/391 [00:29<00:00, 13.20it/s]
Test set: Average loss: 0.6496, Accuracy: 7766/10000 (77.66%)

Epoch 18
Train: Loss=0.8748 Batch_id=390 Accuracy=70.91: 100%|██████████| 391/391 [00:29<00:00, 13.48it/s]
Test set: Average loss: 0.6235, Accuracy: 7837/10000 (78.37%)

Epoch 19
Train: Loss=1.0140 Batch_id=390 Accuracy=70.92: 100%|██████████| 391/391 [00:29<00:00, 13.47it/s]
Test set: Average loss: 0.6326, Accuracy: 7819/10000 (78.19%)

Epoch 20
Train: Loss=0.8050 Batch_id=390 Accuracy=71.24: 100%|██████████| 391/391 [00:29<00:00, 13.44it/s]
Test set: Average loss: 0.6232, Accuracy: 7838/10000 (78.38%)

Epoch 21
Train: Loss=0.7010 Batch_id=390 Accuracy=71.44: 100%|██████████| 391/391 [00:29<00:00, 13.26it/s]
Test set: Average loss: 0.6113, Accuracy: 7933/10000 (79.33%)

Epoch 22
Train: Loss=0.8163 Batch_id=390 Accuracy=71.90: 100%|██████████| 391/391 [00:31<00:00, 12.28it/s]
Test set: Average loss: 0.6079, Accuracy: 7904/10000 (79.04%)

Epoch 23
Train: Loss=0.7213 Batch_id=390 Accuracy=71.99: 100%|██████████| 391/391 [00:28<00:00, 13.60it/s]
Test set: Average loss: 0.6103, Accuracy: 7893/10000 (78.93%)

Epoch 24
Train: Loss=0.6287 Batch_id=390 Accuracy=72.66: 100%|██████████| 391/391 [00:29<00:00, 13.27it/s]
Test set: Average loss: 0.5942, Accuracy: 7982/10000 (79.82%)

Epoch 25
Train: Loss=0.8709 Batch_id=390 Accuracy=72.72: 100%|██████████| 391/391 [00:29<00:00, 13.31it/s]
Test set: Average loss: 0.5850, Accuracy: 7986/10000 (79.86%)

Epoch 26
Train: Loss=0.9703 Batch_id=390 Accuracy=72.69: 100%|██████████| 391/391 [00:29<00:00, 13.24it/s]
Test set: Average loss: 0.5735, Accuracy: 8028/10000 (80.28%)

Epoch 27
Train: Loss=0.9548 Batch_id=390 Accuracy=73.22: 100%|██████████| 391/391 [00:29<00:00, 13.20it/s]
Test set: Average loss: 0.5882, Accuracy: 7979/10000 (79.79%)

Epoch 28
Train: Loss=0.6751 Batch_id=390 Accuracy=73.34: 100%|██████████| 391/391 [00:31<00:00, 12.40it/s]
Test set: Average loss: 0.5624, Accuracy: 8068/10000 (80.68%)

Epoch 29
Train: Loss=0.7615 Batch_id=390 Accuracy=73.47: 100%|██████████| 391/391 [00:33<00:00, 11.79it/s]
Test set: Average loss: 0.5587, Accuracy: 8087/10000 (80.87%)

Epoch 30
Train: Loss=0.5819 Batch_id=390 Accuracy=73.55: 100%|██████████| 391/391 [00:29<00:00, 13.04it/s]
Test set: Average loss: 0.5643, Accuracy: 8096/10000 (80.96%)

Epoch 31
Train: Loss=0.6932 Batch_id=390 Accuracy=74.00: 100%|██████████| 391/391 [00:29<00:00, 13.06it/s]
Test set: Average loss: 0.5516, Accuracy: 8135/10000 (81.35%)

Epoch 32
Train: Loss=0.7349 Batch_id=390 Accuracy=74.06: 100%|██████████| 391/391 [00:30<00:00, 12.80it/s]
Test set: Average loss: 0.5631, Accuracy: 8104/10000 (81.04%)

Epoch 33
Train: Loss=0.6324 Batch_id=390 Accuracy=74.66: 100%|██████████| 391/391 [00:29<00:00, 13.17it/s]
Test set: Average loss: 0.5678, Accuracy: 8044/10000 (80.44%)

Epoch 34
Train: Loss=0.7277 Batch_id=390 Accuracy=74.26: 100%|██████████| 391/391 [00:29<00:00, 13.12it/s]
Test set: Average loss: 0.5434, Accuracy: 8150/10000 (81.50%)

Epoch 35
Train: Loss=0.6618 Batch_id=390 Accuracy=74.83: 100%|██████████| 391/391 [00:29<00:00, 13.31it/s]
Test set: Average loss: 0.5331, Accuracy: 8196/10000 (81.96%)

Epoch 36
Train: Loss=0.8535 Batch_id=390 Accuracy=74.79: 100%|██████████| 391/391 [00:29<00:00, 13.19it/s]
Test set: Average loss: 0.5317, Accuracy: 8209/10000 (82.09%)

Epoch 37
Train: Loss=0.6205 Batch_id=390 Accuracy=75.20: 100%|██████████| 391/391 [00:31<00:00, 12.31it/s]
Test set: Average loss: 0.5383, Accuracy: 8165/10000 (81.65%)

Epoch 38
Train: Loss=0.7004 Batch_id=390 Accuracy=75.27: 100%|██████████| 391/391 [00:28<00:00, 13.61it/s]
Test set: Average loss: 0.5365, Accuracy: 8209/10000 (82.09%)

Epoch 39
Train: Loss=0.6462 Batch_id=390 Accuracy=74.90: 100%|██████████| 391/391 [00:28<00:00, 13.55it/s]
Test set: Average loss: 0.5267, Accuracy: 8172/10000 (81.72%)

Epoch 40
Train: Loss=0.6895 Batch_id=390 Accuracy=75.30: 100%|██████████| 391/391 [00:28<00:00, 13.71it/s]
Test set: Average loss: 0.5383, Accuracy: 8180/10000 (81.80%)