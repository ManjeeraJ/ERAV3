# CNN Implementation and Training Pipeline

This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch.

## Project Structure 

- `model.py`: Defines the CNN architecture.
- `train.py`: Implements the training pipeline.
- `test_model.py`: Contains unit tests for the model.
- `utils.py`: Utility functions for data loading and model evaluation.

## Dependencies

`pip install -r requirements.txt`

## Usage

To train the model, run `python train.py`.

To run unit tests, run `python test_model.py`.

To evaluate the model, run `python utils.py`.

## Training logs

Epoch 1
Train: Loss=0.0141 Batch_id=937 Accuracy=89.31: 100%|██████████| 938/938 [01:23<00:00, 11.27it/s]
Test set: Average loss: 0.0811, Accuracy: 9790/10000 (97.90%)

Epoch 2
Train: Loss=0.0505 Batch_id=937 Accuracy=97.35: 100%|██████████| 938/938 [01:23<00:00, 11.18it/s]
Test set: Average loss: 0.0750, Accuracy: 9779/10000 (97.79%)

Epoch 3
Train: Loss=0.2704 Batch_id=937 Accuracy=97.81: 100%|██████████| 938/938 [01:19<00:00, 11.75it/s]
Test set: Average loss: 0.0495, Accuracy: 9847/10000 (98.47%)

Epoch 4
Train: Loss=0.1310 Batch_id=937 Accuracy=98.10: 100%|██████████| 938/938 [01:27<00:00, 10.77it/s]
Test set: Average loss: 0.0376, Accuracy: 9885/10000 (98.85%)

Epoch 5
Train: Loss=0.0622 Batch_id=937 Accuracy=98.26: 100%|██████████| 938/938 [01:21<00:00, 11.52it/s]
Test set: Average loss: 0.0397, Accuracy: 9881/10000 (98.81%)

Epoch 6
Train: Loss=0.0285 Batch_id=937 Accuracy=98.68: 100%|██████████| 938/938 [01:24<00:00, 11.12it/s]
Test set: Average loss: 0.0308, Accuracy: 9907/10000 (99.07%)

Epoch 7
Train: Loss=0.0691 Batch_id=937 Accuracy=98.78: 100%|██████████| 938/938 [01:21<00:00, 11.47it/s]
Test set: Average loss: 0.0310, Accuracy: 9903/10000 (99.03%)

Epoch 8
Train: Loss=0.0804 Batch_id=937 Accuracy=98.77: 100%|██████████| 938/938 [01:19<00:00, 11.84it/s]
Test set: Average loss: 0.0300, Accuracy: 9909/10000 (99.09%)

Epoch 9
Train: Loss=0.0293 Batch_id=937 Accuracy=98.77: 100%|██████████| 938/938 [01:26<00:00, 10.86it/s]
Test set: Average loss: 0.0286, Accuracy: 9909/10000 (99.09%)

Epoch 10
Train: Loss=0.0768 Batch_id=937 Accuracy=98.82: 100%|██████████| 938/938 [01:19<00:00, 11.73it/s]
Test set: Average loss: 0.0299, Accuracy: 9905/10000 (99.05%)

Epoch 11
Train: Loss=0.1250 Batch_id=937 Accuracy=98.82: 100%|██████████| 938/938 [01:21<00:00, 11.55it/s]
Test set: Average loss: 0.0286, Accuracy: 9913/10000 (99.13%)

Epoch 12
Train: Loss=0.0179 Batch_id=937 Accuracy=98.87: 100%|██████████| 938/938 [01:19<00:00, 11.83it/s]
Test set: Average loss: 0.0292, Accuracy: 9907/10000 (99.07%)

Epoch 13
Train: Loss=0.0159 Batch_id=937 Accuracy=98.83: 100%|██████████| 938/938 [01:21<00:00, 11.53it/s]
Test set: Average loss: 0.0297, Accuracy: 9907/10000 (99.07%)

Epoch 14
Train: Loss=0.0055 Batch_id=937 Accuracy=98.86: 100%|██████████| 938/938 [01:18<00:00, 11.94it/s]
Test set: Average loss: 0.0297, Accuracy: 9907/10000 (99.07%)

Epoch 15
Train: Loss=0.0093 Batch_id=937 Accuracy=98.87: 100%|██████████| 938/938 [01:20<00:00, 11.62it/s]
Test set: Average loss: 0.0286, Accuracy: 9915/10000 (99.15%)
