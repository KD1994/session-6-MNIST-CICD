# Simple Neural Network for MNIST

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3810/)  [![Python application](https://github.com/KD1994/session-6-MNIST-CICD/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/KD1994/session-6-MNIST-CICD/actions/workflows/python-app.yml)


This repository contains a simple neural network implemented in PyTorch to classify MNIST digits with less than 20K parameters with test accuracy more than 99.4% along with a CI/CD pipeline to test the model on different test cases.


## Requirements

- Python 3.8
- PyTorch
- torchvision
- pytest
- torchinfo


## Usage

1. Install the required packages.
2. Run `MNIST_S6_99_4_E_20.ipynb` to train the model to achieve more than 99.4% accuracy.
3. Run `pytest -v` to execute tests.


## Network Architecture

| INPUT    | KERNEL     | OUTPUT |
|----------|------------|--------|
| 28x28x1  | 3x3x1x8    | 26     |
| 26x26x8  | 3x3x8x16   | 24     |
| 24x24x16 | 1x1x16x8   | 24     |
| Maxpool()             | 12     |
| 12x12x8  | 3x3x8x16   | 10     |
| 10x10x16 | 3x3x16x16  | 8      |
| 8x8x16   | 3x3x16x16  | 6      |
| AdaptiveAvgPool()     | 1      | 
| 1x1x16   | 1x1x16x10  | 1      |
| Flatten()             |        | 


## RF Calculation

**Formula of RF**: \(r_{out}\) = \(r_{in}\) + (\(k\) - 1) * \(j_{in}\)

**Formula of Jump**: \(j_{out}\) = \(j_{in}\) * \(s\)


| \(n_{in}\) | \(k\) | \(s\) | \(p\) | \(n_{out}\) | \(r_{in}\) | \(j_{in}\) | \(r_{out}\) | \(j_{out}\) |
|------------|-------|-------|-------|-------------|------------|------------|-------------|-------------|
| 28         | 3     | 1     | 0     | 26          | 1          | 1          | 3           | 1           |
| 26         | 3     | 1     | 0     | 24          | 3          | 1          | 5           | 1           |
| 24         | 1     | 1     | 0     | 24          | 5          | 1          | 5           | 1           |
| 24         | 2     | 2     | 0     | 12          | 5          | 1          | 6           | 2           |
| 12         | 3     | 1     | 0     | 10          | 6          | 2          | 10          | 2           |
| 10         | 3     | 1     | 0     | 8           | 10         | 2          | 14          | 2           |
| 8          | 3     | 1     | 0     | 6           | 14         | 2          | 18          | 2           |



## Model Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MnistNet                                 [1, 10]                   --
├─Sequential: 1-1                        [1, 8, 12, 12]            --
│    └─Conv2d: 2-1                       [1, 8, 26, 26]            72
│    └─BatchNorm2d: 2-2                  [1, 8, 26, 26]            16
│    └─Dropout: 2-3                      [1, 8, 26, 26]            --
│    └─ReLU: 2-4                         [1, 8, 26, 26]            --
│    └─Conv2d: 2-5                       [1, 16, 24, 24]           1,152
│    └─BatchNorm2d: 2-6                  [1, 16, 24, 24]           32
│    └─Dropout: 2-7                      [1, 16, 24, 24]           --
│    └─ReLU: 2-8                         [1, 16, 24, 24]           --
│    └─Conv2d: 2-9                       [1, 8, 24, 24]            128
│    └─MaxPool2d: 2-10                   [1, 8, 12, 12]            --
├─Sequential: 1-2                        [1, 16, 6, 6]             --
│    └─Conv2d: 2-11                      [1, 16, 10, 10]           1,152
│    └─BatchNorm2d: 2-12                 [1, 16, 10, 10]           32
│    └─Dropout: 2-13                     [1, 16, 10, 10]           --
│    └─ReLU: 2-14                        [1, 16, 10, 10]           --
│    └─Conv2d: 2-15                      [1, 16, 8, 8]             2,304
│    └─BatchNorm2d: 2-16                 [1, 16, 8, 8]             32
│    └─Dropout: 2-17                     [1, 16, 8, 8]             --
│    └─ReLU: 2-18                        [1, 16, 8, 8]             --
│    └─Conv2d: 2-19                      [1, 16, 6, 6]             2,304
│    └─BatchNorm2d: 2-20                 [1, 16, 6, 6]             32
│    └─Dropout: 2-21                     [1, 16, 6, 6]             --
│    └─ReLU: 2-22                        [1, 16, 6, 6]             --
├─AdaptiveAvgPool2d: 1-3                 [1, 16, 1, 1]             --
├─Sequential: 1-4                        [1, 10, 1, 1]             --
│    └─Conv2d: 2-23                      [1, 10, 1, 1]             160
==========================================================================================
Total params: 7,416
Trainable params: 7,416
Non-trainable params: 0
Total mult-adds (M): 1.13
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.32
Params size (MB): 0.03
Estimated Total Size (MB): 0.35
==========================================================================================
```

## Data Transformations

```python
transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
transforms.RandomAffine(degrees=10,  scale=(0.95, 1.05)),
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
```


## Training Logs

```
Adjusting learning rate of group 0 to 1.5000e-02.
Epoch=0 Batch=468 loss=0.1062707 Accuracy=90.91%: 100%|██████████| 469/469 [00:10<00:00, 44.82it/s]

Test set: Average loss: 0.0004583, Accuracy: 9828/10000 (98.28%)

Adjusting learning rate of group 0 to 1.5000e-02.
Epoch=1 Batch=468 loss=0.0760556 Accuracy=97.00%: 100%|██████████| 469/469 [00:10<00:00, 42.83it/s]

Test set: Average loss: 0.0003724, Accuracy: 9850/10000 (98.50%)

Adjusting learning rate of group 0 to 1.5000e-02.
Epoch=2 Batch=468 loss=0.1601883 Accuracy=97.48%: 100%|██████████| 469/469 [00:10<00:00, 43.56it/s]

Test set: Average loss: 0.0003454, Accuracy: 9861/10000 (98.61%)

Adjusting learning rate of group 0 to 1.5000e-02.
Epoch=3 Batch=468 loss=0.0473514 Accuracy=97.66%: 100%|██████████| 469/469 [00:10<00:00, 45.30it/s]

Test set: Average loss: 0.0003399, Accuracy: 9871/10000 (98.71%)

Adjusting learning rate of group 0 to 1.5000e-02.
Epoch=4 Batch=468 loss=0.1223510 Accuracy=97.88%: 100%|██████████| 469/469 [00:10<00:00, 42.83it/s]

Test set: Average loss: 0.0002233, Accuracy: 9913/10000 (99.13%)

Adjusting learning rate of group 0 to 1.5000e-02.
Epoch=5 Batch=468 loss=0.0516343 Accuracy=97.95%: 100%|██████████| 469/469 [00:10<00:00, 43.90it/s]

Test set: Average loss: 0.0002884, Accuracy: 9890/10000 (98.90%)

Adjusting learning rate of group 0 to 1.5000e-02.
Epoch=6 Batch=468 loss=0.0829413 Accuracy=98.00%: 100%|██████████| 469/469 [00:11<00:00, 42.23it/s]

Test set: Average loss: 0.0002314, Accuracy: 9903/10000 (99.03%)

Adjusting learning rate of group 0 to 6.0000e-03.
Epoch=7 Batch=468 loss=0.0218383 Accuracy=98.44%: 100%|██████████| 469/469 [00:11<00:00, 41.24it/s]

Test set: Average loss: 0.0001749, Accuracy: 9933/10000 (99.33%)

Adjusting learning rate of group 0 to 6.0000e-03.
Epoch=8 Batch=468 loss=0.0519804 Accuracy=98.53%: 100%|██████████| 469/469 [00:10<00:00, 45.39it/s]

Test set: Average loss: 0.0001585, Accuracy: 9939/10000 (99.39%)

Adjusting learning rate of group 0 to 6.0000e-03.
Epoch=9 Batch=468 loss=0.0046923 Accuracy=98.50%: 100%|██████████| 469/469 [00:10<00:00, 44.72it/s]

Test set: Average loss: 0.0001731, Accuracy: 9932/10000 (99.32%)

Adjusting learning rate of group 0 to 6.0000e-03.
Epoch=10 Batch=468 loss=0.0186869 Accuracy=98.50%: 100%|██████████| 469/469 [00:11<00:00, 41.76it/s]

Test set: Average loss: 0.0001478, Accuracy: 9944/10000 (99.44%)

Adjusting learning rate of group 0 to 6.0000e-03.
Epoch=11 Batch=468 loss=0.0358143 Accuracy=98.50%: 100%|██████████| 469/469 [00:10<00:00, 43.57it/s]

Test set: Average loss: 0.0001564, Accuracy: 9944/10000 (99.44%)

Adjusting learning rate of group 0 to 6.0000e-03.
Epoch=12 Batch=468 loss=0.0221422 Accuracy=98.57%: 100%|██████████| 469/469 [00:10<00:00, 42.97it/s]

Test set: Average loss: 0.0002138, Accuracy: 9917/10000 (99.17%)

Adjusting learning rate of group 0 to 6.0000e-03.
Epoch=13 Batch=468 loss=0.0318980 Accuracy=98.53%: 100%|██████████| 469/469 [00:11<00:00, 41.52it/s]

Test set: Average loss: 0.0001625, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 2.4000e-03.
Epoch=14 Batch=468 loss=0.0072799 Accuracy=98.78%: 100%|██████████| 469/469 [00:11<00:00, 42.47it/s]

Test set: Average loss: 0.0001497, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 2.4000e-03.
Epoch=15 Batch=468 loss=0.0469378 Accuracy=98.81%: 100%|██████████| 469/469 [00:11<00:00, 42.37it/s]

Test set: Average loss: 0.0001564, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 2.4000e-03.
Epoch=16 Batch=468 loss=0.0120237 Accuracy=98.75%: 100%|██████████| 469/469 [00:11<00:00, 41.50it/s]

Test set: Average loss: 0.0001616, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 2.4000e-03.
Epoch=17 Batch=468 loss=0.0100550 Accuracy=98.74%: 100%|██████████| 469/469 [00:10<00:00, 43.33it/s]

Test set: Average loss: 0.0001547, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 2.4000e-03.
Epoch=18 Batch=468 loss=0.0358378 Accuracy=98.79%: 100%|██████████| 469/469 [00:10<00:00, 42.89it/s]

Test set: Average loss: 0.0001385, Accuracy: 9944/10000 (99.44%)

Adjusting learning rate of group 0 to 2.4000e-03.
Epoch=19 Batch=468 loss=0.0402809 Accuracy=98.85%: 100%|██████████| 469/469 [00:11<00:00, 41.43it/s]

Test set: Average loss: 0.0001502, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 2.4000e-03.
```


## Tests

The test suite includes various tests to ensure the model's functionality and robustness:

- **Parameter Count Test**: Verifies that the model has fewer than 20,000 parameters.
- **Use of batch normalization**: Ensures the model uses batch normalization.
- **Use of dropout**: Ensures the model uses dropout.
- **Use of fully connected or GAP**: Ensures the model uses a fully connected layer or global average pooling.
