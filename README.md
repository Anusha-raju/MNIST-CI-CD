[![Test Model](https://github.com/Anusha-raju/MNIST-CI-CD/actions/workflows/test.yml/badge.svg)](https://github.com/Anusha-raju/MNIST-CI-CD/actions/workflows/test.yml)
# Model Testing and Evaluation

This repository contains a PyTorch model and a testing pipeline to evaluate its architecture, performance, and correctness. The goal is to validate the model based on certain criteria such as parameter count, the use of batch normalization (BN), dropout, and more.

## Project Structure

```plaintext
.
├── model.pth               # Pre-trained model weights

├── MNIST_model.ipynb       # Python notebook file containing model architecture

├── test_model.py           # Python test file that checks model architecture

├── .github/                # GitHub Actions workflow directory

│   └── workflows/

│       └── test.yml        # GitHub Actions workflow for automated testing

├── requirements.txt        # List of dependencies

└── README.md               # This file

```
## Dataset Information

This model is trained on the MNIST dataset, which is a classic dataset used for image classification tasks.

## Dataset Overview
*Name*: MNIST (Modified National Institute of Standards and Technology)<br>
*Type*: Image classification<br>
*Number of Classes*: 10 (digits 0-9)<br>
*Number of Samples*: 60,000 training images and 10,000 test images<br>
*Input Shape*: 28x28 pixels, grayscale images (1 channel)<br>
#### Dataset Details<br>
*Training Set*: 60,000 28x28 grayscale images of handwritten digits.<br>
*Test Set*: 10,000 28x28 grayscale images for evaluation.<br>
The dataset is used for training the model to classify digits from 0 to 9.<br>
<br><br>
To install the dependencies from requirements.txt:
```
pip install -r requirements.txt
```

To run the tests locally:
```
python test_model.py
```

This will run the test script that validates the following criteria:

Model parameter count (should be less than 20k parameters)<br>
Presence of Batch Normalization layers<br>
Presence of Dropout layers<br>
Presence of Global Average Pooling (GAP) or Fully Connected (FC) layer<br>

The train and test log:

```
EPOCH:  1
loss=0.10297595709562302 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.65it/s]

Test set: Average loss: 0.1136, Accuracy: 9781/10000 (97.8100%)

EPOCH:  2
loss=0.11357773095369339 batch_id=468: 100%|██████████| 469/469 [01:38<00:00,  4.75it/s]

Test set: Average loss: 0.2572, Accuracy: 9242/10000 (92.4200%)

EPOCH:  3
loss=0.042977988719940186 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.69it/s]

Test set: Average loss: 0.0491, Accuracy: 9882/10000 (98.8200%)

EPOCH:  4
loss=0.09455662220716476 batch_id=468: 100%|██████████| 469/469 [01:39<00:00,  4.71it/s]

Test set: Average loss: 0.0639, Accuracy: 9829/10000 (98.2900%)

EPOCH:  5
loss=0.0995449498295784 batch_id=468: 100%|██████████| 469/469 [01:41<00:00,  4.63it/s]

Test set: Average loss: 0.0335, Accuracy: 9902/10000 (99.0200%)

EPOCH:  6
loss=0.03818140923976898 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.69it/s]

Test set: Average loss: 0.0354, Accuracy: 9913/10000 (99.1300%)

EPOCH:  7
loss=0.018259121105074883 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.66it/s]

Test set: Average loss: 0.0285, Accuracy: 9921/10000 (99.2100%)

EPOCH:  8
loss=0.04147778078913689 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.69it/s]

Test set: Average loss: 0.0344, Accuracy: 9902/10000 (99.0200%)

EPOCH:  9
loss=0.0371086560189724 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.68it/s]

Test set: Average loss: 0.0258, Accuracy: 9921/10000 (99.2100%)

EPOCH:  10
loss=0.04877251759171486 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.68it/s]

Test set: Average loss: 0.0245, Accuracy: 9929/10000 (99.2900%)

EPOCH:  11
loss=0.033225953578948975 batch_id=468: 100%|██████████| 469/469 [01:39<00:00,  4.69it/s]

Test set: Average loss: 0.0284, Accuracy: 9913/10000 (99.1300%)

EPOCH:  12
loss=0.026190033182501793 batch_id=468: 100%|██████████| 469/469 [01:40<00:00,  4.66it/s]

Test set: Average loss: 0.0252, Accuracy: 9934/10000 (99.3400%)

EPOCH:  13
loss=0.03852536156773567 batch_id=468: 100%|██████████| 469/469 [01:39<00:00,  4.71it/s]

Test set: Average loss: 0.0206, Accuracy: 9941/10000 (99.4100%)
```
**Total number of epochs taken are 13.**
**Total number of parameters are 18,960.**

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- TensorFlow and Keras for providing the deep learning framework.
