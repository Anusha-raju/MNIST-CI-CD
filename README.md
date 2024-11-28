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
loss=0.23467735946178436 batch_id=468: 100%|██████████| 469/469 [00:59<00:00,  7.89it/s]

Test set: Average loss: 0.1303, Accuracy: 9751/10000 (97.5100%)

EPOCH:  2
loss=0.21513687074184418 batch_id=468: 100%|██████████| 469/469 [00:58<00:00,  8.08it/s]

Test set: Average loss: 0.0851, Accuracy: 9808/10000 (98.0800%)

EPOCH:  3
loss=0.06709787249565125 batch_id=468: 100%|██████████| 469/469 [00:57<00:00,  8.13it/s]

Test set: Average loss: 0.0651, Accuracy: 9852/10000 (98.5200%)

EPOCH:  4
loss=0.06787603348493576 batch_id=468: 100%|██████████| 469/469 [00:58<00:00,  8.05it/s]

Test set: Average loss: 0.0541, Accuracy: 9855/10000 (98.5500%)

EPOCH:  5
loss=0.045409634709358215 batch_id=468: 100%|██████████| 469/469 [00:58<00:00,  8.02it/s]

Test set: Average loss: 0.0574, Accuracy: 9848/10000 (98.4800%)

EPOCH:  6
loss=0.10505013912916183 batch_id=468: 100%|██████████| 469/469 [00:58<00:00,  8.08it/s]

Test set: Average loss: 0.0487, Accuracy: 9866/10000 (98.6600%)

EPOCH:  7
loss=0.08098893612623215 batch_id=468: 100%|██████████| 469/469 [00:57<00:00,  8.18it/s]

Test set: Average loss: 0.0387, Accuracy: 9891/10000 (98.9100%)

EPOCH:  8
loss=0.04133078083395958 batch_id=468: 100%|██████████| 469/469 [00:59<00:00,  7.85it/s]

Test set: Average loss: 0.0375, Accuracy: 9892/10000 (98.9200%)

EPOCH:  9
loss=0.07203114777803421 batch_id=468: 100%|██████████| 469/469 [00:59<00:00,  7.93it/s]

Test set: Average loss: 0.0397, Accuracy: 9886/10000 (98.8600%)
```
**Total number of epochs taken are 9.**

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- TensorFlow and Keras for providing the deep learning framework.
