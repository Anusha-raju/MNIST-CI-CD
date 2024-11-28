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
loss=0.11340221762657166 batch_id=468: 100%|██████████| 469/469 [04:24<00:00,  1.77it/s]

Test set: Average loss: 0.0927, Accuracy: 9849/10000 (98.4900%)

EPOCH:  2
loss=0.12052300572395325 batch_id=468: 100%|██████████| 469/469 [04:25<00:00,  1.77it/s]

Test set: Average loss: 0.0374, Accuracy: 9923/10000 (99.2300%)

EPOCH:  3
loss=0.09099370986223221 batch_id=468: 100%|██████████| 469/469 [04:23<00:00,  1.78it/s]

Test set: Average loss: 0.0442, Accuracy: 9895/10000 (98.9500%)

EPOCH:  4
loss=0.0256239902228117 batch_id=468: 100%|██████████| 469/469 [04:22<00:00,  1.79it/s]

Test set: Average loss: 0.0341, Accuracy: 9933/10000 (99.3300%)

EPOCH:  5
loss=0.03155975416302681 batch_id=468: 100%|██████████| 469/469 [04:24<00:00,  1.77it/s]

Test set: Average loss: 0.0268, Accuracy: 9938/10000 (99.3800%)

EPOCH:  6
loss=0.05659667029976845 batch_id=468: 100%|██████████| 469/469 [04:23<00:00,  1.78it/s]

Test set: Average loss: 0.0228, Accuracy: 9951/10000 (99.5100%)

EPOCH:  7
loss=0.06775494664907455 batch_id=468: 100%|██████████| 469/469 [04:25<00:00,  1.77it/s]

Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.4000%)

EPOCH:  8
loss=0.013504721224308014 batch_id=468: 100%|██████████| 469/469 [04:24<00:00,  1.78it/s]

Test set: Average loss: 0.0201, Accuracy: 9941/10000 (99.4100%)

EPOCH:  9
loss=0.008742149919271469 batch_id=468: 100%|██████████| 469/469 [04:24<00:00,  1.77it/s]

Test set: Average loss: 0.0172, Accuracy: 9956/10000 (99.5600%)
```


## Acknowledgments

- The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- TensorFlow and Keras for providing the deep learning framework.