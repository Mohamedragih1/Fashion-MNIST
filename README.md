# Fashion MNIST Multiclass Classification with Keras

## Objective
This project builds a multiclass classification model using Keras with a TensorFlow backend to classify fashion items from the Fashion MNIST dataset. The model's goal is to achieve high accuracy, targeting a minimum of 90% on the test set.

## Dataset
The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset consists of 28x28 grayscale images of various clothing items, categorized into 10 classes:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset includes:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

## Model Architecture
The model is a deep Artificial Neural Network (ANN) created with the Keras Sequential API. The architecture includes:

- **Input Layer**: Accepts a 784-dimensional flattened vector (28x28 pixels).
- **Hidden Layers**: Five fully connected layers with ReLU activation, batch normalization, and dropout for regularization:
  - 1024 neurons with 20% dropout
  - 512 neurons with 20% dropout
  - 264 neurons with 20% dropout
  - 128 neurons with 20% dropout
  - 64 neurons with 20% dropout
  - 32 neurons with 20% dropout
- **Output Layer**: 10 neurons with softmax activation to output probabilities for each class.

### Model Compilation
The model is compiled with:
- **Optimizer**: Adam
- **Loss Function**: `sparse_categorical_crossentropy` (for multiclass classification)
- **Metric**: Accuracy

### Learning Rate Scheduler
A learning rate scheduler gradually reduces the learning rate:
```python
def scheduler(epoch, lr):
    if epoch > 50:
        return lr * 0.8
    return lr

lr_scheduler = LearningRateScheduler(scheduler)
