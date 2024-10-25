# Fashion MNIST Multiclass Classification with Keras

## Objective
This project aims to build a multiclass classification model using Keras with a TensorFlow backend to classify fashion items from the Fashion MNIST dataset. The goal is to achieve a minimum accuracy of 90% on the test set by implementing a well-regularized Artificial Neural Network (ANN) and visualizing the model's performance.

## Dataset
The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset contains 28x28 grayscale images of fashion items, organized into 10 categories. Each image is labeled with one of the following classes:
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

The dataset consists of:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

## Steps
1. **Data Loading and Preprocessing**:
   - Load the Fashion MNIST dataset from `keras.datasets`.
   - Split the dataset into training and testing sets.
   - Normalize images to scale pixel values between 0 and 1.
   - Flatten each 28x28 image into a 1D vector of 784 features for input to the ANN.

2. **Data Visualization**:
   - Visualize sample images along with their labels to understand the dataset.

3. **Model Architecture**:
   - Used the Keras Sequential API to build the model with the following layers:
      - An **input layer** to accept the 784-dimensional flattened image vector.
      - **3 hidden layers** with a reasonable number of neurons and ReLU activation.
      - An **output layer** with 10 neurons (one per class) and softmax activation for probability output.
   - Added regularization (e.g., Dropout, Batch Normalization) to prevent overfitting.
   - Used a learning rate scheduler.

4. **Compilation**:
   - Defined the loss function as `sparse_categorical_crossentropy` for multiclass classification.
   - Used the Adam optimizer.
   - Tracked accuracy as the evaluation metric during training.

5. **Model Training**:
   - Train the model on the training data, using an 80-20 split for training and validation.
   - Monitor validation metrics to detect overfitting.

6. **Evaluation**:
   - Evaluate the model on the test set to report the test accuracy.
   - Generate a classification report (precision, recall, F1-score).
   - Plot the confusion matrix and a bar graph of the classification report metrics.
   - Visualize the training and validation accuracy and loss curves.

