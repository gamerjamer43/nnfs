# Neural Network From Scratch
this project implements a simple neural network using **only numpy**. 
that's right, every other import is used solely for displaying or formatting images. 
this means the core of the neural network, from the dense layer, to forward and backpropagation and parameter updates, is built entirely with numpy, made in an attempt to how neural networks operate under the hood without using any already existent high-level frameworks.

## Overview
this neural network is designed to recognize handwritten digits from the mnist dataset. it is a two-layer (one hidden layer and one output layer) fully connected network:
- **input layer:** accepts a flattened 28x28 (784-element) image.
- **hidden layer:** consists of 128 neurons using the sigmoid activation function.
- **output layer:** consists of 10 neurons (one for each digit 0-9) with a softmax activation for generating probability distributions.

## How it Works

1. **forward propagation:**
   - **layer 1 (dense):** the input vector is multiplied by the weights of the first dense layer, added to the biases, and then passed through our sigmoid activation function. this non-linearity allows the network to learn complex patterns.
   - **layer 2 (dense):** the output from the hidden layer is passed to the second dense layer. no activation is applied here since this layer works directly into the softmax function.
   - **softmax activation:** the raw output (or logits) from the final dense layer is transformed into a probability table that sums to 1. the softmax function is implemented with numerical stability in mind (by subtracting the max value per sample).

2. **loss calculation:**
   - **cross-entropy loss:** the network uses cross-entropy loss to measure the difference between the predicted probability distribution and the true distribution.

3. **backward propagation:**
   - the derivative of the loss with respect to the output is computed, and then gradients are backpropagated through the network.
   - for the sigmoid activation, the derivative is calculated to correctly adjust the weights during backpropagation.
   - the network parameters (weights and biases) are updated using gradient descent.

4. **training:**
   - the `Trainer` class handles the entire training process. it loads and normalizes the data, shuffles the data per epoch, and performs mini-batch gradient descent to update the network parameters.
   - after each epoch, the training loss and accuracy are printed to track progress.

5. **model persistence and inference:**
   - after training, the model parameters (weights and biases) are saved into a file (`model/model.npz`).
   - the `NeuralApp` class loads these parameters for inference. it allows for prediction on new images—either randomly selected from the mnist csv dataset or loaded from a saved image file.

## File Structure

- **`model.py`:**  
  contains the implementation of activation functions (`sigmoid`, `dsigmoid`, `softmax`), the cross-entropy loss function (`centropy`), and the `Dense` class, the fully connected layer with forward and backward passes.

- **`trainer.py`:**  
  implements the training routine. loads the mnist dataset from the csv provided, constructs the network, and trains it using mini-batch gradient descent. also prints out the loss and accuracy after each epoch and saves the trained model parameters.

- **`test.py`:**  
  provides a simple application for inference. it loads our trained model, reads an image (either a random mnist sample or a user-saved image if one exists), and shows the user the image, then uses the model to predict the digit.

## How to Run

1. **training the model:**  
   run `trainer.py` to train the network on the mnist dataset. the training process will shuffle the data, run through a specified number of epochs, and print training metrics. after training, the model parameters are saved to `model/model.npz`.
   i have found that 250 epochs almost guarantees 100% accuracy on the training set.

2. **testing the model:**  
   run `test.py` to load the saved model and perform a prediction. if an `image.png` exists, it will be used; otherwise, a random sample from the mnist dataset will be used. the digit is then shown to the user in an black/white activation matrix and predicted.

## Conclusion

this project serves as a practical demonstration of how a neural net can be implemented from scratch using only numpy. by building each component manually, the dense layers, the activation functions and the back and forward propagation algorithms, i gained a deeper understanding of the inner workings of neural networks without the abstraction provided by high-level frameworks.