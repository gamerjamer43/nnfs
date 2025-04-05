# a kinda accurate pretty cool neural network from scratch
import numpy as np

def sigmoid(x):
    """### sigmoid activation
    x: input array"""
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    """## derivative of the sigmoid function
    x: input array"""
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    """## softmax activation with numerical stability
    x: input array (shape: (m, num_classes))"""
    # subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def centropy(y_true, y_pred):
    """
    computes cross-entropy loss.
    y_true: integer labels (shape: (m,))
    y_pred: predicted probabilities (shape: (m, num_classes))
    """
    m = y_true.shape[0]
    log_probs = -np.log(y_pred[np.arange(m), y_true] + 1e-9)
    return np.sum(log_probs) / m

class Dense:
    """
    a fully connected (dense) layer.
    
    attributes:
      W : weights matrix
      b : bias vector
      activation : activation function ('sigmoid' or None for linear)
    """
    def __init__(self, input_dim, output_dim, activation=None):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation = activation

    def forward(self, x):
        """forward pass; saves input and linear output for backprop."""
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        if self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        else:
            self.a = self.z  # linear activation
        return self.a

    def backward(self, grad_output):
        """
        backward pass.
        grad_output: gradient of loss with respect to this layer's output.
        returns gradient with respect to the layer input.
        """
        if self.activation == 'sigmoid':
            grad_z = grad_output * dsigmoid(self.z)
        else:
            grad_z = grad_output  # derivative of linear function is 1
        self.dW = np.dot(self.x.T, grad_z)
        self.db = np.sum(grad_z, axis=0, keepdims=True)
        return np.dot(grad_z, self.W.T)

    def update(self, lr):
        """updates weights and biases using gradient descent."""
        self.W -= lr * self.dW
        self.b -= lr * self.db