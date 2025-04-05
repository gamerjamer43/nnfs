# image loading/display
import matplotlib.pyplot as plt
from PIL import Image
import os

# colored prints
from rich import print

# numerical operations
import numpy as np

# get dense and softmax (the layer and activation function) from model.py
from model import Dense, softmax

class NeuralApp:
    """
    loads the trained model and provides a method to predict a digit from an image.
    """
    def __init__(self, model_path='model/model.npz', input_dim=784, hidden_dim=128, output_dim=10):
        data = np.load(model_path)
        # build layers with the same architecture used during training, and load saved weights and biases
        self.layer1 = Dense(input_dim, hidden_dim, activation='sigmoid')
        self.layer2 = Dense(hidden_dim, output_dim=10, activation=None)
        self.layer1.W = data['W1']
        self.layer1.b = data['b1']
        self.layer2.W = data['W2']
        self.layer2.b = data['b2']

    def predict(self, image):
        """
        predicts the digit from a single input image (flattened 784-element vector).
        """
        a1 = self.layer1.forward(image.reshape(1, -1))
        z2 = self.layer2.forward(a1)
        out = softmax(z2)
        return int(np.argmax(out, axis=1)[0])

def get_random_mnist_sample(csv_path='datasets/mnist_train.csv', index=None, save_path='image.png'):
    """
    loads an image from the CSV dataset, normalizes it,
    saves it as an image file, and returns the image array with the label.
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if index is None:
        index = np.random.randint(0, data.shape[0])  # select a random sample
    
    label = int(data[index, 0])   # first column is the digit label
    image = data[index, 1:].reshape(28, 28)
    
    # normalize pixel values (0-255 → 0-1)
    image_normalized = image / 255.0  

    img_pil = Image.fromarray((image * 255).astype(np.uint8))  # convert back to 0-255 grayscale
    img_pil.save(save_path)
    print(f"Image saved as {save_path}")

    return image_normalized.flatten(), label

def load_saved_image(image_path='image.png'):
    """
    loads an existing image, processes it, and returns a flattened vector.
    """
    img = Image.open(image_path).convert('L')  # convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0  # normalize
    return img_array.flatten()

if __name__ == '__main__':
    if os.path.exists('image.png'):
        print("[green]Using saved image.png for prediction.[/green]")
        image_flat = load_saved_image()
        true_label = None  # unknown since it's a user-provided image
    else:
        print("[yellow]No saved image found. Using a random MNIST sample.[/yellow]")
        image_flat, true_label = get_random_mnist_sample()

    # show the selected digit
    plt.imshow(image_flat.reshape(28, 28), cmap='gray')
    if true_label is not None:
        plt.title(f"[green]Actual: {true_label}[/green]")
    plt.show()
    
    # run prediction
    app = NeuralApp()
    prediction = app.predict(image_flat)
    
    if true_label is not None:
        print(f"[blue]Predicted Digit: {prediction} | Actual Digit: {true_label}[/blue]")
    else:
        print(f"[blue]Predicted Digit: {prediction}[/blue]")