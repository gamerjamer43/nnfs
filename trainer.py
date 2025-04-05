# the big boy, needed for almost every operation
import numpy as np

# other imports from model, dense (a full dense layer), centropy (cross entropy), softmax (activation function)
from model import Dense, centropy, softmax

def load_data(file_path):
    """
    loads and normalizes data from CSV.
    CSV format: label, pixel1, pixel2, ..., pixel784
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X = data[:, 1:] / 255.0  # normalize pixel values to [0,1]
    y = data[:, 0].astype(int)
    return X, y

class Trainer:
    """
    builds a two-layer network, trains it, and saves the model.
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        self.layer1 = Dense(input_dim, hidden_dim, activation='sigmoid')
        self.layer2 = Dense(hidden_dim, output_dim, activation=None)

    def forward(self, X) -> np.ndarray:
        """Forward propagation through both layers."""
        a1 = self.layer1.forward(X)
        z2 = self.layer2.forward(a1)
        self.out = softmax(z2)
        return self.out

    def backward(self, X, y, lr) -> None:
        """
        backprop and parameter updates.
        """
        m = X.shape[0]
        # For softmax with cross-entropy, gradient simplifies to:
        grad_out = self.out.copy()
        grad_out[np.arange(m), y] -= 1
        grad_out /= m

        grad_hidden = self.layer2.backward(grad_out)
        self.layer1.backward(grad_hidden)

        # Update weights and biases
        self.layer2.update(lr)
        self.layer1.update(lr)

    def train(self, X, y, epochs=50, lr=0.1, batch_size=64) -> None:
        """
        trains network using mini-batch gradient descent and prints loss and accuracy.
        """
        n = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(n)
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            # Mini-batch training
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.forward(X_batch)
                loss = centropy(y_batch, self.out)
                self.backward(X_batch, y_batch, lr)
            
            # Compute full training loss and accuracy
            output = self.forward(X)
            loss = centropy(y, output)
            preds = np.argmax(output, axis=1)
            accuracy = np.mean(preds == y)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy*100:.2f}%")

    def save_model(self, file_path='model/model.npz') -> None:
        """
        saves the model parameters to a file.
        """
        np.savez(file_path, 
                 W1=self.layer1.W, b1=self.layer1.b,
                 W2=self.layer2.W, b2=self.layer2.b)
        
def main() -> None:
    # load csv as X, y
    data_file = 'datasets/mnist_train.csv'
    print("Loading data...")
    X, y = load_data(data_file)
    print(f"Data loaded. Samples: {X.shape[0]}")
    
    # load trainer and params
    trainer = Trainer()
    print("Training network...")
    trainer.train(X, y, epochs=200, lr=0.1, batch_size=64)
    print("Training complete.")

    # save model to model.npz
    trainer.save_model()
    print("Model saved as model.npz")

if __name__ == '__main__':
    main()