import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def true_function(x):
    return np.sin(x) + 0.1 * x**2

def generate_training_data(seed=42, num_samples=500):
    """
    Generate random training data.
    
    Parameters:
    seed (int): Random seed for reproducibility.
    num_samples (int): Number of training samples.

    Returns:
    x_train (np.ndarray): Input training data.
    y_train (np.ndarray): Output training data.
    """

    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-10, 10, size=(num_samples, 1))
    y_train = true_function(x_train)

    return x_train, y_train

def build_model():
    """
    Builds and returns a neural network model.
    
    Returns:
    model (keras.Model): Neural network model.
    """

    model = keras.Sequential(
    [
        keras.Input(shape=(1,)),
        layers.Dense(64, activation="leaky_relu"),
        layers.Dense(64, activation="leaky_relu"),
        #
        layers.Dense(64, activation="leaky_relu"),
        layers.Dense(1),
    ]
    )   
    
    return model

def train_model(model, x_train, y_train, epochs=500, batch_size=32):
    """
    Trains the model on the training data.
    
    Parameters:
    model (keras.Model): Neural network model.
    x_train (np.ndarray): Input training data.
    y_train (np.ndarray): Output training data.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    history: Training history.
    """

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return history

def evaluate_model(model, x_test):
    """
    Evaluates the model on the test data
    
    Parameters:
    model (keras.Model): Trained neural network model.
    x_test (np.ndarray): Input test data.

    Returns:
    y_pred (np.ndarray): Predicted output
    """
    output = model.predict(x_test)
    return output

def visualize_results(x_train, y_train, x_test, y_pred, filename='results.png'):
    """Visualize the training data and model predictions."""
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, label="Training Data", alpha=0.3)
    plt.plot(x_test, true_function(x_test), label="True Function", color="green", linestyle="dashed")
    plt.plot(x_test, y_pred, label="NN Prediction", color="red")
    plt.legend()
    plt.title("Function Approximation using Neural Network")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.savefig(filename)
    plt.show()


def visualize_history(history, filename='loss.png'):
    """Visualize the loss"""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history['loss']) + 1)

    plt.plot(epochs, history['loss'], label="Training Loss", color="blue")
    plt.plot(epochs, history['val_loss'], label="Validation Loss", color="green")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss during training")
    plt.savefig(filename)
    plt.show()

def predict(model, x_new):
    """
    Predicts the value of a new point.
    
    Parameters:
    model (keras.Model): Trained neural network model.
    x_new (float): New input value.

    Returns:
    predicted_value (float): Predicted output value.
    """

    x_new_array = np.array([[x_new]])
    predicted_value = model.predict(x_new_array)
    
    return predicted_value[0][0]

# Main Execution
if __name__ == "__main__":
    # Step 1: Generate Data
    x_train, y_train = generate_training_data()
    x_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    
    # Step 2: Build and Train Model
    model = build_model()
    history = train_model(model, x_train, y_train)
    
    # Step 3: Evaluate Model
    y_pred = evaluate_model(model, x_test)

    
    # Step 4: Visualize Results
    visualize_results(x_train, y_train, x_test, y_pred, filename='results_64-64-64.png')
    visualize_history(history.history, filename='loss_64-64-64.png')
    
    # Step 5: Predict a new value
    x_new = 7
    y_new_pred = predict(model, x_new)
    print(f"Predicted value at x={x_new}: {y_new_pred:.4f}")
