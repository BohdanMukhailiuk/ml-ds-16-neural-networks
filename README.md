# ML-DS Sprint 16 - Intro to Neural Networks

## Task 1. Forecasting Function Value at a Given Point

### Task Description:
Develop and train neural network using TensorFlow/Keras to approximate the function: 

`𝑓(𝑥) = sin(x) + 0.1 x²`

After training, the model should be able to predict `𝑓(𝑥)` for any given `𝑥` within a specified range.

### Steps:
**1. Define the Function:**

- Select a mathematical function `𝑓(𝑥) = sin(x) + 0.1 x²`
- This function will serve as the ground truth for training data.

**2. Generate Training Data:**

- Sample `𝑥` values from a range (e.g.,  `𝑥 ∈ [ − 10 , 10 ] x∈[−10,10]`).
- Compute the corresponding `𝑦` values using the function: `𝑦 = 𝑓 ( 𝑥 )`
- _Optionally, add some noise to simulate real-world data._

**3. Create a Neural Network Model:**

- Use a simple feedforward neural network (MLP). 
- Example architecture: 
  - Input: `𝑥` (single value)
  - Hidden layers: 2-3 layers with ReLU activation
  - Output: `𝑦` (predicted function value)

**4. Train the Model:**

- Use mean squared error (MSE) as the loss function. 
- Use an optimizer like Adam. 
- Train on the generated dataset.

**5. Evaluate the Model:**

- Split data into training and validation sets.
- Check performance on unseen points.

**6. Make Predictions:**

- Given a new `𝑥` , predict  `𝑓(𝑥)` using the trained network. 
- Compare with the true function value.

## Task 2. Predicting House Prices

### Objective:
Train a neural network to predict house prices based on various input features using TensorFlow/Keras.

### Task Description:
You are given a dataset containing information about different houses, including their features (e.g., size, number of bedrooms, location) and their actual prices. Your task is to:

- Preprocess the data (handle missing values, normalize numerical values, encode categorical variables).
- Train a neural network to predict house prices.
- Evaluate the model using appropriate metrics (e.g., Mean Squared Error, R² score).
- Optimize the model by tuning hyperparameters and testing different architectures.

### Dataset:
You can use the California Housing Prices dataset (available in Scikit-learn). Alternatively, generate synthetic data with predefined patterns.

### Steps:

**1. Load and Explore Data**
- Load the dataset into a Pandas DataFrame.
- Display basic statistics and visualize correlations.

**2. Data Preprocessing**
Handle missing values. 
Normalize numerical features. 
Encode categorical variables (one-hot encoding or label encoding). 
Split the dataset into training and test sets. 

**3. Build the Neural Network** 
- Use TensorFlow/Keras to create a regression model.
- Suggested architecture:
  - Input layer: Accepts all features.
  - Hidden layers: Use at least two fully connected (Dense) layers with activation functions (ReLU).
  - Output layer: A single neuron with a linear activation function (since it’s a regression problem).

**4. Train and Evaluate the Model**
- Choose a loss function (e.g., Mean Squared Error).
- Choose an optimizer (e.g., Adam).
- Train the model and track the loss.
- Evaluate using test data.

**5. Optimize the Model**
- Experiment with different architectures (more layers, dropout, batch normalization).
- Tune hyperparameters (learning rate, batch size, epochs).

**6. Test with New Data**
- Provide a few unseen house samples and check the predicted prices.