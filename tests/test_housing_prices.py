import unittest
import numpy as np
from src.housing_prices import (load_data, preprocess_data, build_model, train_model, predict_house_price)

class TestHousePriceModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.scaler = preprocess_data(cls.df)
        cls.model = build_model(cls.X_train.shape[1])
        cls.history = train_model(cls.model, cls.X_train, cls.y_train, cls.X_test, cls.y_test)

    def test_load_data(self):
        """Test if dataset is loaded correctly."""
        self.assertFalse(self.df.empty, "Dataset should not be empty.")
        self.assertIn('Price', self.df.columns, "Dataset should contain 'Price' column.")

    def test_preprocess_data(self):
        """Test data preprocessing."""
        self.assertEqual(self.X_train.shape[1], self.X_test.shape[1], "Train and test sets should have same feature dimensions.")
        self.assertEqual(len(self.X_train), len(self.y_train), "Number of training samples should match labels.")
        self.assertEqual(len(self.X_test), len(self.y_test), "Number of test samples should match labels.")
        self.assertAlmostEqual(np.mean(self.X_train), 0, delta=0.1, msg="Training data should be standardized.")

    def test_build_model(self):
        """Test model structure."""
        self.assertEqual(self.model.output_shape, (None, 1), "Output layer should have 1 neuron for regression.")

    def test_price_prediction(self):
        """Test house price prediction."""
        new_house = [8.32, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]  # Example house data
        predicted_price = predict_house_price(self.model, new_house, self.scaler)
        self.assertTrue(4 <= predicted_price <= 5, "Predicted price should be between $400,000 and $500,000.")

if __name__ == '__main__':
    unittest.main()
