import unittest
from src.value_of_point import (generate_training_data, build_model, train_model, predict)

# Unit Tests
class TestPointValueModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_train, cls.y_train = generate_training_data()
        cls.model = build_model()
        train_model(cls.model, cls.x_train, cls.y_train)
    
    def test_data_generation(self):
        """Test if data is generated correctly."""
        x_train, y_train = generate_training_data()
        self.assertEqual(x_train.shape, (500, 1), "Input training data should have shape (500, 1).")
        self.assertEqual(y_train.shape, (500, 1), "Output training data should have shape (500, 1).")
    
    def test_model_structure(self):
        """Test the model architecture."""
        self.assertEqual(self.model.input_shape, (None, 1), "Model input shape should be (None, 1).")
    
    def test_prediction_value_of_5(self):
        """Test prediction for x=5."""
        y_pred = predict(self.model, 5)
        self.assertTrue(1.0 <= y_pred <= 2.0, "Predicted value of 5 must be between 1.0 and 2.0")

if __name__ == "__main__":
    unittest.main()
