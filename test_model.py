import torch
import unittest
from torch import nn
from torchsummary import summary

# Assuming 'Net' is the model you defined earlier
from model_architecture import Net  # Replace with the correct import path

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Initialize model and load pre-trained weights (if available)
        self.device = torch.device("cpu")
        self.model = Net().to(self.device)
        

# Load the model weights
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()


   



    def test_param_count(self):
        # Test if the model has less than 20k parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 20000, f"Model has {total_params} parameters, which exceeds 20k.")

    
    def test_batch_normalization(self):
        # Test if the model uses Batch Normalization
        bn_layers = [module for module in self.model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertGreater(len(bn_layers), 0, "Model does not use Batch Normalization.")
    
    def test_dropout(self):
        # Test if the model uses Dropout
        dropout_layers = [module for module in self.model.modules() if isinstance(module, nn.Dropout)]
        self.assertGreater(len(dropout_layers), 0, "Model does not use Dropout.")
    
    def test_gap_or_fc(self):
        # Test if the model uses either Global Average Pooling (GAP) or Fully Connected layer
        has_gap = any(isinstance(layer, nn.AdaptiveAvgPool2d) for layer in self.model.modules())
        has_fc = any(isinstance(layer, nn.Linear) for layer in self.model.modules())
        
        self.assertTrue(has_gap or has_fc, "Model does not use GAP or Fully Connected layer.")
    
    def test_model_summary(self):
        # This test is to generate a model summary and check if the architecture looks correct
        # You can adjust the input size based on your data (e.g., 1x28x28 for MNIST)
        try:
            summary(self.model, input_size=(1, 28, 28))
        except Exception as e:
            self.fail(f"Failed to generate model summary: {str(e)}")

if __name__ == "__main__":
    unittest.main()
