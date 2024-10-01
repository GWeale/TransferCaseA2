import unittest
import numpy as np
from src.model import NeuralNetwork
from src.transformer import Transformer

class TestNeuralNetwork(unittest.TestCase):
    def test_forward(self):
        nn = NeuralNetwork(input_size=10, hidden_sizes=[5], output_size=3)
        X = np.random.randn(2,10)
        activations = nn.forward(X)
        self.assertEqual(activations[-1].shape, (2,3))
    
    def test_backward(self):
        nn = NeuralNetwork(input_size=10, hidden_sizes=[5], output_size=3)
        X = np.random.randn(2,10)
        Y = np.eye(3)[np.array([0,1])]
        activations = nn.forward(X)
        grads = nn.backward(activations, Y)
        self.assertEqual(len(grads), 2)
        self.assertEqual(grads[0]['dW'].shape, (5,3))
        self.assertEqual(grads[0]['db'].shape, (1,3))
    
class TestTransformer(unittest.TestCase):
    def test_transformer_forward(self):
        transformer = Transformer(num_layers=2, d_model=64, num_heads=8, d_ff=256, input_vocab_size=1000, target_vocab_size=10)
        x = np.random.randn(2, 10, 64)
        output = transformer.forward(x)
        self.assertEqual(output.shape, (2, 10, 10))

if __name__ == '__main__':
    unittest.main()
