# 7/29, new implementation of criteria overall test

import unittest
import torch
import torch.nn as nn
from src.scheduler.torch_sparsity_scheduler import SparsityScheduler
from src.layers.sparse_linear import SparseLinear

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = SparseLinear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = SparseLinear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TestTorchSparsityScheduler(unittest.TestCase):

    def setUp(self):
        self.model = SimpleModel()
        self.initial_sparsity = 0.5
        self.final_sparsity = 0.1
        self.total_epochs = 100

    def test_linear_sparsity_scheduler(self):
        scheduler = SparsityScheduler(model=self.model, 
                                      initial_sparsity=self.initial_sparsity, 
                                      final_sparsity=self.final_sparsity, 
                                      total_epochs=self.total_epochs, 
                                      criteria='linear')

        for epoch in range(self.total_epochs):
            scheduler.adjust_sparsity(epoch)
            expected_sparsity = self.initial_sparsity - (self.initial_sparsity - self.final_sparsity) * (epoch / self.total_epochs)
            for layer in self.model.modules():
                if isinstance(layer, SparseLinear):
                    self.assertAlmostEqual(layer.sparsity_level, expected_sparsity, places=5)

    def test_weight_magnitude_sparsity_scheduler(self):
        scheduler = SparsityScheduler(model=self.model, 
                                      initial_sparsity=self.initial_sparsity, 
                                      final_sparsity=self.final_sparsity, 
                                      total_epochs=self.total_epochs, 
                                      criteria='weight_magnitude', 
                                      weight_threshold=0.1)

        for epoch in range(self.total_epochs):
            scheduler.adjust_sparsity(epoch)
            for layer in self.model.modules():
                if isinstance(layer, SparseLinear):
                    self.assertTrue(0 <= layer.sparsity_level <= 1)

if __name__ == '__main__':
    unittest.main()
