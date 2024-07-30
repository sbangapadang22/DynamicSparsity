import unittest
import tensorflow as tf
from src.scheduler.tf_sparsity_scheduler import SparsityScheduler
from src.layers.sparse_dense import SparseDense

class TestTFSparsityScheduler(unittest.TestCase):

    def setUp(self):
        # Create a simple model with sparse layers
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            SparseDense(128),
            tf.keras.layers.ReLU(),
            SparseDense(10)
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.initial_sparsity = 0.5
        self.final_sparsity = 0.1
        self.total_epochs = 100

    def test_linear_sparsity_scheduler(self):
        # Initialize the scheduler with linear criteria
        scheduler = SparsityScheduler(initial_sparsity=self.initial_sparsity, 
                                      final_sparsity=self.final_sparsity, 
                                      total_epochs=self.total_epochs, 
                                      criteria='linear')

        # Fit the model for one epoch to initialize the scheduler
        self.model.fit(tf.random.normal((10, 784)), tf.random.uniform((10,), maxval=10, dtype=tf.int32), epochs=1, callbacks=[scheduler])

        # Simulate epochs manually
        for epoch in range(self.total_epochs):
            scheduler.on_epoch_begin(epoch)
            expected_sparsity = self.initial_sparsity - (self.initial_sparsity - self.final_sparsity) * (epoch / self.total_epochs)
            for layer in self.model.layers:
                if hasattr(layer, 'sparsity_level'):
                    self.assertAlmostEqual(layer.sparsity_level, expected_sparsity, places=5)

    def test_weight_magnitude_sparsity_scheduler(self):
        # Initialize the scheduler with weight magnitude criteria
        scheduler = SparsityScheduler(initial_sparsity=self.initial_sparsity, 
                                      final_sparsity=self.final_sparsity, 
                                      total_epochs=self.total_epochs, 
                                      criteria='weight_magnitude', 
                                      weight_threshold=0.1)

        # Fit the model for one epoch to initialize the scheduler
        self.model.fit(tf.random.normal((10, 784)), tf.random.uniform((10,), maxval=10, dtype=tf.int32), epochs=1, callbacks=[scheduler])

        # Simulate epochs manually
        for epoch in range(self.total_epochs):
            scheduler.on_epoch_begin(epoch)
            for layer in self.model.layers:
                if hasattr(layer, 'sparsity_level'):
                    self.assertTrue(0 <= layer.sparsity_level <= 1)  # Sparsity should be between 0 and 1

if __name__ == '__main__':
    unittest.main()
