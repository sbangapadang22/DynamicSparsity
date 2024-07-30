import unittest
import tensorflow as tf
from src.layers.sparse_dense import SparseDense
from src.scheduler.tf_sparsity_scheduler import SparsityScheduler
from src.training.custom_training import custom_training_loop

class TestCustomTraining(unittest.TestCase):

    def setUp(self):
        # Create a simple model with sparse layers
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            SparseDense(128),
            tf.keras.layers.ReLU(),
            SparseDense(10)
        ])
        self.initial_sparsity = 0.5
        self.final_sparsity = 0.1
        self.total_epochs = 10

        # Initialize the SparsityScheduler
        self.scheduler = SparsityScheduler(initial_sparsity=self.initial_sparsity, final_sparsity=self.final_sparsity, total_epochs=self.total_epochs, criteria='linear')

        # Prepare the data
        self.train_data = tf.random.normal((1000, 784))
        self.train_labels = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)
        self.train_data = tf.data.Dataset.from_tensor_slices((self.train_data, self.train_labels)).batch(32)

    def test_custom_training_loop(self):
        # Manually set the model for the scheduler
        self.scheduler.set_model(self.model)
        custom_training_loop(self.model, self.scheduler, epochs=self.total_epochs, train_data=self.train_data)

if __name__ == '__main__':
    unittest.main()
