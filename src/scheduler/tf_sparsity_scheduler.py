import tensorflow as tf

class SparsityScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_sparsity=0.5, final_sparsity=0.1, total_epochs=100):
        super(SparsityScheduler, self).__init__()
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Linearly decrease sparsity from initial_sparsity to final_sparsity
        sparsity = self.initial_sparsity - (self.initial_sparsity - self.final_sparsity) * (epoch / self.total_epochs)
        # Update sparsity level in all sparse layers
        for layer in self.model.layers:
            if hasattr(layer, 'sparsity_level'):
                layer.sparsity_level = sparsity
