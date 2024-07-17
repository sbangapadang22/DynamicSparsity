import tensorflow as tf

class SparseDense(tf.keras.layers.Layer):
    def __init__(self, units, sparsity_level=0.5, **kwargs):
        super(SparseDense, self).__init__(**kwargs)
        self.units = units
        self.sparsity_level = sparsity_level

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        self.prune_mask = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='ones',
                                          trainable=False)
        super(SparseDense, self).build(input_shape)

    def call(self, inputs, training=False):
        if training:
            # Apply sparsity during training
            sparse_w = self.apply_sparsity(self.w)
            return tf.matmul(inputs, sparse_w) + self.b
        else:
            return tf.matmul(inputs, self.w) + self.b

    def apply_sparsity(self, weights):
        num_elements = tf.size(weights)
        k = tf.cast(self.sparsity_level * tf.cast(num_elements, tf.float32), tf.int32)
        k = tf.minimum(k, num_elements)
        k = tf.maximum(k, 1)  # Ensure k is at least 1

        abs_weights = tf.math.abs(weights)
        top_k_values = tf.math.top_k(tf.reshape(abs_weights, [-1]), k=k).values
        threshold = tf.reduce_min(top_k_values)
        mask = abs_weights >= threshold
        sparse_weights = tf.where(mask, weights, tf.zeros_like(weights))
        return sparse_weights

    def get_config(self):
        config = super(SparseDense, self).get_config()
        config.update({
            'units': self.units,
            'sparsity_level': self.sparsity_level
        })
        return config
