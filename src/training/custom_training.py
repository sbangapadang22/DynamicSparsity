import tensorflow as tf

def custom_training_loop(model, scheduler, epochs, train_data, val_data=None):
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Manually set the model in the scheduler
    scheduler.set_model(model)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        # Adjust sparsity
        scheduler.on_epoch_begin(epoch)

        for step, (batch_data, batch_labels) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(batch_data, training=True)
                loss_value = loss_fn(batch_labels, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 10 == 0:
                print(f'Step {step}, Loss: {loss_value.numpy()}')

        # Evaluate on validation set if provided
        if val_data:
            val_loss = 0
            val_steps = 0
            for batch_data, batch_labels in val_data:
                val_logits = model(batch_data, training=False)
                val_loss += loss_fn(batch_labels, val_logits).numpy()
                val_steps += 1

            val_loss /= val_steps
            print(f'Epoch {epoch+1} Loss: {loss_value.numpy()}, Validation Loss: {val_loss}')

        # Debug: Print sparsity levels
        for layer in model.layers:
            if hasattr(layer, 'sparsity_level'):
                print(f'Layer {layer.name} sparsity level: {layer.sparsity_level}')
