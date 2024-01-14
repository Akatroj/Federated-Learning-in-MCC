import tensorflow as tf


class TFLiteModelWrapper(tf.Module):
    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss)

    def train_epoch(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            model_preds = self.model(x_batch)
            loss = self.model.loss(y_batch, model_preds)
        gradients = tape.gradient(loss, self.model.trainable_variables) 
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    def predict(self, x):
        return {"output": self.model(x)}

    def save(self, path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(filename=path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')
        return {"result": tf.constant(1)}

    def compute_loss(self, y_true, y_pred):
        return {"loss": self.model.loss(y_true, y_pred)}

    def restore(self, path):
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(file_pattern=path, tensor_name=var.name, dt=var.dtype, name='restore')
            var.assign(restored)
        return {"result": tf.constant(1)}

    def get_weights_for_fl(self, unused): 
        # android tflite throws err for functions with no inputs 
        return {
            f'a{index}': weight for index, weight in enumerate(self.model.weights)
        }
    
    def set_weights_from_fl(self, **params):
        for index, weight in enumerate(self.model.weights):
            param = params[f'a{index}']
            weight.assign(param)
        return self.get_weights_for_fl(unused="...")
    