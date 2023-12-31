import os

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 28
NUM_CLASSES = 10

class FmnistModel(tf.Module):
    def __init__(self, lr=1e-4):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        optimizer = tf.keras.optimizers.Adam(lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer, loss=loss)

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
        tf.TensorSpec([None], tf.float32)
    ])
    def train_epoch(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            model_preds = self.model(x_batch)
            loss = self.model.loss(y_batch, model_preds)
        gradients = tape.gradient(loss, self.model.trainable_variables) 
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32)
    ])
    def predict(self, x):
        logits = self.model(x) 
        y = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1, output_type=tf.int32)
        return {
            "output": y,
            "logits": logits
        }

    @tf.function(input_signature=[tf.TensorSpec([], tf.string)]) 
    def save(self, path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(filename=path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')
        return {"result": tf.constant(1)}

    @tf.function(input_signature=[
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32),
    ]) 
    def compute_loss(self, y_true, logits_pred):
        return {"loss": self.model.loss(y_true, logits_pred)}

    @tf.function(input_signature=[tf.TensorSpec([], tf.string)]) 
    def restore(self, path):
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(file_pattern=path, tensor_name=var.name, dt=var.dtype, name='restore')
            var.assign(restored)
        return {"result": tf.constant(1)}

    # android tflite throws err for functions with no inputs 
    @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
    def get_weights_for_fl(self, unused): 
        return {
            f'a{index}': weight for index, weight in enumerate(self.model.weights)
        }
    
    @tf.function
    def set_weights_from_fl(self, **params):
        for index, weight in enumerate(self.model.weights):
            param = params[f'a{index}']
            weight.assign(param)
        return self.get_weights_for_fl(unused="...")
    

fmnist = tf.keras.datasets.fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = fmnist

def pretrain() -> FmnistModel:
    NUM_EPOCHS = 20
    BATCH_SIZE = 100
    train_ds = tf.data.Dataset.from_tensor_slices((train_images.astype(np.float32), train_labels.astype(np.float32)))
    train_ds = train_ds.shuffle(train_images.shape[0]).batch(BATCH_SIZE)

    model = FmnistModel()

    losses = []

    for i in range(NUM_EPOCHS):
        epoch_loss = 0.
        batches = 0
        for x, y in train_ds:
            res = model.train_epoch(x, y)
            epoch_loss += res['loss']
            batches += 1
        loss = epoch_loss / batches
        losses.append(loss)
        if (i + 1) % 5 == 0:
            print(f'epoch {i + 1}: loss={loss}')

    model.save(tf.constant('./model'))
    return model 


def test(predictor):
    test_ds = tf.data.Dataset.from_tensor_slices((test_images.astype(np.float32), test_labels.astype(np.int64)))
    test_ds = test_ds.batch(32)
    accs = []
    for x, y in test_ds:
        # res = model.predict(x)
        res = predictor(x)
        y_pred = res['output']
        logits = res['logits']
        acc = tf.reduce_sum(tf.cast(tf.cast(y, tf.int32) == y_pred, tf.int32)) / tf.shape(y_pred)[0]
        accs.append(acc)

    print(f'accuracy: {tf.reduce_mean(accs)}')

    
def load_lite(path='./model_lite'):
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    return converter.convert()

def model_to_lite(model: FmnistModel, tf_model_path='./tfmodel', tf_lite_model_path='./model.tflite'):
    get_weights_for_fl = model.get_weights_for_fl.get_concrete_function()
    init_params = get_weights_for_fl(unused="trash")
    print(f"Initial parameters is {init_params}.")
    set_weights_from_fl = model.set_weights_from_fl.get_concrete_function(**init_params)
    restore_test = set_weights_from_fl(**init_params)
    print(f"Restore test result: {restore_test}.")
    tf.saved_model.save(
        model,
        tf_model_path,
        signatures={
            'train_epoch': model.train_epoch.get_concrete_function(),
            'predict': model.predict.get_concrete_function(),
            'save': model.save.get_concrete_function(),
            'restore': model.restore.get_concrete_function(),
            "compute_loss": model.compute_loss.get_concrete_function(),
            'get_weights_for_fl': get_weights_for_fl,
            'set_weights_from_fl': set_weights_from_fl
        })
    
    lite_model = load_lite(tf_model_path)
    with open(tf_lite_model_path, 'wb') as model_file:
        model_file.write(lite_model)


def write_input_files(images: np.ndarray, y: np.ndarray, format='png', path='fmnist_images'):
    try:
        os.mkdir(path)
    except:
        pass

    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(path, f'x_{i}.{format}'))

def create_pretrained_tflite_model():
    model = pretrain()
    test(lambda x: model.predict(x))
    model_to_lite(model)

create_pretrained_tflite_model()

# # model = FmnistModel()
# model.restore(tf.constant('./model'))
# test(lambda x: model.predict(x))

# model = FmnistModel()
# model.save('./model2')
# model = FmnistModel()
# model.restore(tf.constant('./model'))
# test(lambda x: model.predict(x))
# model_to_lite(model)

# lite_model = load_lite()

# interpreter = tf.lite.Interpreter(model_content=lite_model)
# interpreter = tf.lite.Interpreter(model_path='./model.tflite')
# interpreter.allocate_tensors()

# predict = interpreter.get_signature_runner("predict")
# img = Image.open('fmnist_images/x_2.png')

# def gs(pixel):
#     # return (255 - (
#     #     ((pixel >> 16) & 0xFF) * 0.299 +
#     #     ((pixel >> 8) & 0xFF) * 0.587 +
#     #     (pixel & 0xFF) * 0.114 
#     # )) / 255.
#     return (255 - pixel) / 255.


# arr = np.array(img)

# buf = np.empty((28, 28), dtype=np.float32)
# for i in range(28):
#     for j in range(28):
#         buf[i, j] = gs(arr[i, j])

# print(arr.shape, arr.dtype)
# res = predict(x=buf[np.newaxis,...])
# print(res)
# logits = res['logits']

# compute_loss = interpreter.get_signature_runner("compute_loss")
# print(compute_loss.get_input_details())
# print(f'loss={compute_loss(y_true=np.array([1.], dtype=np.float32), logits_pred=logits)}')

# res = predict(x=test_images[2][np.newaxis,...].astype(np.float32))
# print(res)
# test(lambda v: predict(x=v))

# write_input_files(test_images[:100], test_labels[:100])
# print(list(test_labels[:100]))

# train_epoch = interpreter.get_signature_runner("train_epoch")
# print(test_labels[:4].astype(np.float32))
# res = train_epoch(x_batch=test_images[:4].astype(np.float32), y_batch=test_labels[:4].astype(np.float32))
# print(res)

# print(test_labels[:4].shape)