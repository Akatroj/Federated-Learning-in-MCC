import numpy as np
import tensorflow as tf

from tflite_model_wrapper import TFLiteModelWrapper


def apply_tf_function_decorators(model: TFLiteModelWrapper, input_dimensions: int):
    model.train_epoch = tf.function(input_signature=[
        tf.TensorSpec([None, input_dimensions], tf.float32),
        tf.TensorSpec([None], tf.float32)
    ])(model.train_epoch)
    model.predict = tf.function(input_signature=[
        tf.TensorSpec([None, input_dimensions], tf.float32)
    ])(model.predict)
    model.save = tf.function(input_signature=[tf.TensorSpec([], tf.string)])(model.save)
    model.compute_loss = tf.function(input_signature=[
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None], tf.float32),
    ])(model.compute_loss)
    model.restore = tf.function(input_signature=[tf.TensorSpec([], tf.string)])(model.restore)
    model.get_weights_for_fl = tf.function(input_signature=[tf.TensorSpec([], tf.string)])(model.get_weights_for_fl)
    model.set_weights_from_fl = tf.function(model.set_weights_from_fl)
    return model

def load_tflite_model(path: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    return converter.convert()

def save_tflite_model(model: TFLiteModelWrapper, tf_model_path: str, tf_lite_model_path: str):
    get_weights_for_fl = model.get_weights_for_fl.get_concrete_function()
    init_params = get_weights_for_fl(unused="trash")
    set_weights_from_fl = model.set_weights_from_fl.get_concrete_function(**init_params)
    set_weights_from_fl(**init_params)
    
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

    lite_model = load_tflite_model(tf_model_path)
    with open(tf_lite_model_path, 'wb') as model_file:
        model_file.write(lite_model)

def init_tflite_requirements(model_wrapper: TFLiteModelWrapper, input_dimensions: int, batch_size=None):
    apply_tf_function_decorators(model_wrapper, input_dimensions)
    model_wrapper.model.build(input_shape=(batch_size, input_dimensions))
    