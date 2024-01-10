import tensorflow as tf

from tflite_model_utils import (init_tflite_requirements,
                                print_model_tensor_sizes, save_tflite_model)
from tflite_model_wrapper import TFLiteModelWrapper

MODELS_DIR = './models'

def create_local_time_model(output_dir=MODELS_DIR) -> tuple[TFLiteModelWrapper, str]:
    input_dimensions = 6
    output_path = f'{output_dir}/local_time.tflite'

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()

    tflite_wrapper = TFLiteModelWrapper(model, optimizer, loss)
    init_tflite_requirements(tflite_wrapper, input_dimensions)
    save_tflite_model(tflite_wrapper, f'{output_dir}/local_time_model', output_path)
    print('local time model params:')
    print_model_tensor_sizes(tflite_wrapper)
    return tflite_wrapper, output_path

def create_cloud_computation_time_model(output_dir=MODELS_DIR) -> tuple[TFLiteModelWrapper, str]:
    input_dimensions = 7
    output_path = f'{output_dir}/cloud_computation_time.tflite'

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()

    tflite_wrapper = TFLiteModelWrapper(model, optimizer, loss)
    init_tflite_requirements(tflite_wrapper, input_dimensions)
    
    save_tflite_model(tflite_wrapper, f'{output_dir}/cloud_computation_time_model', output_path)
    print('cloud computation time model params:')
    print_model_tensor_sizes(tflite_wrapper)
    return tflite_wrapper, output_path

def create_cloud_transmission_time_model(output_dir=MODELS_DIR) -> tuple[TFLiteModelWrapper, str]:
    input_dimensions = 7
    output_path = f'{output_dir}/cloud_transmission_time.tflite'

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()

    tflite_wrapper = TFLiteModelWrapper(model, optimizer, loss)
    init_tflite_requirements(tflite_wrapper, input_dimensions)
    save_tflite_model(tflite_wrapper, f'{output_dir}/cloud_transmission_time_model', output_path)
    print('cloud transmission time model params:')
    print_model_tensor_sizes(tflite_wrapper)
    return tflite_wrapper, output_path

if __name__ == "__main__":
    create_local_time_model()
    create_cloud_computation_time_model()
    create_cloud_transmission_time_model()
 