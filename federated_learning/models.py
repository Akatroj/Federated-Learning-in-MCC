import tensorflow as tf

from tflite_model_utils import init_tflite_requirements, save_tflite_model
from tflite_model_wrapper import TFLiteModelWrapper

MODELS_DIR = './models'

def create_local_time_model(output_dir=MODELS_DIR) -> tuple[TFLiteModelWrapper, str]:
    input_dimensions = 3
    output_path = f'{output_dir}/local_time.tflite'

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()

    tflite_wrapper = TFLiteModelWrapper(model, optimizer, loss)
    init_tflite_requirements(tflite_wrapper, input_dimensions)
    save_tflite_model(tflite_wrapper, f'{output_dir}/local_time_model', output_path)
    return tflite_wrapper, output_path


def create_cloud_time_model(output_dir=MODELS_DIR) -> tuple[TFLiteModelWrapper, str]:
    input_dimensions = 3
    output_path = f'{output_dir}/cloud_time.tflite'

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanSquaredError()

    tflite_wrapper = TFLiteModelWrapper(model, optimizer, loss)
    init_tflite_requirements(tflite_wrapper, input_dimensions)
    save_tflite_model(tflite_wrapper, f'{output_dir}/cloud_time_model', output_path)
    return tflite_wrapper, output_path

if __name__ == "__main__":
    create_local_time_model()
    create_cloud_time_model()
    