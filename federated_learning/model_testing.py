from typing import Any, Callable

import numpy as np
import tensorflow as tf

from models import create_local_time_model
from tflite_model_wrapper import TFLiteModelWrapper


def pretrain(model: TFLiteModelWrapper, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int) -> TFLiteModelWrapper:
    train_ds = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32), y_train.astype(np.float32)))
    train_ds = train_ds.shuffle(x_train.shape[0]).batch(batch_size)

    for i in range(epochs):
        epoch_loss = 0.
        batches = 0
        for x, y in train_ds:
            res = model.train_epoch(x, y)
            epoch_loss += res['loss']
            batches += 1
        loss = epoch_loss / batches
        print(f'epoch {i + 1}: loss={loss}')

def test_regression(predictor: Callable[[np.ndarray], dict[str, Any]], x_test: np.ndarray, y_test: np.ndarray, batch_size: int):
    test_ds = tf.data.Dataset.from_tensor_slices((x_test.astype(np.float32), y_test.astype(np.float32)))
    test_ds = test_ds.batch(batch_size)
    preds = []
    ys = []
    for x, y in test_ds:
        res = predictor(x)
        y_pred = res['output']
        preds.extend(y_pred.numpy().ravel())
        ys.extend(y.numpy().ravel())

    preds = np.array(preds)
    ys = np.array(ys)

    rmse = np.sqrt(np.sum((ys - preds)**2))
    mae = np.sum(np.abs(ys - preds))
    print(f'rmse={rmse}\tmae={mae}')

    return preds

X = np.array([
    [1326, 790, 218884],
    [1326, 690, 198884],
    [1300, 800, 258884],
    [1000, 790, 150555],
    [1052, 1241, 120471],
    [1231, 750, 100555],
    [1234, 900, 214314],
    [1000, 200, 15050],
], dtype=np.float32)
Y = np.array([6670, 6000, 9000, 4200, 3500, 4200, 4500, 1100], dtype=np.float32)

y_mean = np.mean(Y)
y_std = np.std(Y)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = (Y - np.mean(Y)) / np.std(Y)

X_train, X_test = X[:6], X[6:]
Y_train, Y_test = Y[:6], Y[6:]

# local_time_model, tflite_path = create_local_time_model()

# pretrain(local_time_model, X_train, Y_train, 20, 2)
# test_regression(lambda x: local_time_model.predict(x), X_test, Y_test, 1)

interpreter = tf.lite.Interpreter(model_path='./models/local_time.tflite')
interpreter.allocate_tensors()

predict = interpreter.get_signature_runner("predict")
train_epoch = interpreter.get_signature_runner("train_epoch")
train_res = train_epoch(x_batch=X_train[:4], y_batch=Y_train[:4])
print(train_res)
predict_res = predict(x=X_test[:1]) 
print(predict_res)
