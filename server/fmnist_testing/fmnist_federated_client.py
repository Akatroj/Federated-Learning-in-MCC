import flwr as fl
import numpy as np
import tensorflow as tf

fmnist = tf.keras.datasets.fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = fmnist
N = 4096
idx = np.random.randint(0, train_images.shape[0] // N - 1)
test_images = test_images.astype(np.float32)[:1024]
test_labels = test_labels.astype(np.float32)[:1024]
train_images = train_images.astype(np.float32)[idx * N:(idx+1) * N]
train_labels = train_labels.astype(np.float32)[idx * N:(idx+1) * N]
print(f'client idx: {idx}')


class FederatedClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        super().__init__()
        self.interpreter = tf.lite.Interpreter(model_path='./model.tflite')
        self.interpreter.allocate_tensors()

        self.get_weights_for_fl = self.interpreter.get_signature_runner('get_weights_for_fl') 
        self.set_weights_from_fl = self.interpreter.get_signature_runner('set_weights_from_fl')
        self.train_epoch = self.interpreter.get_signature_runner('train_epoch')
        self.predict = self.interpreter.get_signature_runner('predict')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for k, v in self.get_weights_for_fl().items():
            print(f'{k}: {v.shape} (total = {np.prod(v.shape)})')


    def get_weights(self):
        weights = sorted(self.get_weights_for_fl().items(), key=lambda x: x[0])
        return [w[1] for w in weights]
    
    def set_weights(self, weights):
        weights = {
            f'a{index}': weight for index, weight in enumerate(weights)
        }
        self.set_weights_from_fl(**weights)

    def get_parameters(self, config):
        print('getting parameters')
        # print(config) # empty
        return self.get_weights()

    def fit(self, parameters, config):
        print('fitting')
        # # print(parameters)
        epochs = config.get('local_epochs', 1)
        batch_size = config.get('batch_size', 16)
        
        self.set_weights(parameters)
        for _ in range(epochs):
            batch_loss = []
            for i in range(0, train_images.shape[0], batch_size):
                res = self.train_epoch(x_batch=train_images[i:i+batch_size], y_batch=train_labels[i:i+batch_size])
                batch_loss.append(res['loss'])
            print(f'loss: {np.mean(batch_loss)}')

        # overflow on larger wtf?
        return self.get_weights(), 4, {}

    def evaluate(self, parameters, config):
        print('evaluating')
        # print(config) # empty

        self.set_weights(parameters)
        res = self.predict(x=train_images)
        y = res['output']
        logits = res['logits']
        loss = self.loss(y.astype(np.float32), logits)
        accuracy = np.sum(y.astype(np.int32) == train_labels.astype(np.int32)) / train_labels.shape[0]
        print(f'accuracy: {accuracy}')
        print(f'loss: {loss}')

        return float(loss), 4, {"accuracy": float(accuracy)}
        # return float(2.2), 4, {"accuracy": float(0.85)}
    
    def save(self):
        self.interpreter.get_signature_runner('save')(path=np.array(['./fed_trained_model']))

    def restore(self):
        self.interpreter.get_signature_runner('restore')(path=np.array(['./fed_trained_model']))

client = FederatedClient()
fl.client.start_numpy_client(server_address="127.0.0.1:8085", client=client)

# client.save()