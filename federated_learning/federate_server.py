import sys
from concurrent.futures import ThreadPoolExecutor

from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
        "batch_size": 8,
        "local_epochs": 3,
    }
    return config

def run_server(port, min_clients, training_rounds):
    strategy = FedAvgAndroid(
        fraction_fit=1.0, 
        fraction_evaluate=1.0,
        min_fit_clients=min_clients, # start training after this number of devices connect
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
    )

    try:
        # Start Flower server for 10 rounds of federated learning
        print(f'running server on port {port}')
        start_server(
            server_address=f"0.0.0.0:{port}",
            config=ServerConfig(num_rounds=training_rounds), # won't start next round if client has small dataset
            strategy=strategy,
        )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        min_clients = 1
        training_rounds = 10
    else:
        min_clients = int(sys.argv[1])
        training_rounds = int(sys.argv[2])

    print(f'min_clients={min_clients} training_rounds={training_rounds}')

    ports = {
        "local_time": 8885,
        "cloud_computation_time": 8886,
        "cloud_transmission_time": 8887
    }
    with ThreadPoolExecutor(len(ports)) as executor:
        jobs = [executor.submit(run_server, port, min_clients, training_rounds) for port in ports.values()]
        executor.shutdown()
    