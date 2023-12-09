from concurrent.futures import ThreadPoolExecutor

from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
        "batch_size": 2,
        "local_epochs": 3,
    }
    return config


def run_server(port):
    strategy = FedAvgAndroid(
        fraction_fit=1.0, 
        fraction_evaluate=1.0,
        min_fit_clients=1, # start training after this number of devices connect
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
    )

    try:
        # Start Flower server for 10 rounds of federated learning
        print(f'running server on port {port}')
        start_server(
            server_address=f"0.0.0.0:{port}",
            config=ServerConfig(num_rounds=10), # won't start next round if client has small dataset
            strategy=strategy,
        )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    local_time_port, cloud_time_port = 8885, 8886
    with ThreadPoolExecutor(2) as executor:
        jobs = [executor.submit(run_server, port) for port in (local_time_port, cloud_time_port)]
        executor.shutdown()
    