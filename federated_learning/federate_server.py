from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid

PORT = 8085

def fit_config(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
        "batch_size": 10,
        "local_epochs": 5,
    }
    return config


def main():
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
        start_server(
            server_address=f"0.0.0.0:{PORT}",
            config=ServerConfig(num_rounds=10),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
    