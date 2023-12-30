## Requirements

- python>=3.10

## Installation

`pip install -r requirements.txt`


## Usage

- Create models as in `models.py` file, then save result `.tflite` files in android app assets. 
- Run federated learning server using `federate_server.py` file

Fmnist directory contains testing/junk files for flower and tflite with fmnist. `fmnist_model.py` file contains functions that allow building tflite model that should be copied to assets of mobile app. `fmnist_federated_client.py` is an example usage of this model with flower (training + evaluation), not needed in general. 