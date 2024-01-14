#!/bin/bash

sudo docker build -t flok3n/federated-server:1.0.1 .

sudo docker push flok3n/federated-server:1.0.1
