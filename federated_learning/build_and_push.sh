#!/bin/bash

sudo docker build -t flok3n/federated-server:latest .

sudo docker push flok3n/federated-server:latest
