FROM python:3.10-slim

WORKDIR /app

RUN pip install flwr

COPY ./federate_server.py /app

ENTRYPOINT ["python", "federate_server.py"]
