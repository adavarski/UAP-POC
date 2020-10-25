# Server and Client for monitoring Docker containers

Docker images ready to run for Sensu server and client service.
The client has already a couple of plugins installed for monitoring docker containers, disk space and other few things.

## Requirements

First make sure *Docker* and *Docker Compose* are installed on the machine with:

    $ docker -v
    $ docker-compose -v

## How to Use

## Settings Up the Environment

The following settings are available:

| Variable       | Description                                                             | Default |
|----------------|-------------------------------------------------------------------------|---------|
| SENSU_HOST     | The Sensu server host                                                   |         |
| SENSU_USER     | The username to access RabbitMQ                                         |         |
| SENSU_PASSWORD | The password to access RabbitMQ                                         |         |
| CLIENT_NAME    | The client name that will show up in uchiwa UI                          |         |
| CLIENT_IP      | The client IP that will be used by Sensu server to talk with the client |         |


```
Start sensu monitoring stack:

docker-compose -f docker-compose-dev.yml up -d
docker-compose up -d 

Checks:

docker-compose ps
docker-compose logs
WEB UI (uchiwa): http://{IP}:3001

Clean: 

docker-compose down
docker rmi `docker images|grep sensu |awk '{print $3}'` -f
```
