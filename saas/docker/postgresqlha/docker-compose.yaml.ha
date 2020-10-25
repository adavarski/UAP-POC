version: '3.1'

services:
  haproxy:
    build: 'haproxy'
    ports:
      - 5432:5432
      - 7000:7000
    depends_on:
      - zookeeper-3
      - pg-master
      - pg-slave

  zookeeper-1:
    image: confluentinc/cp-zookeeper:latest
    hostname: zookeeper1
    container_name: zookeeper1
    environment:
      ZOOKEEPER_SERVER_ID: 1
      ZOOKEEPER_CLIENT_PORT: "2181"
      ZOOKEEPER_TICK_TIME: "2000"
      ZOOKEEPER_SERVERS: zookeeper1:2888:3888;zookeeper2:2888:3888;zookeeper3:2888:3888
    ports:
      - "2181:2181"

  zookeeper-2:
    image: confluentinc/cp-zookeeper:latest
    hostname: zookeeper2
    container_name: zookeeper2
    environment:
      ZOOKEEPER_SERVER_ID: 2
      ZOOKEEPER_CLIENT_PORT: "2182"
      ZOOKEEPER_TICK_TIME: "2000"
      ZOOKEEPER_SERVERS: zookeeper1:2888:3888;zookeeper2:2888:3888;zookeeper3:2888:3888
    ports:
      - "2182:2182"

  zookeeper-3:
    image: confluentinc/cp-zookeeper:latest
    hostname: zookeeper3
    container_name: zookeeper3
    environment:
      ZOOKEEPER_SERVER_ID: 3
      ZOOKEEPER_CLIENT_PORT: "2183"
      ZOOKEEPER_TICK_TIME: "2000"
      ZOOKEEPER_SERVERS: zookeeper1:2888:3888;zookeeper2:2888:3888;zookeeper3:2888:3888
    ports:
      - "2183:2183"

  pg-master:
    build: 'postgres'
    restart: always
    hostname: pg-master
    environment:
      POSTGRES_USER: 'postgres'
      POSTGRES_PASSWORD: 'postgres'
      PGDATA: '/var/lib/postgresql/data/pgdata'
    expose:
      - 5432
      - 8008
    ports:
      - 5433:5432
    volumes:
     - '/var/lib/postgresql/data'
    command: su - postgres -c 'patroni /postgres0.yml'

  pg-slave:
    build: 'postgres'
    restart: always
    hostname: pg-slave
    expose:
      - 5432
      - 8008
    ports:
      - 5434:5432
    volumes:
     - '/var/lib/postgresql/data'
    environment:
      POSTGRES_USER: 'postgres'
      POSTGRES_PASSWORD: 'postgres'
      PGDATA: '/var/lib/postgresql/data/pgdata'
      #REPLICATE_FROM: 'pg-master'
    command: su - postgres -c 'patroni /postgres1.yml'
