---
version: '3'
services:
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
 
    kafka-1:
        image: confluentinc/cp-kafka:latest
        hostname: kafka1
        container_name: kafka1
        depends_on:
            - zookeeper-1
            - zookeeper-2
            - zookeeper-3
        environment:
            KAFKA_BROKER_ID: 1
            KAFKA_ZOOKEEPER_CONNECT: zookeeper1:2181,zookeeper2:2181,zookeeper3:2181
            KAFKA_LISTENERS: INTERNAL://kafka1:19091, PLAINTEXT://kafka1:9091
            KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,PLAINTEXT:PLAINTEXT
            KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9091,INTERNAL://kafka1:19091
            KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
            KAFKA_JMX_PORT: 9991
            KAFKA_JMX_HOSTNAME: kafka1
        ports:
           - "9091:9091"
           - "9991:9991"

    kafka-2:
        image: confluentinc/cp-kafka:latest
        hostname: kafka2
        container_name: kafka2
        depends_on:
            - kafka-1
        environment:
            KAFKA_BROKER_ID: 2
            KAFKA_ZOOKEEPER_CONNECT: zookeeper1:2181,zookeeper2:2181,zookeeper3:2181
            KAFKA_LISTENERS: INTERNAL://kafka2:19092, PLAINTEXT://kafka2:9092
            KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,PLAINTEXT:PLAINTEXT
            KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,INTERNAL://kafka2:19092
            KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
            KAFKA_JMX_PORT: 9992
            KAFKA_JMX_HOSTNAME: localhost
        ports:
           - "9092:9092"
           - "9992:9992"

    kafka-3:
        image: confluentinc/cp-kafka:latest
        hostname: kafka3
        container_name: kafka3
        depends_on:
            - kafka-2
        environment:
            KAFKA_BROKER_ID: 3
            KAFKA_ZOOKEEPER_CONNECT: zookeeper1:2181,zookeeper2:2181,zookeeper3:2181
            KAFKA_LISTENERS: INTERNAL://kafka3:19093, PLAINTEXT://kafka3:9093
            KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,PLAINTEXT:PLAINTEXT
            KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9093,INTERNAL://kafka3:19093
            KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
            KAFKA_JMX_PORT: 9993
            KAFKA_JMX_HOSTNAME: localhost
        ports:
           - "9093:9093"
           - "9993:9993"


    schema_registry:
        image: confluentinc/cp-schema-registry
        hostname: schema_registry
        container_name: schema_registry
        depends_on:
            - kafka-3
        ports:
            - "8081:8081"
        environment:
            SCHEMA_REGISTRY_HOST_NAME: schema_registry
            SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: PLAINTEXT://kafka1:19091,PLAINTEXT://kafka2:19092,PLAINTEXT://kafka3:19093

    connect:
       image: confluentinc/cp-kafka-connect:latest
       hostname: connect
       container_name: connect
       depends_on:
         - zookeeper-3
         - kafka-3 
         - schema_registry
       ports:
         - "8083:8083"
       environment:
         CONNECT_BOOTSTRAP_SERVERS: PLAINTEXT://kafka1:19091
         # CONNECT_KAFKASTORE_SECURITY_PROTOCOL: PLAINTEXT
         CONNECT_REST_ADVERTISED_HOST_NAME: connect
         CONNECT_REST_PORT: 8083
         CONNECT_GROUP_ID: compose-connect-group
         CONNECT_CONFIG_STORAGE_TOPIC: docker-connect-configs
         CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR: 1
         CONNECT_OFFSET_FLUSH_INTERVAL_MS: 10000
         CONNECT_OFFSET_STORAGE_TOPIC: docker-connect-offsets
         CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR: 1
         CONNECT_STATUS_STORAGE_TOPIC: docker-connect-status
         CONNECT_STATUS_STORAGE_REPLICATION_FACTOR: 1
         CONNECT_KEY_CONVERTER: org.apache.kafka.connect.storage.StringConverter
         CONNECT_VALUE_CONVERTER: io.confluent.connect.protobuf.ProtobufConverter

         CONNECT_VALUE_CONVERTER_SCHEMA_REGISTRY_URL: http://schema_registry:8081
         CONNECT_INTERNAL_KEY_CONVERTER: "org.apache.kafka.connect.json.JsonConverter"
         CONNECT_INTERNAL_VALUE_CONVERTER: "org.apache.kafka.connect.json.JsonConverter"
         CONNECT_ZOOKEEPER_CONNECT: 'zookeeper1:2181'
         CONNECT_PLUGIN_PATH: "/usr/share/java,/usr/share/confluent-hub-components,/usr/local/share/kafka/plugins"
         CONNECT_LOG4J_LOGGERS: org.apache.zookeeper=ERROR,org.I0Itec.zkclient=ERROR,org.reflections=ERROR
       volumes:
         # Although we only need the .jar, I had issues when attempting to use the single file as the host volume.
         # Just specified the parent directory (/target), and it seems to work now.
         - ./connect/target:/usr/local/share/kafka/plugins/transform

