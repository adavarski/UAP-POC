### Start messaging stack using Docker Compose up command.

### Non-HA

Run:

```
# docker-compose up -d

# docker-compose ps
          Name                       Command            State                                    Ports                                   
----------------------------------------------------------------------------------------------------------------------------------------
messaging_connect           /etc/confluent/docker/run   Up      0.0.0.0:8083->8083/tcp, 9092/tcp                                         
messaging_kafka             /etc/confluent/docker/run   Up      0.0.0.0:29092->29092/tcp, 0.0.0.0:9092->9092/tcp, 0.0.0.0:9093->9093/tcp 
messaging_schema_registry   /etc/confluent/docker/run   Up      0.0.0.0:8081->8081/tcp                                                   
messaging_zookeeper         /etc/confluent/docker/run   Up      0.0.0.0:2182->2181/tcp, 2888/tcp, 3888/tcp    
```

Check zookeeper and kafka:

```
# docker-compose logs kafka|grep started
messaging_kafka    | [2020-07-31 07:54:39,288] INFO [KafkaServer id=1] started (kafka.server.KafkaServer)

docker-compose logs zookeeper
```
Now that the brokers are up, you can test that they’re working as expected by creating a topic.

```
# docker-compose exec kafka kafka-topics --create --topic InfraAPI-Clusters --partitions 3 --replication-factor 1 --if-not-exists --zookeeper zookeeper:2182
Created topic InfraAPI-Clusters.

```
Now verify that the topic is created successfully by describing the topic.

```
# docker-compose exec kafka kafka-topics --describe --topic InfraAPI-Clusters --zookeeper zookeeper:2182
Topic: InfraAPI-Clusters	PartitionCount: 3	ReplicationFactor: 1	Configs: 
	Topic: InfraAPI-Clusters	Partition: 0	Leader: 1	Replicas: 1	Isr: 1
	Topic: InfraAPI-Clusters	Partition: 1	Leader: 1	Replicas: 1	Isr: 1
	Topic: InfraAPI-Clusters	Partition: 2	Leader: 1	Replicas: 1	Isr: 1
```
Next, you will generate some data to the InfraAPI-Clusters topic that was just created.

```
# docker-compose exec kafka bash -c "seq 10|kafka-console-producer --broker-list kafka:29092 --topic InfraAPI-Clusters&& echo 'Produced 10 messages.'"
>>>>>>>>>>>Produced 10 messages.
```

Try reading the messages back using the Console Consumer and make sure they’re all accounted for.
```
# docker-compose exec kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic InfraAPI-Clusters --from-beginning -max-messages 2
1
2

Processed a total of 2 messages
```

Clean:

```
docker-compose down
```

### HA

Run:

```
# docker-compose -f ./docker-compose.yml.ha up -d
# docker-compose -f ./docker-compose.yml.ha ps
     Name                  Command            State                            Ports                           
--------------------------------------------------------------------------------------------------------------
connect           /etc/confluent/docker/run   Up      0.0.0.0:8083->8083/tcp, 9092/tcp                         
kafka1            /etc/confluent/docker/run   Up      0.0.0.0:9091->9091/tcp, 9092/tcp, 0.0.0.0:9991->9991/tcp 
kafka2            /etc/confluent/docker/run   Up      0.0.0.0:9092->9092/tcp, 0.0.0.0:9992->9992/tcp           
kafka3            /etc/confluent/docker/run   Up      9092/tcp, 0.0.0.0:9093->9093/tcp, 0.0.0.0:9993->9993/tcp 
schema_registry   /etc/confluent/docker/run   Up      0.0.0.0:8081->8081/tcp                                   
zookeeper1        /etc/confluent/docker/run   Up      0.0.0.0:2181->2181/tcp, 2888/tcp, 3888/tcp               
zookeeper2        /etc/confluent/docker/run   Up      2181/tcp, 0.0.0.0:2182->2182/tcp, 2888/tcp, 3888/tcp     
zookeeper3        /etc/confluent/docker/run   Up      2181/tcp, 0.0.0.0:2183->2183/tcp, 2888/tcp, 3888/tcp 
```

Check kafka brokers:

```
# docker-compose logs kafka-1|grep started
kafka1             | [2020-07-31 10:43:32,873] INFO [KafkaServer id=1] started (kafka.server.KafkaServer)
# docker-compose logs kafka-2|grep started
kafka2             | [2020-07-31 10:43:35,074] INFO [KafkaServer id=2] started (kafka.server.KafkaServer)
# docker-compose logs kafka-3|grep started
kafka3             | [2020-07-31 10:43:32,676] INFO [KafkaServer id=3] started (kafka.server.KafkaServer)
```
Now that the brokers are up, you can test that they’re working as expected by creating a topic.
```
# docker-compose exec kafka-1 kafka-topics --create --topic InfraAPI-Clusters --partitions 3 --replication-factor 3 --if-not-exists --zookeeper zookeeper1:2181
Created topic InfraAPI-Clusters.
```
Now verify that the topic is created successfully by describing the topic.
```
# docker-compose exec kafka-1 kafka-topics --describe --topic InfraAPI-Clusters --zookeeper zookeeper1:2181
Topic: InfraAPI-Clusters	PartitionCount: 3	ReplicationFactor: 3	Configs: 
	Topic: InfraAPI-Clusters	Partition: 0	Leader: 1	Replicas: 1,3,2	Isr: 1,3,2
	Topic: InfraAPI-Clusters	Partition: 1	Leader: 2	Replicas: 2,1,3	Isr: 2,1,3
	Topic: InfraAPI-Clusters	Partition: 2	Leader: 3	Replicas: 3,2,1	Isr: 3,2,1
```
Next, you will generate some data to the InfraAPI-Clusters topic that was just created.

```
# docker-compose exec kafka-1 bash -c "seq 10|kafka-console-producer --broker-list kafka1:19091 --topic InfraAPI-Clusters&& echo 'Produced 10 messages.'"
>>>>>>>>>>>Produced 10 messages.
```

Try reading the messages back using the Console Consumer and make sure they’re all accounted for.
```
# docker-compose exec kafka-1 kafka-console-consumer --bootstrap-server kafka1:19091 --topic InfraAPI-Clusters --from-beginning -max-messages 3
1
2
3
Processed a total of 3 messages
```
Clean:

```
# docker-compose -f docker-compose.yml.ha down
```
### Non-HA + monitoring

Run:

```
# docker-compose -f ./docker-compose.yml.monitoring up -d
# docker-compose -f ./docker-compose.yml.monitoring ps
          Name                         Command               State                                    Ports                                   
---------------------------------------------------------------------------------------------------------------------------------------------
messaging_connect           /etc/confluent/docker/run        Up      0.0.0.0:8083->8083/tcp, 9092/tcp                                         
messaging_elasticsearch     /usr/local/bin/docker-entr ...   Up      0.0.0.0:9200->9200/tcp, 9300/tcp                                         
messaging_grafana           /run.sh                          Up      0.0.0.0:3000->3000/tcp                                                   
messaging_kafka             /etc/confluent/docker/run        Up      0.0.0.0:29092->29092/tcp, 0.0.0.0:9092->9092/tcp, 0.0.0.0:9093->9093/tcp 
messaging_kibana            /usr/local/bin/dumb-init - ...   Up      0.0.0.0:5601->5601/tcp                                                   
messaging_schema_registry   /etc/confluent/docker/run        Up      0.0.0.0:8081->8081/tcp                                                   
messaging_zookeeper         /etc/confluent/docker/run        Up      0.0.0.0:2181->2181/tcp, 2888/tcp, 3888/tcp  
```

#### Service locations

##### Kibana (Elasticsearch console/search interface)
http://localhost:5601/
##### Create the index in the index management section
http://localhost:5601/app/kibana#/management/kibana/index_patterns?_g=()
##### Go to discover page and look at your data
http://localhost:5601/app/kibana#/discover?_g=()

##### Dev console to send API requests to ES/debug grok
http://localhost:5601/app/kibana#/dev_tools/console

##### Grafana
http://localhost:3000/ login: admin / admin

##### Schema Registry / Kafka Connect
TODO: add some info and utility scripts
You can use the rest api's via

##### Schema Registry
http://localhost:8081/

http://localhost:8081/subjects # lists registered schemas

http://localhost:8081/schemas/ids/1 # will show the schema of a message

##### Kafka Connect
http://localhost:8083/

http://localhost:8083/connectors shows you the registered connectors


Clean:

```
# docker-compose -f ./docker-compose.yml.monitoring down
```
