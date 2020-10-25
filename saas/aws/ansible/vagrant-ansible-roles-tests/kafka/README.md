## Kafka cluster (messaging stack)
======================================

### Overview

This repo it helps you quickly spin up a kafka cluster + zookeeper cluster, and setup kafka + zookeeper cluster (messaging stack) with ansible.

Note: All packages are from CentOS 7


### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started
`yum install ansible`

`vagrant up`

```
# vagrant status
Current machine states:

zoo01                     running (virtualbox)
zoo02                     running (virtualbox)
zoo03                     running (virtualbox)
kafka01                   running (virtualbox)
kafka02                   running (virtualbox)
kafka03                   running (virtualbox)
```

### Check zookeeper cluster
```
[root@zoo01 ~]# /opt/zookeeper-3.4.13/bin/zkServer.sh status
ZooKeeper JMX enabled by default
Using config: /opt/zookeeper-3.4.13/bin/../conf/zoo.cfg
Mode: follower

[root@zoo01 ~]# echo stat | nc zoo01 2181 | grep Mode
Mode: follower

[root@zoo01 ~]# echo stat | nc zoo02 2181 | grep Mode
Mode: leader

[root@zoo01 ~]# echo stat | nc zoo03 2181 | grep Mode
Mode: follower

[root@zoo01 ~]# echo mntr | nc zoo02 2181
zk_version	3.4.13-2d71af4dbe22557fda74f9a9b4309b15a7487f03, built on 06/29/2018 04:05 GMT
zk_avg_latency	0
zk_max_latency	0
zk_min_latency	0
zk_packets_received	2
zk_packets_sent	1
zk_num_alive_connections	1
zk_outstanding_requests	0
zk_server_state	leader
zk_znode_count	4
zk_watch_count	0
zk_ephemerals_count	0
zk_approximate_data_size	27
zk_open_file_descriptor_count	28
zk_max_file_descriptor_count	4096
zk_fsync_threshold_exceed_count	0
zk_followers	2
zk_synced_followers	2
zk_pending_syncs	0
zk_last_proposal_size	-1
zk_max_proposal_size	-1
zk_min_proposal_size	-1

```
### Check Kafka cluster:

Check kafka brokers:
```
# ssh -p 2264 -i .vagrant/machines/kafka01/virtualbox/private_key vagrant@127.0.0.1 'sudo grep started /var/log/kafka/server.log'
[2020-09-27 17:11:16,479] INFO [KafkaServer id=0] started (kafka.server.KafkaServer)
# ssh -p 2265 -i .vagrant/machines/kafka02/virtualbox/private_key vagrant@127.0.0.1 'sudo grep started /var/log/kafka/server.log'
[2020-09-27 17:13:29,761] INFO [KafkaServer id=1] started (kafka.server.KafkaServer)
# ssh -p 2266 -i .vagrant/machines/kafka03/virtualbox/private_key vagrant@127.0.0.1 'sudo grep started /var/log/kafka/server.log'
[2020-09-27 17:15:43,377] INFO [KafkaServer id=2] started (kafka.server.KafkaServer)
```

Now that the brokers are up, you can test that they’re working as expected by creating a topic.
```
[root@kafka01 ~]# kafka-topics --create --topic InfraAPI --partitions 3 --replication-factor 3 --if-not-exists --zookeeper zoo01:2181
Created topic InfraAPI.
```
Now verify that the topic is created successfully by describing the topic.

```
[root@kafka01 ~]# kafka-topics --describe --topic InfraAPI --zookeeper zoo01:2181
Topic: InfraAPI	PartitionCount: 3	ReplicationFactor: 3	Configs: 
	Topic: InfraAPI	Partition: 0	Leader: 2	Replicas: 2,0,1	Isr: 2,0,1
	Topic: InfraAPI	Partition: 1	Leader: 0	Replicas: 0,1,2	Isr: 0,1,2
	Topic: InfraAPI	Partition: 2	Leader: 1	Replicas: 1,2,0	Isr: 1,2,0
```
Next, you will generate some data to the InfraAPI topic that was just created.

```
[root@kafka01 ~]# bash -c "seq 10|kafka-console-producer --broker-list kafka01:9092 --topic InfraAPI&& echo 'Produced 10 messages.'"
>>>>>>>>>>>Produced 10 messages.
```
Try reading the messages back using the Console Consumer and make sure they’re all accounted for.

```
[root@kafka01 ~]# kafka-console-consumer --bootstrap-server kafka01:9092 --topic InfraAPI --from-beginning -max-messages 3
1
2
3
Processed a total of 3 messages

```

### References
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)

