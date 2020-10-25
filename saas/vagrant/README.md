### SaaS vagrant environment - FULL

### Overview

This repo it helps you quickly spin up FULL SaaS using Vagrant+Vbox, and setup all stacks/services with ansible.


### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started
`yum install ansible -y`

`vagrant up`

```
Example: 

[root@dl360p35 saas]# vagrant up
Bringing machine 'dcs01' up with 'virtualbox' provider...
Bringing machine 'dcs02' up with 'virtualbox' provider...
Bringing machine 'dcs03' up with 'virtualbox' provider...
Bringing machine 'pg01' up with 'virtualbox' provider...
Bringing machine 'pg02' up with 'virtualbox' provider...
Bringing machine 'pg03' up with 'virtualbox' provider...
Bringing machine 'zoo01' up with 'virtualbox' provider...
Bringing machine 'zoo02' up with 'virtualbox' provider...
Bringing machine 'zoo03' up with 'virtualbox' provider...
Bringing machine 'kafka01' up with 'virtualbox' provider...
Bringing machine 'kafka02' up with 'virtualbox' provider...
Bringing machine 'kafka03' up with 'virtualbox' provider...
Bringing machine 'schemaregistry' up with 'virtualbox' provider...
Bringing machine 'kafkaconnect' up with 'virtualbox' provider...
Bringing machine 'influxdb' up with 'virtualbox' provider...
Bringing machine 'grafana' up with 'virtualbox' provider...
Bringing machine 'telegraf' up with 'virtualbox' provider...
Bringing machine 'elasticnode1' up with 'virtualbox' provider...
Bringing machine 'elasticnode2' up with 'virtualbox' provider...
Bringing machine 'elasticnode3' up with 'virtualbox' provider...
Bringing machine 'kibana' up with 'virtualbox' provider...
Bringing machine 'logstash' up with 'virtualbox' provider...
...
...

[root@dl360p35 saas]# vagrant status
Current machine states:

dcs01                     running (virtualbox)
dcs02                     running (virtualbox)
dcs03                     running (virtualbox)
pg01                      running (virtualbox)
pg02                      running (virtualbox)
pg03                      running (virtualbox)
zoo01                     running (virtualbox)
zoo02                     running (virtualbox)
zoo03                     running (virtualbox)
kafka01                   running (virtualbox)
kafka02                   running (virtualbox)
kafka03                   running (virtualbox)
schemaregistry            running (virtualbox)
kafkaconnect              running (virtualbox)
influxdb                  running (virtualbox)
grafana                   running (virtualbox)
telegraf                  running (virtualbox)
elasticnode1              running (virtualbox)
elasticnode2              running (virtualbox)
elasticnode3              running (virtualbox)
kibana                    running (virtualbox)
logstash                  running (virtualbox)

This environment represents multiple VMs. The VMs are all listed
above with their current state. For more information about a specific
VM, run `vagrant status NAME`.


```

### Postgres HA 

Spin up a 3-node cluster of PostgreSQL, managed by Patroni using Consul cluster (3-nodes).

#### What's in the cluster?

When you start the cluster, you get 3 nodes (pg01, pg02 and pg03), each running:

  - PostgreSQL
  - Patroni
  - Consul agent

All packages are from CentOS 7, except for PostgreSQL itself, which is at version 11.x.

The cluster is configured with a single primary and two asynchronous streaming replica.

#### Test HA

```

[vagrant@dcs01 ~]$ consul members
Node   Address           Status  Type    Build  Protocol  DC      Segment
dcs01  10.51.21.61:8301  alive   server  1.6.1  2         noc-mc  <all>
dcs02  10.51.21.62:8301  alive   server  1.6.1  2         noc-mc  <all>
dcs03  10.51.21.63:8301  alive   server  1.6.1  2         noc-mc  <all>
pg01   10.51.21.64:8301  alive   client  1.6.1  2         noc-mc  <default>
pg02   10.51.21.65:8301  alive   client  1.6.1  2         noc-mc  <default>
pg03   10.51.21.66:8301  alive   client  1.6.1  2         noc-mc  <default>


[vagrant@pg01 ~]$ patronictl -c /etc/patroni.yml list
+ Cluster: saas (6878606133818827720) --+----+-----------+-----------------+
| Member | Host        | Role    | State   | TL | Lag in MB | Pending restart |
+--------+-------------+---------+---------+----+-----------+-----------------+
| pg01   | 10.51.21.64 | Leader  | running |  2 |           | *               |
| pg02   | 10.51.21.65 | Replica | running |  2 |       0.0 |                 |
| pg03   | 10.51.21.66 | Replica | running |  2 |       0.0 |                 |
+--------+-------------+---------+---------+----+-----------+-----------------+
# vagrant status
Current machine states:

client                    running (virtualbox)
dcs01                     running (virtualbox)
dcs02                     running (virtualbox)
dcs03                     running (virtualbox)
pg01                      running (virtualbox)
pg02                      running (virtualbox)
pg03                      running (virtualbox)

This environment represents multiple VMs. The VMs are all listed
above with their current state. For more information about a specific
VM, run `vagrant status NAME`.
# vagrant halt pg01
==> pg01: Attempting graceful shutdown of VM...
# vagrant ssh pg02
Last login: Wed Sep 16 22:54:02 2020 from 10.0.2.2
[vagrant@pg02 ~]$ patronictl -c /etc/patroni.yml list
+ Cluster: saas (6873175265125204662) --+----+-----------+
| Member | Host        | Role    | State   | TL | Lag in MB |
+--------+-------------+---------+---------+----+-----------+
| pg02   | 10.51.21.66 | Replica | running |  3 |       0.0 |
| pg03   | 10.51.21.67 | Leader  | running |  3 |           |
+--------+-------------+---------+---------+----+-----------+
[vagrant@pg02 ~]$ exit
logout
Connection to 127.0.0.1 closed.
# vagrant up pg01
# vagrant ssh pg01
Last login: Wed Sep 16 23:01:23 2020 from 10.0.2.2
[vagrant@pg01 ~]$ patronictl -c /etc/patroni.yml list
+ Cluster: saas (6873175265125204662) --+----+-----------+
| Member | Host        | Role    | State   | TL | Lag in MB |
+--------+-------------+---------+---------+----+-----------+
| pg01   | 10.51.21.65 | Replica | running |  3 |       0.0 |
| pg02   | 10.51.21.66 | Replica | running |  3 |       0.0 |
| pg03   | 10.51.21.67 | Leader  | running |  3 |           |
+--------+-------------+---------+---------+----+-----------+

```

#### Further reading

* [PostgreSQL and Patroni cluster](https://www.linode.com/docs/databases/postgresql/create-a-highly-available-postgresql-cluster-using-patroni-and-haproxy/#before-you-begin)

#### References
* [PostgreSQL](https://www.postgresql.org)
* [Patroni](https://patroni.readthedocs.io/en/latest/)
* [HAProxy](https://www.haproxy.org/)


### Kafka/Messaging stack

Spin up a masaging stack zookeeper cluster + kafka cluster + schema registry + kafka connect , and setup messaging stack with ansible.

Note: All packages are from CentOS 7 (Conluent 5.5.1)


#### Check zookeeper cluster
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

#### Check kafka brokers:
```
[root@dl360p35 saas]# ssh -p 2260 -i .vagrant/machines/kafka01/virtualbox/private_key vagrant@127.0.0.1 'sudo grep started /var/log/kafka/server.log'
[2020-10-01 14:21:23,131] INFO [KafkaServer id=0] started (kafka.server.KafkaServer)
[root@dl360p35 saas]# ssh -p 2261 -i .vagrant/machines/kafka02/virtualbox/private_key vagrant@127.0.0.1 'sudo grep started /var/log/kafka/server.log'
[2020-10-01 14:24:08,484] INFO [KafkaServer id=1] started (kafka.server.KafkaServer)
[root@dl360p35 saas]# ssh -p 2262 -i .vagrant/machines/kafka03/virtualbox/private_key vagrant@127.0.0.1 'sudo grep started /var/log/kafka/server.log'
[2020-10-01 14:27:00,324] INFO [KafkaServer id=2] started (kafka.server.KafkaServer)
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

#### Check schema_registry

```
[root@schemaregistry ~]# netstat -antlp|grep 8081
tcp6       0      0 :::8081                 :::*                    LISTEN      6522/java

[root@schemaregistry ~]# ps -ef|grep java
cp-sche+  6522     1  8 13:05 ?        00:00:08 java -Xmx512M -server -XX:+UseG1GC -XX:MaxGCPauseMillis=20 -XX:InitiatingHeapOccupancyPercent=35 -XX:+ExplicitGCInvokesConcurrent -Djava.awt.headless=true -Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dschema-registry.log.dir=/var/log/confluent/schema-registry -Dlog4j.configuration=file:/etc/schema-registry/log4j.properties -cp :/usr/bin/../package-schema-registry/target/kafka-schema-registry-package-*-development/share/java/schema-registry/*:/usr/bin/../share/java/confluent-security/schema-registry/*:/usr/bin/../share/java/confluent-common/*:/usr/bin/../share/java/rest-utils/*:/usr/bin/../share/java/schema-registry/* io.confluent.kafka.schemaregistry.rest.SchemaRegistryMain /etc/schema-registry/schema-registry.properties
root      6627  6603  0 13:07 pts/0    00:00:00 grep --color=auto java

[root@kafka01 ~]# kafka-topics --describe --topic _schemas --zookeeper zoo01:2181
Topic: _schemas	PartitionCount: 1	ReplicationFactor: 3	Configs: cleanup.policy=compact
	Topic: _schemas	Partition: 0	Leader: 0	Replicas: 0,1,2	Isr: 0,1,2
[root@kafka01 ~]# kafka-console-consumer --bootstrap-server kafka01:9092 --topic _schemas --from-beginning
null
null 
```
#### Check Kafka Connect

```
[root@kafkaconnect ~]# netstat -antlp|grep LIST|grep java
tcp6       0      0 :::36970                :::*                    LISTEN      23418/java          
tcp6       0      0 :::8083                 :::*                    LISTEN      23418/java          
[root@kafkaconnect ~]# ps -ef|grep java
cp-kafk+ 23418     1 11 18:43 ?        00:00:19 java -Xms256M -Xmx2G -server -XX:+UseG1GC -XX:MaxGCPauseMillis=20 -XX:InitiatingHeapOccupancyPercent=35 -XX:+ExplicitGCInvokesConcurrent -XX:MaxInlineLevel=15 -Djava.awt.headless=true -Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/var/log/kafka -Dlog4j.configuration=file:/etc/kafka/connect-log4j.properties -cp /usr/share/java/kafka/*:/usr/share/java/confluent-common/*:/usr/share/java/kafka-serde-tools/*:/usr/share/java/monitoring-interceptors/*:/usr/bin/../share/java/kafka/*:/usr/bin/../support-metrics-client/build/dependant-libs-2.12.10/*:/usr/bin/../support-metrics-client/build/libs/*:/usr/share/java/support-metrics-client/* org.apache.kafka.connect.cli.ConnectDistributed /etc/kafka/connect-distributed.properties
root     23547 23517  0 18:46 pts/0    00:00:00 grep --color=auto java

[root@kafka01 ~]# kafka-topics --describe --topic connect-configs --zookeeper zoo01:2181
Topic: connect-configs	PartitionCount: 1	ReplicationFactor: 1	Configs: cleanup.policy=compact
	Topic: connect-configs	Partition: 0	Leader: 0	Replicas: 0	Isr: 0

[root@kafka01 ~]# kafka-console-consumer --bootstrap-server kafka01:9092 --topic connect-configs --from-beginning
{"key":"O3QG5ox5B4aW35mQvmLBeMR4WxRWa7lNSeKDTqB2kw0=","algorithm":"HmacSHA256","creation-timestamp":1601307828626}

[root@kafka01 ~]# kafka-topics --describe --topic connect-status --zookeeper zoo01:2181
Topic: connect-status	PartitionCount: 5	ReplicationFactor: 1	Configs: cleanup.policy=compact
	Topic: connect-status	Partition: 0	Leader: 1	Replicas: 1	Isr: 1
	Topic: connect-status	Partition: 1	Leader: 0	Replicas: 0	Isr: 0
	Topic: connect-status	Partition: 2	Leader: 2	Replicas: 2	Isr: 2
	Topic: connect-status	Partition: 3	Leader: 1	Replicas: 1	Isr: 1
	Topic: connect-status	Partition: 4	Leader: 0	Replicas: 0	Isr: 0

```
### TIG monitoring stack

Spin up a TIG monitoring stack (VMs: influxdb, grafana, telegraf), and setup TIG monitoring stack with ansible.

Install telegraf on all hosts for TIG Infrastructure Monitoring : Vagrant DEV SaaS && AWS DEV/Staging/PROD SaaS && On-Prem Infrastructure 

```
1.For ansible inventory setup (Vagrant DEV SaaS) —> Example: mc_dev_hosts_vagrant add ansible_user=vagrant for all VMs:

[all:vars]
ansible_user=vagrant

Note. For AWS fix ansible inventory and add:

[all:vars]
ansible_connection: ssh
ansible_user: centos
ansible_become: true
ansible_ssh_private_key_file: /home/davar/saas.pem

For bare-metal/VMs on-prem infrastructure similar (fix inventory file —> you have to have all hosts/VMs IPs/hostnames/etc. @ansible inventory file) before deploy TIG monitoring stack. Also you can use ansible roles/playbooks from saas-dev-env repo to setup ELK, Kafka/Messaging, etc. stacks @ on-prem infrastructure.


2.Create playbook file 

[root@dl360p35 saas]# cat 00_telegraf_install_all.yml 
- name: 00_telegraf
  hosts:
    - all
  become: true
  roles:
    - tig_stack/provisioning/roles/telegraf

3.Fix telegraf role template file 

[root@dl360p35 saas]# sed -i 's/telegraf/{{ansible_hostname}}/g' tig_stack/provisioning/roles/telegraf/templates/telegraf.conf.j2 

[root@dl360p35 saas]# ansible-playbook -i mc_dev_hosts_vagrant --extra-vars "@variables" 00_telegraf_install_all.yml

PLAY RECAP *******************************************************************************************************************************************************************
dcs01                      : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
dcs02                      : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
dcs03                      : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
elasticnode1               : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
elasticnode2               : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
elasticnode3               : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
grafana                    : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
influxdb                   : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
kafka01                    : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
kafka02                    : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
kafka03                    : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
kafkaconnect               : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
kibana                     : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
logstash                   : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
pg01                       : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
pg02                       : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
pg03                       : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
schemaregistry             : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
telegraf                   : ok=6    changed=2    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
zoo01                      : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
zoo02                      : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
zoo03                      : ok=7    changed=4    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0  
```

Go to grafana (user: admin password:scale250) and import dashboards.

Example: Vagrant SaaS env: http://10.100.1.35:3000/ import Dashboards —> Example https://grafana.com/grafana/dashboards/10581

Note: To access grafana port outside Vagrant host: 

Vagrantfile:     c.vm.network "forwarded_port", adapter: 1, guest: 3000, host: 3000, id: "grafana", host_ip: '127.0.0.1'

[root@dl360p35 saas]# sysctl -w net.ipv4.conf.all.route_localnet=1

iptables -t nat -I PREROUTING -p tcp --dport 3000 -j DNAT --to 127.0.0.1:3000

All hosts has to be monitored now: 



### ELK stack

#### Check elasticsearch

Note: Elastic cluster (1 master: nodemaster=true; nodedata=true; nodeingest=true && 2 mixednodes: nodemaster=true; nodedata=true; nodeingest=true)


```
[root@kibana ~]# curl -XGET 'elasticnode1:9200/_cluster/health?pretty'
{
  "cluster_name" : "local-elk",
  "status" : "green",
  "timed_out" : false,
  "number_of_nodes" : 3,
  "number_of_data_nodes" : 2,
  "active_primary_shards" : 3,
  "active_shards" : 6,
  "relocating_shards" : 0,
  "initializing_shards" : 0,
  "unassigned_shards" : 0,
  "delayed_unassigned_shards" : 0,
  "number_of_pending_tasks" : 0,
  "number_of_in_flight_fetch" : 0,
  "task_max_waiting_in_queue_millis" : 0,
  "active_shards_percent_as_number" : 100.0
}

```
### Destroy vagrant environment
```
vagrant destroy -f
```

### References
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)

