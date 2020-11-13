
## SaaS development env (default dev env: docker-compose based)

Running the whole SaaS infrastructure using docker-compose: 

SaaS: HashiCorp Vault container + Monitoring/Debugging and Log management containers: ElasticSearch, Logstash/Fluentd, Kibana, Grafana, InfuxDB, Sensu, etc. + Kafka containers: Kafka cluster (3 containers), Kafka Replicator, Kafka Connect, Schema Registry + Consul cluster (3 containers) + Zookeper cluster (3 containers) + Postgres cluster containers + etc.

#### Note: Additional to docker-compose based (default dev env), Vagrant based (vagrant), k8s (k8s-local: minikube, kubespray) based SaaS dev environments we have to have AWS dev env.



### Example (run all SaaS services : docker-compose ):

```
./bin/run_saas_all.sh

```

#### Note1: Continuous Monitoring, Debugging and Log management (view README.md files for ELK, EFK, TIG stack:Telegraf+InfluxDB+Grafana and Prometheus+Grafana+cAdvisor+NodeExporter+AlertManager, SENSU, ZABBIX monitoring stacks for different config options)

##### Run Prometheus+Grafana+cAdvisor+NodeExporter+AlertManager monitoring stack for docker host and containers (continuous monitoring)

```
cd ./saas/docker/monitoring/prometheus-grafana-nodeexporter; ADMIN_USER=admin ADMIN_PASSWORD=admin docker-compose up -d
Creating network "prometheus-grafana-nodeexporter_monitor-net" with driver "bridge"
Creating nodeexporter ... done
Creating caddy        ... done
Creating cadvisor     ... done
Creating grafana      ... done
Creating pushgateway  ... done
Creating alertmanager ... done
Creating prometheus   ... done

```

##### Run TIG :Telegraf+InfluxDB+Grafana  monitoring stack for docker host and containers (continuous monitoring)

```
cd ./saas/docker/monitoring/TIG; docker-compose up -d
Creating network "tig-stack_backend" with the default driver
Creating network "tig-stack_frontend" with the default driver
Creating influxdb ... done
Creating telegraf ... done
Creating grafana  ... done
```

##### Run SENSU monitoring stack for docker host and containers (continuous monitoring)
```
cd ./saas/docker/monitoring/SENSU; docker-compose up -d
Creating sensu_rabbitmq_1 ... 
Creating sensu_rabbitmq_1 ... done
Creating sensu_sensu-server_1 ... 
Creating sensu_sensu-server_1 ... done
Creating sensu_uchiwa_1 ... 
Creating sensu_sensu-client_1 ... 
Creating sensu_uchiwa_1
Creating sensu_uchiwa_1 ... done
```

##### Run ZABBIX monitoring stack (continuous monitoring)

```
cd ./saas/docker/monitoring/ZABBIX; docker-compose up -d

docker-compose up -d
Creating network "zabbix_zabbix" with driver "bridge"
Creating mysql ... done
Creating zabbix-frontend ... done
Creating zabbix-server   ... done
Creating grafana         ... done
Creating zabbix-agent    ... done
```

##### Run ELK stack (debugging and log management)

```
cd ./saas/docker/monitoring/ELK ; docker-compose up -d
Creating elk_elasticsearch_1 ... done
Creating elk_kibana_1 ... 
Creating elk_logstash_1 ... done
Creating filebeat ... done

```

##### Run EFK stack (debugging and log management)

```
Creating network "docker_default" with the default driver
Creating docker_elasticsearch_1 ... done
Creating docker_fluentd_1       ... done
Creating docker_kibana_1        ... done
```

##### Stop stacks

```
cd ./saas/docker/monitoring/ELK ; docker-compose down
cd ./saas/docker/monitoring/TIG; docker-compose down
cd ./saas/docker/monitoring/SENSU; docker-compose down
cd ./saas/docker/monitoring/prometheus-grafana-nodeexporter; ADMIN_USER=admin ADMIN_PASSWORD=admin docker-compose down
...
...
```

#### Note2: Messaging: Kafka stack, etc. (view README.md file for different config options)

```
cd ./saas/docker/messaging; docker-compose up -d
```
#### Note3: PostgreSQL HA using Patroni with Zookeeper and HAProxy (view README.md file for different config options)

```
cd ./saas/docker/postgresqlha; docker-compose up -d

```


Example2: (run all SaaS services)
```
./bin/run_saas_all.sh
...
Creating network "docker_default" with the default driver
+ cd /root/saas-dev-env/saas/docker/monitoring/ELK/
+ docker-compose up -d
Creating network "elk_default" with the default driver
Creating elk_elasticsearch_1 ... 
Creating elk_elasticsearch_1 ... done
Creating elk_kibana_1 ... 
Creating elk_logstash_1 ... 
Creating elk_logstash_1
Creating elk_logstash_1 ... done
Creating filebeat ... 
Creating filebeat ... done
+ cd /root/saas-dev-env/saas/docker/monitoring/TIG/
+ docker-compose up -d
Creating network "tig_monitor-net" with driver "bridge"
Creating influxdb ... 
Creating influxdb ... done
Creating grafana ... 
Creating telegraf ... 
Creating grafana
Creating grafana ... done
+ cd /root/saas-dev-env/saas/docker/monitoring/SENSU/
+ docker-compose up -d
Creating network "sensu_default" with the default driver
Creating sensu_rabbitmq_1 ... 
Creating sensu_rabbitmq_1 ... done
Creating sensu_sensu-server_1 ... 
Creating sensu_sensu-server_1 ... done
Creating sensu_uchiwa_1 ... 
Creating sensu_sensu-client_1 ... 
Creating sensu_uchiwa_1
Creating sensu_uchiwa_1 ... done
+ cd /root/saas-dev-env/saas/docker/messaging/
+ docker-compose up -d
Creating network "messaging_messaging_network" with the default driver
Creating messaging_zookeeper ... 
Creating messaging_zookeeper ... done
Creating messaging_kafka ... 
Creating messaging_kafka ... done
Creating messaging_schema_registry ... 
Creating messaging_schema_registry ... done
Creating messaging_connect ... 
Creating messaging_connect ... done
+ cd /root/saas-dev-env/saas/docker/postgresqlha
+ docker-compose up -d
Creating network "postgresqlha_default" with the default driver
Creating postgresqlha_zoo1_1 ... 
Creating postgresqlha_pg-master_1 ... 
Creating postgresqlha_pg-slave_1 ... 
Creating postgresqlha_zoo1_1
Creating postgresqlha_pg-master_1
Creating postgresqlha_pg-slave_1 ... done
Creating postgresqlha_haproxy_1 ... 
Creating postgresqlha_haproxy_1 ... done

...

Clean env:
# cd ./saas/docker/monitoring/ELK ; docker-compose down
# cd ./saas/docker/monitoring/TIG; docker-compose down
# cd ./saas/docker/monitoring/SENSU; docker-compose down
# cd ./saas/docker/messaging; docker-compose down
# cd ./saas/docker/postgresqlha; docker-compose down

