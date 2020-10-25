#### Note: view README.md files for ELK , EFK, TIG: Telegraf+InfluxDB+Grafana or Prometheus+Grafana+cAdvisor+NodeExporter+AlertManager, SENSU, ZABBIX for different config options

##### Run Prometheus + Grafana + cAdvisor + NodeExporter + AlertManager monitoring stack for docker host and containers (continuous monitoring)

```
cd ./prometheus-grafana-nodeexporter; ADMIN_USER=admin ADMIN_PASSWORD=admin docker-compose up -d

```
##### Run TIG: Telegraf+InfluxDB+Grafana monitoring stack for docker host and containers (continuous monitoring)

```
cd ./TIG; docker-compose up -d
```
##### Run SENSU for docker host and containers (continuous monitoring)

```
cd ./SENSU; docker-compose up -d
```

##### Run ZABBIX (continuous monitoring)

```
cd ./ZABBIX; docker-compose up -d
```

##### Run ELK stack (debugging and log management)

```
cd ./ELK ; docker-compose up -d

```
##### Run EFK stack (debugging and log management): view README.md file



##### Stop monitoring/debugging and log management stacks

```
cd ./ELK ; docker-compose down
cd ./TIG; docker-compose down
cd ./SENSU; docker-compose down
cd ./prometheus-grafana-nodeexporter; ADMIN_USER=admin ADMIN_PASSWORD=admin docker-compose down
...
...

```

Note1: Monitoring stack (Prometheus+Grafana+cAdvisor+NodeExporter+AlertManager+Pushgateway etc.) has similar functionality as proposed SAAS PRD v1 monitoring stack:  Sensu+OpsGenie+Telegraf+InfluxDB+Grafana. 

Note2: Full monitoring stack InfluxDB+Telegraf+Grafana+Atlassian OpsGenie, can’t be implemented for saas-dev-env-docker, because Atlassian OpsGenie is not open-source and there is no public repo with OpsGenie source. This product is not free. So only TIG stack for now (without Atlassian OpsGenie).

#### Run EFK (debugging and log management), TIG + SENSU (continuous monitoring) : View README files and SAAS PRD/TRD v1.0.0

Screens -> 

Grafana: 

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/monitoring/screens/noc-grafana-system.png?raw=true" width="650">

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/monitoring/screens/noc-grafana-docker.png?raw=true" width="650">

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/monitoring/screens/noc-grafana-docker-2.png?raw=true" width="650">


Kibana: 

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/monitoring/screens/noc-kibana.png?raw=true" width="650">

Sensu: 

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/monitoring/screens/noc-sensu-example.png?raw=true" width="650">



#### TODO:

TODO1: Research Kafka stack/Kafka monitoring stack: Kafka+Kafka Replicator+Kafka Connect+Schema Registry+Zookeeper+Elasticsearch+Kibana+Grafana. Kafka, Kafka Replicator, Kafka Connect, Schema Registry, Consul are not implemented currently into SAAS SaaS architecture (engine code, installer, agent, orchestrator, etc.) so will not have InfluxDB/Elasticsearch with Kafka data (via Kafka Connect),  and Telegraf will not collect & forward metrics & logs to Kafka.


TODO2: Research other solutions:

Open-source continuous monitoring: 
```
1.Sensu: redis_server, rabbitmq_server,  sensu_master , dashboard, plugins, clients
2.Zabbix: mysql/posgresql, zabbix_master, zabbix_clients 
3.Icinga2: Nagios fork, good API, modules and integration 
4.Prometheus+Grafana+NodeExporter+AlertManager+Graphite+StatsD+collectd+Telegraf+InfluxDB+etc.
```
Debugging and log management:
```
1.ELK (Elasticsearch+Logstash+Kibana) or EFK(Elasticsearch+FluentD+Kibana)
2.Elasticsearch+Greylog
3.Splunk
4.New Relic, AppDinamics
```
