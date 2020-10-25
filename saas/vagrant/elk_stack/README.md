
## ELK stack 
======================================

### Overview

This repo it helps you quickly spin up a ELK stack, and setup ELK stack with ansible.

Note1: All packages are from CentOS 7

Note2: Elastic cluster (1 master: nodemaster=true; nodedata=true; nodeingest=true && 2 mixednodes: nodemaster=true; nodedata=true; nodeingest=true)

### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started
`yum install ansible`
`vagrant up`

```
root@dl360p30 ELK]# vagrant status
Current machine states:

elasticnode1              running (virtualbox)
elasticnode2              running (virtualbox)
elasticnode3              running (virtualbox)
kibana                    running (virtualbox)
logstash                  running (virtualbox)
```

### Check elasticsearch
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

### References
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)
