## Zookeeper cluster 
======================================

### Overview

This repo it helps you quickly spin up a zookeeper cluster, and setup zoo cluster stack with ansible.

Note: All packages are from CentOS 7


### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started
`yum install ansible`
`vagrant up`

```
root@dl360p30 zookeeper]# vagrant status
Current machine states:

zoo01              running (virtualbox)
zoo02              running (virtualbox)
zoo03              running (virtualbox)
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

### References
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)

