vagrant-postgresql-patroni-zookeeper-ha-setup
======================================

### Overview

This repo it helps you quickly spin up a 3-node cluster of PostgreSQL, managed by Patroni using Zookeeper cluster (3-nodes).

### What's in the cluster?

When you start the cluster, you get 3 nodes (pg01, pg02 and pg03), each running:

  - PostgreSQL
  - Patroni

All packages are from CentOS 7, except for PostgreSQL itself, which is at version 11.x.

The cluster is configured with a single primary and two asynchronous streaming replica.

### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started

`vagrant up`

```
# vagrant status 

zoo01                     running (virtualbox)
zoo02                     running (virtualbox)
zoo03                     running (virtualbox)
pg01                      running (virtualbox)
pg02                      running (virtualbox)
pg03                      running (virtualbox)

```
### Test HA

```
# vagrant ssh pg01
Last login: Sat Sep 26 10:42:22 2020 from 10.0.2.2
[vagrant@pg01 ~]$ sudo su -
[root@pg01 ~]# echo stat | nc zoo01 2181 | grep Mode
Mode: follower
[root@pg01 ~]# echo stat | nc zoo02 2181 | grep Mode
Mode: leader
[root@pg01 ~]# echo stat | nc zoo03 2181 | grep Mode
Mode: follower
[root@pg01 ~]# echo mntr | nc zoo02 2181
zk_version	3.4.13-2d71af4dbe22557fda74f9a9b4309b15a7487f03, built on 06/29/2018 04:05 GMT
zk_avg_latency	0
zk_max_latency	0
zk_min_latency	0
zk_packets_received	2
zk_packets_sent	1
zk_num_alive_connections	1
zk_outstanding_requests	0
zk_server_state	leader
zk_znode_count	16
zk_watch_count	0
zk_ephemerals_count	4
zk_approximate_data_size	1560
zk_open_file_descriptor_count	29
zk_max_file_descriptor_count	4096
zk_fsync_threshold_exceed_count	0
zk_followers	2
zk_synced_followers	2
zk_pending_syncs	0
zk_last_proposal_size	259
zk_max_proposal_size	624
zk_min_proposal_size	32


[root@pg01 ~]# patronictl -c /etc/patroni.yml list
+ Cluster: saas (6876698412528044821) --+----+-----------+-----------------+
| Member | Host        | Role    | State   | TL | Lag in MB | Pending restart |
+--------+-------------+---------+---------+----+-----------+-----------------+
| pg01   | 10.51.21.65 | Leader  | running |  2 |           | *               |
| pg02   | 10.51.21.66 | Replica | running |  2 |       0.0 |                 |
| pg03   | 10.51.21.67 | Replica | running |  2 |       0.0 |                 |
+--------+-------------+---------+---------+----+-----------+-----------------+
[root@pg01 ~]# exit

# vagrant halt pg01
==> pg01: Attempting graceful shutdown of VM...
# vagrant halt pg02
==> pg02: Attempting graceful shutdown of VM...
# vagrant up pg02
Bringing machine 'pg02' up with 'virtualbox' provider...


# vagrant ssh pg02
[root@pg02 ~]# patronictl -c /etc/patroni.yml list
+ Cluster: saas (6876698412528044821) --+----+-----------+
| Member | Host        | Role    | State   | TL | Lag in MB |
+--------+-------------+---------+---------+----+-----------+
| pg02   | 10.51.21.66 | Replica | running |  4 |       0.0 |
| pg03   | 10.51.21.67 | Leader  | running |  4 |           |
+--------+-------------+---------+---------+----+-----------+
[root@pg02 ~]# exit

# vagrant status 
Current machine states:

zoo01                     running (virtualbox)
zoo02                     running (virtualbox)
zoo03                     running (virtualbox)
pg01                      poweroff (virtualbox)
pg02                      running (virtualbox)
pg03                      running (virtualbox)

# vagrant up pg01
Bringing machine 'pg01' up with 'virtualbox' provider...

# vagrant ssh pg01
[root@pg01 ~]# patronictl -c /etc/patroni.yml list
+ Cluster: saas (6876698412528044821) --+----+-----------+
| Member | Host        | Role    | State   | TL | Lag in MB |
+--------+-------------+---------+---------+----+-----------+
| pg01   | 10.51.21.65 | Replica | running |  4 |       0.0 |
| pg02   | 10.51.21.66 | Replica | running |  4 |       0.0 |
| pg03   | 10.51.21.67 | Leader  | running |  4 |           |
+--------+-------------+---------+---------+----+-----------+
```

### Further reading

* [PostgreSQL and Patroni cluster](https://www.linode.com/docs/databases/postgresql/create-a-highly-available-postgresql-cluster-using-patroni-and-haproxy/#before-you-begin)

### References
* [PostgreSQL](https://www.postgresql.org)
* [Patroni](https://patroni.readthedocs.io/en/latest/)
* [HAProxy](https://www.haproxy.org/)
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)
