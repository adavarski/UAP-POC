vagrant-postgresql-patroni-consul-setup
======================================

### Overview

This repo it helps you quickly spin up a 3-node cluster of PostgreSQL, managed by Patroni using Consul cluster 3-nodes.

### What's in the cluster?

When you start the cluster, you get 3 nodes (pg01, pg02 and pg03), each running:

  - PostgreSQL
  - Patroni
  - Consul agent

All packages are from CentOS 7, except for PostgreSQL itself, which is at version 11.x.

The cluster is configured with a single primary and two asynchronous streaming replica.

### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started

`vagrant up`

### Test HA

```

[vagrant@dcs01 ~]$ consul members
Node   Address           Status  Type    Build  Protocol  DC      Segment
dcs01  10.51.21.61:8301  alive   server  1.6.1  2         noc-mc  <all>
dcs02  10.51.21.62:8301  alive   server  1.6.1  2         noc-mc  <all>
dcs03  10.51.21.63:8301  alive   server  1.6.1  2         noc-mc  <all>
pg01   10.51.21.65:8301  alive   client  1.6.1  2         noc-mc  <default>
pg02   10.51.21.66:8301  alive   client  1.6.1  2         noc-mc  <default>
pg03   10.51.21.67:8301  alive   client  1.6.1  2         noc-mc  <default>

[vagrant@pg01 ~]$ patronictl -c /etc/patroni.yml list
+ Cluster: saas (6873175265125204662) --+----+-----------+-----------------+
| Member | Host        | Role    | State   | TL | Lag in MB | Pending restart |
+--------+-------------+---------+---------+----+-----------+-----------------+
| pg01   | 10.51.21.65 | Leader  | running |  2 |           | *               |
| pg02   | 10.51.21.66 | Replica | running |  2 |       0.0 |                 |
| pg03   | 10.51.21.67 | Replica | running |  2 |       0.0 |                 |
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

### Further reading

* [PostgreSQL and Patroni cluster](https://www.linode.com/docs/databases/postgresql/create-a-highly-available-postgresql-cluster-using-patroni-and-haproxy/#before-you-begin)

### References
* [PostgreSQL](https://www.postgresql.org)
* [Patroni](https://patroni.readthedocs.io/en/latest/)
* [HAProxy](https://www.haproxy.org/)
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)

