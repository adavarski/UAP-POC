Setup of PostgreSQL High Available using Patroni with Zookeeper and HAProxy.

## How-to
Simply start using docker-compose (docker-compose up -d). Initially the zookeeper container will start and than patroni will get
the cluster working.

## Reference

The set-up is based on the patroni documentation which can be found on [github](https://github.com/zalando/patroni) and the
docker image provided by PostgreSQL.

HAproxy uses the default haproxy docker image but it is still build to have the config file
inside the image for convenience. You can see the stats on http://localhost:7000

Default PostgreSQL port exposed on `localhost:5432`. (login: _postgres/grespost_)

Test with DataGrip: 

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/postgresqlha/screens/pg-datagrip-test.png?raw=true" width="650">


Check HA:

```
# docker-compose ps
          Name                        Command               State                       Ports                      
------------------------------------------------------------------------------------------------------------------
postgresqlha_haproxy_1     /docker-entrypoint.sh hapr ...   Up      0.0.0.0:5432->5432/tcp, 0.0.0.0:7000->7000/tcp 
postgresqlha_pg-master_1   docker-entrypoint.sh su -  ...   Up      0.0.0.0:5433->5432/tcp, 8008/tcp               
postgresqlha_pg-slave_1    docker-entrypoint.sh su -  ...   Up      0.0.0.0:5434->5432/tcp, 8008/tcp               
postgresqlha_zoo1_1        /etc/confluent/docker/run        Up      0.0.0.0:2181->2181/tcp, 2888/tcp, 3888/tcp     
# docker exec -it postgresqlha_pg-master_1 bash
root@pg-master:/# psql -h localhost -U postgres
psql (10.13 (Debian 10.13-1.pgdg90+1))
Type "help" for help.

postgres=# select * from pg_stat_replication;
 pid | usesysid |  usename   | application_name | client_addr | client_hostname | client_port |         backend_start         | backend_xmin |   state   | sent_lsn  | write_ls
n | flush_lsn | replay_lsn | write_lag | flush_lag | replay_lag | sync_priority | sync_state 
-----+----------+------------+------------------+-------------+-----------------+-------------+-------------------------------+--------------+-----------+-----------+---------
--+-----------+------------+-----------+-----------+------------+---------------+------------
  73 |    16384 | replicator | postgresql1      | 10.20.17.3  |                 |       34840 | 2020-07-30 07:55:03.650031+00 |              | streaming | 0/3000548 | 0/300054
8 | 0/3000548 | 0/3000548  |           |           |            |             1 | sync
(1 row)

postgres=# \q
root@pg-master:/# exit
# docker exec -it postgresqlha_pg-slave_1 bash
root@pg-slave:/# psql -h localhost -U postgres
psql (10.13 (Debian 10.13-1.pgdg90+1))
Type "help" for help.

postgres=# select * from pg_stat_wal_receiver;
 pid |  status   | receive_start_lsn | receive_start_tli | received_lsn | received_tli |      last_msg_send_time       |     last_msg_receipt_time     | latest_end_lsn |      
  latest_end_time        |  slot_name  |                                                                                                                      conninfo         
                                                                                                              
-----+-----------+-------------------+-------------------+--------------+--------------+-------------------------------+-------------------------------+----------------+------
-------------------------+-------------+---------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
  38 | streaming | 0/3000000         |                 1 | 0/3000628    |            1 | 2020-07-30 08:00:11.260164+00 | 2020-07-30 08:00:11.260264+00 | 0/3000628      | 2020-
07-30 08:00:11.260164+00 | postgresql1 | user=replicator passfile=/tmp/pgpass1 dbname=replication host=pg-master port=5432 application_name=postgresql1 fallback_application_na
me=walreceiver sslmode=prefer sslcompression=0 gssencmode=prefer krbsrvname=postgres target_session_attrs=any
(1 row)

```
Check HAProxy:

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/postgresqlha/screens/haproxy-pg-master.png?raw=true" width="650">


Stop/start master

```
# docker-compose stop pg-master
Stopping postgresqlha_pg-master_1 ... done
# docker-compose start pg-master
Starting pg-master ... done
```
Check new master and new slave

```
[root@dl360p31 postgresqlha]# docker exec -it postgresqlha_pg-slave_1 bash
root@pg-slave:/# psql -h localhost -U postgres
psql (10.13 (Debian 10.13-1.pgdg90+1))
Type "help" for help.

postgres=# select * from pg_stat_replication;
 pid | usesysid |  usename   | application_name | client_addr | client_hostname | client_port |         backend_start         | backend_xmin |   state   | sent_lsn  | write_ls
n | flush_lsn | replay_lsn | write_lag | flush_lag | replay_lag | sync_priority | sync_state 
-----+----------+------------+------------------+-------------+-----------------+-------------+-------------------------------+--------------+-----------+-----------+---------
--+-----------+------------+-----------+-----------+------------+---------------+------------
 267 |    16384 | replicator | postgresql0      | 10.20.17.4  |                 |       47158 | 2020-07-30 08:03:45.329084+00 |              | streaming | 0/4000288 | 0/400028
8 | 0/4000288 | 0/4000288  |           |           |            |             1 | sync
(1 row)

postgres=# \q
root@pg-slave:/# exit
exit
[root@dl360p31 postgresqlha]# docker exec -it postgresqlha_pg-master_1 bash
root@pg-master:/# psql -h localhost -U postgres
psql (10.13 (Debian 10.13-1.pgdg90+1))
Type "help" for help.

postgres=# select * from pg_stat_wal_receiver;
 pid |  status   | receive_start_lsn | receive_start_tli | received_lsn | received_tli |      last_msg_send_time       |     last_msg_receipt_time     | latest_end_lsn |      
  latest_end_time        |  slot_name  |                                                 conninfo                                                  
-----+-----------+-------------------+-------------------+--------------+--------------+-------------------------------+-------------------------------+----------------+------
-------------------------+-------------+-----------------------------------------------------------------------------------------------------------
  36 | streaming | 0/4000000         |                 2 | 0/4000288    |            2 | 2020-07-30 08:04:45.455489+00 | 2020-07-30 08:04:45.455548+00 | 0/4000288      | 2020-
07-30 08:03:45.333488+00 | postgresql0 | user=replicator passfile=/tmp/pgpass0 host=pg-slave port=5432 sslmode=prefer application_name=postgresql0
(1 row)

postgres=# \q
root@pg-master:/# exit
```
Check HAproxy:

<img src="https://github.com/SAASInc/saas-dev-env/blob/master/saas/docker/postgresqlha/screens/haproxy-pg-slave.png?raw=true" width="650">


### Note: Setup of PostgreSQL High Available using Patroni with Zookeeper cluster (3-nodes) and HAProxy 
```
# mv postgres postgres.ORIG
# mv postgres-ha postgres
# docker-compose -f ./docker-compose.yml.ha up -d
# docker-compose -f docker-compose.yaml.ha ps
          Name                        Command               State                          Ports                         
------------------------------------------------------------------------------------------------------------------------
postgresqlha_haproxy_1     /docker-entrypoint.sh hapr ...   Up      0.0.0.0:5432->5432/tcp, 0.0.0.0:7000->7000/tcp       
postgresqlha_pg-master_1   docker-entrypoint.sh su -  ...   Up      0.0.0.0:5433->5432/tcp, 8008/tcp                     
postgresqlha_pg-slave_1    docker-entrypoint.sh su -  ...   Up      0.0.0.0:5434->5432/tcp, 8008/tcp                     
zookeeper1                 /etc/confluent/docker/run        Up      0.0.0.0:2181->2181/tcp, 2888/tcp, 3888/tcp           
zookeeper2                 /etc/confluent/docker/run        Up      2181/tcp, 0.0.0.0:2182->2182/tcp, 2888/tcp, 3888/tcp 
zookeeper3                 /etc/confluent/docker/run        Up      2181/tcp, 0.0.0.0:2183->2183/tcp, 2888/tcp, 3888/tcp 
```
### Note: Setup of PostgreSQL High Available using Patroni with Consul and HAProxy

```
# mv postgres postgres.ORIG
# mv postgres-consul postgres
# docker-compose -f ./docker-compose.yml.consul up -d
# docker-compose -f docker-compose.yaml.consul ps
# docker-compose ps
                   Name                                       Command                                       State                                        Ports                    
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
consul1                                      docker-entrypoint.sh agent ...               Up                                           0.0.0.0:53->53/udp,                        
                                                                                                                                       0.0.0.0:8300->8300/tcp,                    
                                                                                                                                       0.0.0.0:8301->8301/tcp,                    
                                                                                                                                       0.0.0.0:8301->8301/udp,                    
                                                                                                                                       0.0.0.0:8302->8302/tcp,                    
                                                                                                                                       0.0.0.0:8302->8302/udp,                    
                                                                                                                                       0.0.0.0:8400->8400/tcp,                    
                                                                                                                                       0.0.0.0:8500->8500/tcp, 8600/tcp, 8600/udp 
postgresqlha_haproxy_1                       /docker-entrypoint.sh hapr ...               Up                                           0.0.0.0:5432->5432/tcp,                    
                                                                                                                                       0.0.0.0:7000->7000/tcp                     
postgresqlha_pg-master_1                     docker-entrypoint.sh su -  ...               Up                                           0.0.0.0:5433->5432/tcp, 8008/tcp           
postgresqlha_pg-slave_1                      docker-entrypoint.sh su -  ...               Up                                           0.0.0.0:5434->5432/tcp, 8008/tcp   
```
