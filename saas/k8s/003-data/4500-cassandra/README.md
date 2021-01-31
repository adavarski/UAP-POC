### Apache Cassandra on Kubernetes 

#### Install Helm repositories
```
$ helm repo add datastax https://datastax.github.io/charts
"datastax" has been added to your repositories
```

#### Install cass-operator (data ns)
```
$ helm install cass-operator datastax/cass-operator -n data
NAME: cass-operator
LAST DEPLOYED: Sun Jan 31 10:43:25 2021
NAMESPACE: data
STATUS: deployed
REVISION: 1
TEST SUITE: None
```
#### Provision Cassandra cluster (data ns)
```
$ kubectl apply -f cluster.yaml
cassandradatacenter.cassandra.datastax.com/dc1 created

```
#### Check k8s resources and cqlsh CLI:
```
$ kubectl get po -n data|grep cluster
cluster1-dc1-default-sts-0          2/2     Running            0          62s


$ kubectl get all -n data|grep cass
pod/cass-operator-885f948c6-pz7zl       1/1     Running            0          3m27s
service/cassandradatacenter-webhook-service   ClusterIP   10.43.240.154   <none>        443/TCP                                        3m27s
service/cass-operator-metrics                 ClusterIP   10.43.125.158   <none>        8383/TCP,8686/TCP                              3m19s
deployment.apps/cass-operator       1/1     1            1           3m27s
replicaset.apps/cass-operator-885f948c6       1         1         1       3m27s

$ kubectl -n data describe cassdc dc1

$ kubectl -n data get secret cluster1-superuser -o yaml|grep username|grep -v {}
  username: Y2x1c3RlcjEtc3VwZXJ1c2Vy
$ kubectl -n data get secret cluster1-superuser -o yaml|grep password|grep -v {}
  password: WVFidllCSm1RZXM3NEJseDNXZmNERHJ5djFYX2F2YnRmaUtzOUtvZlRZWGc0UzVGVFZ4bjdn
  
$ kubectl exec -it cluster1-dc1-default-sts-0 -c cassandra bash -n data
root@cluster1-dc1-default-sts-0:/# echo Y2x1c3RlcjEtc3VwZXJ1c2Vy|base64 -d
cluster1-superuser

root@cluster1-dc1-default-sts-0:/# echo 'WVFidllCSm1RZXM3NEJseDNXZmNERHJ5djFYX2F2YnRmaUtzOUtvZlRZWGc0UzVGVFZ4bjdn'|base64 -d
YQbvYBJmQes74Blx3WfcDDryv1X_avbtfiKs9KofTYXg4S5FTVxn7g

root@cluster1-dc1-default-sts-0:/# cqlsh localhost -u cluster1-superuser -p YQbvYBJmQes74Blx3WfcDDryv1X_avbtfiKs9KofTYXg4S5FTVxn7g
Connected to cluster1 at localhost:9042.
[cqlsh 5.0.1 | Cassandra 3.11.7 | CQL spec 3.4.4 | Native protocol v4]
Use HELP for help.
cluster1-superuser@cqlsh>   
  
```
### Clean/Tear down resources:
```
$ kubectl delete -f cluster.yaml 
cassandradatacenter.cassandra.datastax.com "dc1" deleted
$ helm delete cass-operator -n data
release "cass-operator" uninstalled
```
