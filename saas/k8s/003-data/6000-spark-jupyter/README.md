### Driver Service Account
The set of commands below will create a special service account (spark-driver) that can be used by the driver pods. It is configured to provide full administrative access to the namespace.

#### Create spark-driver service account
`kubectl create serviceaccount spark-driver`

#### Create a cluster and namespace "role-binding" to grant the account administrative privileges
`kubectl create rolebinding spark-driver-rb --clusterrole=cluster-admin --serviceaccount=default:spark-driver`

### Executor Service Account
While it is possible to have the executor reuse the spark-driver account, it's better to use a separate user account for workers. This allows for finer-grained tuning of the permissions. The worker account uses the "edit" permission, which allows for read/write access to most resources in a namespace but prevents it from modifying important details of the namespace itself.

#### Create Spark executor account
`kubectl create serviceaccount spark-minion`

#### Create rolebinding to offer "edit" privileges
`kubectl create rolebinding spark-minion-rb --clusterrole=edit --serviceaccount=default:spark-minion`

### k8s:  spark with jupyter and k8s as cluster manager 

Ref: Dockerfile: COPY --from=deps /tmp/spark-3.0.1-bin-hadoop3.2/kubernetes/dockerfiles/spark/entrypoint.sh /opt/
```
kubectl apply -f *.yaml
```
