
# k8s-based data-driven Analytics/ML/DeepML SaaS platform for data analyst/data engineer/data scientist playground (MVP/POC/environmints)


### Used stacks and products
VPN (WireGuard, k8s:Kilo); Monitoring stacks (Prometheus-based, TIG:Telegraf+InfluxDB+Grafana, Sensu, Zabbix); Indexing and Analytics/Debugging and Log management stacks (ELK/EFK); Pipeline: Messaging/Kafka stack (Kafka cluster, Zookeper cluster, Kafka Replicator, Kafka Connect, Schema Registry); Routing and Transformation (Serverless:OpenFaaS; ETL:Apache NiFi); Data Lake/Big Data (MinIO s3-compatable Object Storage); DWHs (HIVE SQL-Engine with MinIO:s3, Presto SQL query engine with Hive/Cassandra/MySql/Postgresql/etc. as data sources); Apache Spark for large-scale distributed Big Data processing and data analytics with Delta Lake (Lakehouses) and MinIO(S3); Machine Learning/Deep Learning/AutoML (TensorFlow, Pandas/Koalas, Keras, Scikit-learn, Spark MLlib, etc.; k8s:Model Development with AutoML: Kubeflow(MLflow, etc.) and k8s:AI Model Deployment (Seldon Core); Spark ML with S3(MinIO) as Data Source); GitLab/Jenkins/Jenkins X/Argo CD In-Platform CI/CD (GitOps); Identity and Access Management (IAM:Keycloak); JupyterHub/JupyterLab for data science; HashiCorp Vault cluster; k8s Persistent Volumes (Rook Ceph, Gluster); etc.

Summary: k8s-based Analytics/ML/DeepML SaaS using Big Data/Data Lake: MinIO (s3-compatable object storage) with Hive(s3) SQL-Engine/DWHs (instead of Snowflake as big data platform for example), Apache Spark(Hive for metadata)/Delta Lake(lakehouses)/Jupyter/etc. (instead of Databricks for example) + Kafka stack + ELK/EFK + Serverless(OpenFaaS) + ETL(Apache NiFi) + ML/DeepML/AutoML + GitOps. For building Analytics/ML SaaS platform we can also use cloud-native SaaSs as reference (or build SaaS based on cloud-native SaaSs): Snowflake(SaaS/DWaaS) as big data solution on a single data platform (DWH, S3, etc.) + Databricks(cloud-based big data processing SaaS) + AWS S3/MKS/SQS/ELK/Lambda/etc.

### PaaS/SaaS MVP/POC/Development environments used:

- k8s (k8s-local: k3s, minikube, kubespray) `Note: Default development environment: k3s` 
- k8s AWS (KOPS for k8s clusters deploy on AWS, and k8s Operators/Helm Charts/YAML manifests for creating k8s deployments (PasS&SaaS services).  

### PaaS/SaaS objectives:

Platform as a Service (PaaS) will be data-driven and data-science platform allowing end user to develop, run, and manage applications without the complexity of building and maintaining the infrastructure.

Software as a Service (SaaS) will be "on-demand software", accessed/used by end users using a web browser.

# Local & AWS SaaS development k8s environment

## AWS (build k8s cluster with KOPS)

KOPS is based on Terraform and is working very well for AWS k8s deployments. After AWS k8s cluster has been deployed, you can use [003-data/](https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/003-data) as base or Helm Charts or k8s Operators for SaaS services deployment @ k8s cluster (create Helm Charts: Consul cluster, Kafka cluster, Elasticsearch cluster, etc. based on stable Helm charts for all needed SaaS services: Kafka, ELK, Postgres, Consul, Grafana, Sensu, InfluxDB, etc., etc. →  Ref:https://artifacthub.io/; https://github.com/artifacthub/hub https://github.com/helm/charts (obsolete) ; etc. Better is to create k8s Operators for all needed SaaS services (ref: https://github.com/adavarski/k8s-operators-playground) than Helm Charts, based on https://github.com/operator-framework/community-operators. There are many k8s Operators @ https://operatorhub.io/ for example https://operatorhub.io/operator/postgres-operator, https://operatorhub.io/operator/elastic-cloud-eck, https://operatorhub.io/operator/banzaicloud-kafka-operator, etc. so create own based on them. 

## Local k8s development  (minikube, kubespray, k3s, etc.) 

For setting up Kubernetes local development environment, there are three recommended methods

    - k3s (default) https://k3s.io/
    - minikube https://minikube.sigs.k8s.io/docs/
    - kubespary https://kubespray.io/#/

Note: Of the three (k3s & minikube & kubespay), k3s tends to be the most viable. It is closer to a production style deployment. 

# k3s: (Default) k8s local development environment HOWTO 

k3s is deafult k8s developlent environment, because k3s is closer to a production style deployment, than minikube & kubespary .

k3s is 40MB binary that runs “a fully compliant production-grade Kubernetes distribution” and requires only 512MB of RAM. k3s is a great way to wrap applications that you may not want to run in a full production Cluster but would like to achieve greater uniformity in systems deployment, monitoring, and management across all development operations.


## Prerequisite

### DNS setup

Example: Setup local DNS server

```
$ sudo apt-get install bind9 bind9utils bind9-doc dnsutils
root@carbon:/etc/bind# cat named.conf.options
options {
        directory "/var/cache/bind";
        auth-nxdomain no;    # conform to RFC1035
        listen-on-v6 { any; };
        listen-on port 53 { any; };
        allow-query { any; };
        forwarders { 8.8.8.8; };
        recursion yes;
        };
root@carbon:/etc/bind# cat named.conf.local 
//
// Do any local configuration here
//

// Consider adding the 1918 zones here, if they are not used in your
// organization
//include "/etc/bind/zones.rfc1918";
zone    "davar.com"   {
        type master;
        file    "/etc/bind/forward.davar.com";
 };

zone   "0.168.192.in-addr.arpa"        {
       type master;
       file    "/etc/bind/reverse.davar.com";
 };
root@carbon:/etc/bind# cat reverse.davar.com 
;
; BIND reverse data file for local loopback interface
;
$TTL    604800
@       IN      SOA     davar.com. root.davar.com. (
                             21         ; Serial
                         604820         ; Refresh
                          864500        ; Retry
                        2419270         ; Expire
                         604880 )       ; Negative Cache TTL

;Your Name Server Info
@       IN      NS      primary.davar.com.
primary IN      A       192.168.0.101

;Reverse Lookup for Your DNS Server
101      IN      PTR     primary.davar.com.

;PTR Record IP address to HostName
101      IN      PTR     gitlab.dev.davar.com.
101      IN      PTR     reg.gitlab.dev.davar.com.
101      IN      PTR     dev-k3s.davar.com.


root@carbon:/etc/bind# cat forward.davar.com 
;
; BIND data file for local loopback interface
;
$TTL    604800

@       IN      SOA     primary.davar.com. root.primary.davar.com. (
                              6         ; Serial
                         604820         ; Refresh
                          86600         ; Retry
                        2419600         ; Expire
                         604600 )       ; Negative Cache TTL

;Name Server Information
@       IN      NS      primary.davar.com.

;IP address of Your Domain Name Server(DNS)
primary IN       A      192.168.0.101

;Mail Server MX (Mail exchanger) Record
davar.local. IN  MX  10  mail.davar.com.

;A Record for Host names
gitlab.dev     IN       A       192.168.0.101
reg.gitlab.dev IN       A       192.168.0.101
dev-k3s        IN       A       192.168.0.101
stats.eth      IN       A       192.168.0.101
kib.data       IN       A       192.168.0.101
nifi.data      IN       A       192.168.0.101
faas.data      IN       A       192.168.0.101
minio.data     IN       A       192.168.0.101
mlflow.data    IN       A       192.168.0.101
quality.data   IN       A       192.168.0.101
spark.data     IN       A       192.168.0.101
jupyter.data   IN       A       192.168.0.101


;CNAME Record
www     IN      CNAME    www.davar.com.

$ sudo systemctl restart bind9
$ sudo systemctl enable bind9

root@carbon:/etc/bind# ping -c1 gitlab.dev.davar.com
PING gitlab.dev.davar.com (192.168.0.101) 56(84) bytes of data.
64 bytes from primary.davar.com (192.168.0.101): icmp_seq=1 ttl=64 time=0.030 ms

--- gitlab.dev.davar.com ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.030/0.030/0.030/0.000 ms
root@carbon:/etc/bind# nslookup gitlab.dev.davar.com
Server:		192.168.0.101
Address:	192.168.0.101#53

Name:	gitlab.dev.davar.com
Address: 192.168.0.101

$ sudo apt install resolvconf
$ cat /etc/resolvconf/resolv.conf.d/head|grep nameserver
# run "systemd-resolve --status" to see details about the actual nameservers.
nameserver 192.168.0.101
$ sudo systemctl start resolvconf.service
$ sudo systemctl enable resolvconf.service


```

### Install k3s

k3s is "Easy to install. A binary of less than 40 MB. Only 512 MB of RAM required to run." this allows us to utilized Kubernetes for managing the Gitlab application container on a single node while limited the footprint of Kubernetes itself. 

```bash
$ export K3S_CLUSTER_SECRET=$(head -c48 /dev/urandom | base64)
# copy the echoed secret
$ echo $K3S_CLUSTER_SECRET
$ curl -sfL https://get.k3s.io | sh -
```
### Remote Access with `kubectl`

From your local workstation you should be able to issue a curl command to Kubernetes:

```bash
curl --insecure https://SERVER_IP:6443/
```

The new k3s cluster should return a **401 Unauthorized** response with the following payload:

```json
{
  "kind": "Status",
  "apiVersion": "v1",
  "metadata": {

  },
  "status": "Failure",
  "message": "Unauthorized",
  "reason": "Unauthorized",
  "code": 401
}
```

k3s credentials are stored on the server at `/etc/rancher/k3s/k3s.yaml`:

Review the contents of the generated `k8s.yml` file:

```bash
cat /etc/rancher/k3s/k3s.yaml
```
The `k3s.yaml` is a Kubernetes config file used by `kubectl` and contains (1) one cluster, (3) one user and a (2) context that ties them together. `kubectl` uses contexts to determine the cluster you wish to connect to and use for access credentials. The `current-context` section is the name of the context currently selected with the `kubectl config use-context` command.

Before you being configuring [k3s] make sure `kubectl` pointed to the correct cluster: 

```
$ sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/k3s-config
$ sed -i "s/127.0.0.1/192.168.0.101/" k3s-config
$ export KUBECONFIG=~/.kube/k3s-config
$ kubectl cluster-info
Kubernetes master is running at https://127.0.0.1:6443
CoreDNS is running at https://127.0.0.1:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
Metrics-server is running at https://127.0.0.1:6443/api/v1/namespaces/kube-system/services/https:metrics-server:/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.

```

### Fix k3s CoreDNS for local development

```
$ cd k8s/utils/
$ sudo cp  /var/lib/rancher/k3s/server/manifests/coredns.yaml ./coredns-fixes.yaml
$ vi coredns-fixes.yaml 
$ sudo chown $USER: coredns-fixes.yaml 
$ sudo diff coredns-fixes.yaml /var/lib/rancher/k3s/server/manifests/coredns.yaml 
75,79d74
<     davar.com:53 {
<         errors
<         cache 30
<         forward . 192.168.0.101
<     }
$ kubectl apply -f coredns-fixes.yaml
serviceaccount/coredns unchanged
Warning: rbac.authorization.k8s.io/v1beta1 ClusterRole is deprecated in v1.17+, unavailable in v1.22+; use rbac.authorization.k8s.io/v1 ClusterRole
clusterrole.rbac.authorization.k8s.io/system:coredns unchanged
Warning: rbac.authorization.k8s.io/v1beta1 ClusterRoleBinding is deprecated in v1.17+, unavailable in v1.22+; use rbac.authorization.k8s.io/v1 ClusterRoleBinding
clusterrolebinding.rbac.authorization.k8s.io/system:coredns unchanged
configmap/coredns unchanged
deployment.apps/coredns configured
service/kube-dns unchanged
```

### Crate namespace: data

```bash
kubectl apply -f ./003-data/000-namespace/00-namespace.yml
```

### Install Cert Manager / Self-Signed Certificates

Note: Let's Encrypt will be used with Cert Manager for PRODUCTION/PUBLIC when we have internet accessble public IPs and public DNS domain. davar.com is local domain, so we use Self-Signed Certificates, Let's Encrypt is using public DNS names and if you try to use Let's Encrypt for local domain and IPs you will have issue:


```bash
# Kubernetes 1.16+
$ kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.1.0/cert-manager.yaml

# Kubernetes <1.16
$ kubectl apply --validate=false -f https://github.com/jetstack/cert-manager/releases/download/v1.1.0/cert-manager-legacy.yaml
```

Ensure that cert manager is now running:
```bash
kubectl get all -n cert-manager
```

Output:
```plain
$ kubectl get all -n cert-manager
NAME                                          READY   STATUS    RESTARTS   AGE
pod/cert-manager-cainjector-bd5f9c764-z7bh6   1/1     Running   0          13h
pod/cert-manager-webhook-5f57f59fbc-49jk7     1/1     Running   0          13h
pod/cert-manager-5597cff495-rrl52             1/1     Running   0          13h

NAME                           TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
service/cert-manager           ClusterIP   10.43.162.66   <none>        9402/TCP   13h
service/cert-manager-webhook   ClusterIP   10.43.202.9    <none>        443/TCP    13h

NAME                                      READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/cert-manager-cainjector   1/1     1            1           13h
deployment.apps/cert-manager-webhook      1/1     1            1           13h
deployment.apps/cert-manager              1/1     1            1           13h

NAME                                                DESIRED   CURRENT   READY   AGE
replicaset.apps/cert-manager-cainjector-bd5f9c764   1         1         1       13h
replicaset.apps/cert-manager-webhook-5f57f59fbc     1         1         1       13h
replicaset.apps/cert-manager-5597cff495             1         1         1       13h


```

Add a ClusterIssuer to handle the generation of Certs cluster-wide:

```bash
kubectl apply -f ./003-data/000-namespace/003-issuer.yaml
kubectl apply -f ./003-data/000-namespace/005-clusterissuer.yml
```

## PaaS/SaaS deploy 

### Monitoring

```
git clone git@github.com:coreos/kube-prometheus.git
cd kube-prometheus
# Create the namespace and CRDs, and then wait for them to be availble before creating the remaining resources
kubectl create -f manifests/setup
until kubectl get servicemonitors --all-namespaces ; do date; sleep 1; echo ""; done
kubectl create -f manifests/
$ kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090
$ kubectl --namespace monitoring port-forward svc/alertmanager-main 9093
$ kubectl --namespace monitoring port-forward svc/grafana 3000
````
Open http://localhost:3000 on a local workstation, and log in to Grafana with the default administrator credentials, username: admin, password: admin. Explore the prebuilt dashboards for monitoring many aspects of the Kubernetes cluster, including Nodes, Namespaces, and Pods.

#to teardown the stack: kubectl delete --ignore-not-found=true -f manifests/ -f manifests/setup

### Messaging

```
kubectl apply -f ./003-data/010-zookeeper/10-service.yml
kubectl apply -f ./003-data/010-zookeeper/10-service-headless.yml
kubectl apply -f ./003-data/010-zookeeper/40-statefulset.yml

  
kubectl apply -f ./003-data/020-kafka/10-service.yml  
kubectl apply -f ./003-data/020-kafka/10-service-headless.yml
kubectl apply -f ./003-data/020-kafka/40-statefulset.yml  
kubectl apply -f ./003-data/020-kafka/45-pdb.yml  
kubectl apply -f ./003-data/020-kafka/99-pod-test-client.yml
```
### ELK

```
kubectl apply -f ./003-data/050-elasticsearch/10-service.yml
kubectl apply -f ./003-data/050-elasticsearch/40-statefulset.yml

kubectl apply -f ./003-data/051-logstash/10-service.yml
kubectl apply -f ./003-data/051-logstash/30-configmap-config.yml
kubectl apply -f ./003-data/051-logstash/30-configmap-pipeline.yml
kubectl apply -f ./003-data/051-logstash/40-deployment.yml

kubectl apply -f ./003-data/052-kibana/10-service.yml
kubectl apply -f ./003-data/052-kibana/20-configmap.yml
kubectl apply -f ./003-data/052-kibana/30-deployment.yml
kubectl apply -f ./003-data/052-kibana/50-ingress.yml
```
Note: isit https://kib.data.davar.com in a web browser for Kibana UI access.

```
$ kubectl port-forward elasticsearch-0 9200:9200 -n data
$ curl http://localhost:9200/_cluster/health
```
Note: Elasticsearch is designed to shard and replicate data across a large cluster of nodes. Single-node development clusters only support a single
shard per index and are unable to replicate data because there are no other nodes available. Using curl, POST a (JSON) template to this single-node
cluster, informing Elasticsearch to configure any new indexes with one shard and zero replicas.
```
$ cat <<EOF | curl -X POST \
-H "Content-Type: application/json" \
-d @- http://localhost:9200/_template/all
{
  "index_patterns": "*",
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  }
}
EOF
```
Test Logstash(Kafka)/Elasticsearch:
```
$ kubectl -n data exec -it kafka-client-util -- \
bash -c "echo '{\"usr\": 1, \"msg\": \"Hello ES\" }' | \
kafka-console-producer --broker-list kafka:9092 \
--topic messages"
$ curl http://localhost:9200/es-cluster-messages-*/_search
# Review the default mapping generated for the new index
$ curl http://localhost:9200/es-cluster-messages-*/_mapping

```

### ETL (Routing and Transformation): Apache NiFi

```
kubectl apply -f ./003-data/060-nifi/10-service-headless.yml
kubectl apply -f ./003-data/060-nifi/40-statefulset.yml
kubectl apply -f ./003-data/060-nifi/60-ingress.yml
```

### Serverless (OpenFaas)

```
$ helm repo add openfaas https://openfaas.github.io/faas-netes/
$ helm repo update
$ helm upgrade k8s-data-openfaas –install openfaas/openfaas --namespace data --set functionNamespace=data --set exposeServices=false --set ingress.enabled=true --set generateBasicAuth=true

```
```
kubectl apply -f ./003-data/070-openfaas/50-ingress.yml
```
Note: Visit https://faas.data.davar.com in a web browser for OpenFaaS UI portal. 

```
$ echo $(kubectl -n data get secret basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode)
```
Install faas-cli 

```
$ curl -sLSf https://cli.openfaas.com | sudo sh
$ faas-cli
```

### mqtt
```
kubectl apply -f ./003-data/040-mqtt/10-service-headless.yml
kubectl apply -f ./003-data/040-mqtt/20-configmap.yml
kubectl apply -f ./003-data/040-mqtt/30-deployment.yml
```

### MinIO (simple deploy)
```
kubectl apply -f ./003-data/100-minio/40-deployment.yml
kubectl apply -f ./003-data/100-minio/50-service.yml
kubectl apply -f ./003-data/100-minio/60-service-headles.yml
kubectl apply -f ./003-data/100-minio/70-ingress.yml

```

Note: MinIO (via MinIO operator) example:

Install the operator by running the following command:
```shell script
kubectl apply -k github.com/minio/operator
```
Setup MinIO 
```
sudo mkdir /mnt/disks/minio
kubectl apply -f ./003-data/080-minio/10-LocalPV.yml
kubectl apply -f ./003-data/080-minio/40-cluster.yml
kubectl apply -f ./003-data/080-minio/50-ingress.ym
```

### MLFlow

```
mc mb minio-cluster/mlflow
kubectl apply -f ./003-data/800-mlflow/40-statefulset.yml
kubectl apply -f ./003-data/800-mlflow/50-service.yml
kubectl apply -f ./003-data/800-mlflow/60-ingress.yml
```

### Seldon Core

Create a Namespace for Seldon Core:
```
$ kubectl create namespace seldon-system
```
Install Seldon Core Operator with Helm:
```
$ helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --namespace seldon-system
```
100-sd-quality.yml Change the modelUri: value to the location of the MLflow
model configuration:

modelUri: s3://mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model

```
kubectl apply -f ./003-data/1000-seldoncore/000-sd-s3-secret.yml
kubectl apply -f ./003-data/1000-seldoncore/100-sd-quality.yml
```

### DWH: Hive SQL-Engine with MinIO DataLake (s3 Object Storage)
```
kubectl apply -f ./003-data/3000-hive/10-mysql-metadata_backend.yml
kubectl apply -f ./003-data/3000-hive/20-service.yml
kubectl apply -f ./003-data/3000-hive/30-deployment.yml
```

### DWH: Presto SQL-Engine with MinIO, Hive, MySql, Cassandra, etc
```
cd ./003-data/4000-presto/
git clone git@github.com:apk8s/presto-chart.git
helm upgrade --install presto-data --namespace data --values values.yml ./presto-chart/presto
kubectl apply -f ./50-ingress.yml
```


## Check k8s development cluster:

```
davar@carbon:~$ kubectl get svc -n data
NAME                     TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                        AGE
zookeeper                ClusterIP   10.43.215.75    <none>        2181/TCP                       14d
zookeeper-headless       ClusterIP   None            <none>        2181/TCP,3888/TCP,2888/TCP     14d
kafka                    ClusterIP   10.43.166.24    <none>        9092/TCP                       14d
kafka-headless           ClusterIP   None            <none>        9092/TCP                       14d
mqtt                     ClusterIP   10.43.133.39    <none>        1883/TCP                       14d
elasticsearch            ClusterIP   10.43.243.188   <none>        9200/TCP                       14d
logstash                 ClusterIP   10.43.69.80     <none>        5044/TCP                       14d
kibana                   ClusterIP   10.43.142.124   <none>        80/TCP                         14d
nifi                     ClusterIP   None            <none>        8080/TCP,6007/TCP              14d
gateway                  ClusterIP   10.43.169.202   <none>        8080/TCP                       14d
nats                     ClusterIP   10.43.142.201   <none>        4222/TCP                       14d
basic-auth-plugin        ClusterIP   10.43.70.145    <none>        8080/TCP                       14d
alertmanager             ClusterIP   10.43.17.222    <none>        9093/TCP                       14d
prometheus               ClusterIP   10.43.21.25     <none>        9090/TCP                       14d
sentimentanalysis        ClusterIP   10.43.190.146   <none>        8080/TCP                       14d
minio-service            ClusterIP   10.43.248.161   <none>        9000/TCP                       9d
minio-service-headless   ClusterIP   None            <none>        9000/TCP                       9d
mlflow                   ClusterIP   10.43.60.15     <none>        5000/TCP                       8d
mysql-service            ClusterIP   10.43.161.77    <none>        3306/TCP                       10h
hive                     ClusterIP   10.43.178.75    <none>        10000/TCP,9083/TCP,10002/TCP   10h
presto-data              ClusterIP   10.43.226.235   <none>        80/TCP                         52m

davar@carbon:~$ kubectl get svc --all-namespaces
NAMESPACE       NAME                      TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                        AGE
default         kubernetes                ClusterIP      10.43.0.1       <none>          443/TCP                        21d
kube-system     metrics-server            ClusterIP      10.43.139.93    <none>          443/TCP                        21d
kube-system     traefik-prometheus        ClusterIP      10.43.78.216    <none>          9100/TCP                       21d
kube-system     kube-dns                  ClusterIP      10.43.0.10      <none>          53/UDP,53/TCP,9153/TCP         21d
cert-manager    cert-manager              ClusterIP      10.43.162.66    <none>          9402/TCP                       15d
cert-manager    cert-manager-webhook      ClusterIP      10.43.202.9     <none>          443/TCP                        15d
data            zookeeper                 ClusterIP      10.43.215.75    <none>          2181/TCP                       14d
data            zookeeper-headless        ClusterIP      None            <none>          2181/TCP,3888/TCP,2888/TCP     14d
data            kafka                     ClusterIP      10.43.166.24    <none>          9092/TCP                       14d
data            kafka-headless            ClusterIP      None            <none>          9092/TCP                       14d
data            mqtt                      ClusterIP      10.43.133.39    <none>          1883/TCP                       14d
data            elasticsearch             ClusterIP      10.43.243.188   <none>          9200/TCP                       14d
data            logstash                  ClusterIP      10.43.69.80     <none>          5044/TCP                       14d
data            kibana                    ClusterIP      10.43.142.124   <none>          80/TCP                         14d
data            nifi                      ClusterIP      None            <none>          8080/TCP,6007/TCP              14d
data            gateway                   ClusterIP      10.43.169.202   <none>          8080/TCP                       14d
data            nats                      ClusterIP      10.43.142.201   <none>          4222/TCP                       14d
data            basic-auth-plugin         ClusterIP      10.43.70.145    <none>          8080/TCP                       14d
data            alertmanager              ClusterIP      10.43.17.222    <none>          9093/TCP                       14d
data            prometheus                ClusterIP      10.43.21.25     <none>          9090/TCP                       14d
data            sentimentanalysis         ClusterIP      10.43.190.146   <none>          8080/TCP                       14d
data            minio-service             ClusterIP      10.43.248.161   <none>          9000/TCP                       9d
data            minio-service-headless    ClusterIP      None            <none>          9000/TCP                       9d
data            mlflow                    ClusterIP      10.43.60.15     <none>          5000/TCP                       8d
seldon-system   seldon-webhook-service    ClusterIP      10.43.220.83    <none>          443/TCP                        7d22h
default         quality-default-quality   ClusterIP      10.43.244.50    <none>          9000/TCP,9500/TCP              7d7h
default         quality-default           ClusterIP      10.43.68.44     <none>          8000/TCP,5001/TCP              7d7h
data            mysql-service             ClusterIP      10.43.161.77    <none>          3306/TCP                       10h
data            hive                      ClusterIP      10.43.178.75    <none>          10000/TCP,9083/TCP,10002/TCP   10h
data            presto-data               ClusterIP      10.43.226.235   <none>          80/TCP                         52m
kube-system     traefik                   LoadBalancer   10.43.100.221   192.168.0.100   80:31768/TCP,443:30058/TCP     21d

davar@carbon:~$ kubectl get certificates --all-namespaces
NAMESPACE   NAME                     READY   SECRET                   AGE
data        data-production-tls      True    data-production-tls      14d
data        minio-production-tls     True    minio-production-tls     9d
data        mlflow-production-tls    True    mlflow-production-tls    8d
default     quality-production-tls   True    quality-production-tls   7d7h
data        presto-production-tls    True    presto-production-tls    46m

davar@carbon:~$ kubectl get ingress --all-namespaces
Warning: extensions/v1beta1 Ingress is deprecated in v1.14+, unavailable in v1.22+; use networking.k8s.io/v1 Ingress
NAMESPACE   NAME               CLASS    HOSTS                    ADDRESS         PORTS     AGE
data        openfaas-ingress   <none>   gateway.openfaas.local                   80        14d
data        faas               <none>   faas.data.davar.com      192.168.0.100   80, 443   14d
data        minio-ingress      <none>   minio.data.davar.com     192.168.0.100   80, 443   9d
data        mlflow             <none>   mlflow.data.davar.com    192.168.0.100   80, 443   8d
default     quality            <none>   quality.data.davar.com   192.168.0.100   80, 443   7d7h
data        kibana             <none>   kib.data.davar.com       192.168.0.100   80, 443   14d
data        nifi               <none>   nifi.data.davar.com      192.168.0.100   80, 443   14d
data        presto             <none>   presto.data.davar.com    192.168.0.100   80, 443   46m

davar@carbon:~$ kubectl get all --all-namespaces
NAMESPACE             NAME                                              READY   STATUS      RESTARTS   AGE
kube-system           pod/helm-install-traefik-fbmkt                    0/1     Completed   0          21d
gitlab-managed-apps   pod/install-helm                                  0/1     Error       0          20d
data                  pod/jupyter-notebook                              0/1     Unknown     0          98m
seldon-system         pod/seldon-controller-manager-99f687d8d-5nv74     1/1     Running     88         7d12h
default               pod/dnsutils                                      1/1     Running     226        20d
kube-system           pod/metrics-server-7b4f8b595-964g7                1/1     Running     47         21d
cert-manager          pod/cert-manager-5597cff495-rrl52                 1/1     Running     49         15d
default               pod/busybox                                       1/1     Running     230        20d
data                  pod/kibana-67c68595b7-hgmlb                       1/1     Running     36         14d
kube-system           pod/local-path-provisioner-7ff9579c6-88rrd        1/1     Running     160        21d
kube-system           pod/svclb-traefik-w9lq6                           2/2     Running     92         21d
data                  pod/nifi-1                                        0/1     Pending     0          14d
kube-system           pod/seldon-spartakus-volunteer-5b57b95596-5xpcx   1/1     Running     21         7d12h
data                  pod/logstash-7b445484d8-tn4ww                     1/1     Running     36         14d
kube-system           pod/coredns-66c464876b-lpfv4                      1/1     Running     53         20d
data                  pod/mysql-79ffd9d957-xc6xx                        1/1     Running     5          10h
data                  pod/mqtt-cbdf9fb4-c2grj                           1/1     Running     37         14d
data                  pod/nifi-0                                        1/1     Running     36         14d
data                  pod/sentimentanalysis-9b98675f9-bf6jw             1/1     Running     54         14d
data                  pod/prometheus-78dc788984-m7q4z                   1/1     Running     36         14d
data                  pod/basic-auth-plugin-bc899c574-t55r2             1/1     Running     36         14d
data                  pod/elasticsearch-0                               1/1     Running     37         14d
data                  pod/nats-7d86c64647-lmktk                         1/1     Running     36         14d
cert-manager          pod/cert-manager-cainjector-bd5f9c764-z7bh6       1/1     Running     136        15d
data                  pod/presto-data-worker-678564cfc5-z4699           1/1     Running     1          53m
cert-manager          pod/cert-manager-webhook-5f57f59fbc-49jk7         1/1     Running     48         15d
data                  pod/alertmanager-6fcb5b9b7b-c7hqb                 1/1     Running     36         14d
kube-system           pod/traefik-5dd496474-xbdg2                       1/1     Running     51         21d
data                  pod/mlflow-0                                      1/1     Running     23         8d
data                  pod/kafka-client-util                             1/1     Running     37         14d
data                  pod/presto-data-coordinator-64f7ffbb99-5hlrb      1/1     Running     1          53m
data                  pod/zookeeper-0                                   1/1     Running     37         14d
data                  pod/zookeeper-1                                   1/1     Running     37         14d
data                  pod/queue-worker-5c76c4bd84-dg9db                 1/1     Running     55         14d
data                  pod/gateway-58fd85c86b-5klq4                      2/2     Running     91         14d
data                  pod/faas-idler-6df76476c9-6dhdp                   1/1     Running     100        14d
data                  pod/kafka-0                                       1/1     Running     47         14d
data                  pod/kafka-1                                       1/1     Running     46         14d
data                  pod/minio-698d6d54c8-xkvpq                        1/1     Running     35         9d
data                  pod/hive-dccc9f446-6wsg2                          1/1     Running     10         10h
default               pod/quality-default-0-quality-6d4664bd99-pl28n    2/2     Running     64         7d7h

NAMESPACE       NAME                              TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                        AGE
default         service/kubernetes                ClusterIP      10.43.0.1       <none>          443/TCP                        21d
kube-system     service/metrics-server            ClusterIP      10.43.139.93    <none>          443/TCP                        21d
kube-system     service/traefik-prometheus        ClusterIP      10.43.78.216    <none>          9100/TCP                       21d
kube-system     service/kube-dns                  ClusterIP      10.43.0.10      <none>          53/UDP,53/TCP,9153/TCP         21d
cert-manager    service/cert-manager              ClusterIP      10.43.162.66    <none>          9402/TCP                       15d
cert-manager    service/cert-manager-webhook      ClusterIP      10.43.202.9     <none>          443/TCP                        15d
data            service/zookeeper                 ClusterIP      10.43.215.75    <none>          2181/TCP                       14d
data            service/zookeeper-headless        ClusterIP      None            <none>          2181/TCP,3888/TCP,2888/TCP     14d
data            service/kafka                     ClusterIP      10.43.166.24    <none>          9092/TCP                       14d
data            service/kafka-headless            ClusterIP      None            <none>          9092/TCP                       14d
data            service/mqtt                      ClusterIP      10.43.133.39    <none>          1883/TCP                       14d
data            service/elasticsearch             ClusterIP      10.43.243.188   <none>          9200/TCP                       14d
data            service/logstash                  ClusterIP      10.43.69.80     <none>          5044/TCP                       14d
data            service/kibana                    ClusterIP      10.43.142.124   <none>          80/TCP                         14d
data            service/nifi                      ClusterIP      None            <none>          8080/TCP,6007/TCP              14d
data            service/gateway                   ClusterIP      10.43.169.202   <none>          8080/TCP                       14d
data            service/nats                      ClusterIP      10.43.142.201   <none>          4222/TCP                       14d
data            service/basic-auth-plugin         ClusterIP      10.43.70.145    <none>          8080/TCP                       14d
data            service/alertmanager              ClusterIP      10.43.17.222    <none>          9093/TCP                       14d
data            service/prometheus                ClusterIP      10.43.21.25     <none>          9090/TCP                       14d
data            service/sentimentanalysis         ClusterIP      10.43.190.146   <none>          8080/TCP                       14d
data            service/minio-service             ClusterIP      10.43.248.161   <none>          9000/TCP                       9d
data            service/minio-service-headless    ClusterIP      None            <none>          9000/TCP                       9d
data            service/mlflow                    ClusterIP      10.43.60.15     <none>          5000/TCP                       8d
seldon-system   service/seldon-webhook-service    ClusterIP      10.43.220.83    <none>          443/TCP                        7d22h
default         service/quality-default-quality   ClusterIP      10.43.244.50    <none>          9000/TCP,9500/TCP              7d7h
default         service/quality-default           ClusterIP      10.43.68.44     <none>          8000/TCP,5001/TCP              7d7h
data            service/mysql-service             ClusterIP      10.43.161.77    <none>          3306/TCP                       10h
data            service/hive                      ClusterIP      10.43.178.75    <none>          10000/TCP,9083/TCP,10002/TCP   10h
data            service/presto-data               ClusterIP      10.43.226.235   <none>          80/TCP                         53m
kube-system     service/traefik                   LoadBalancer   10.43.100.221   192.168.0.100   80:31768/TCP,443:30058/TCP     21d

NAMESPACE     NAME                           DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
kube-system   daemonset.apps/svclb-traefik   1         1         1       1            1           <none>          21d

NAMESPACE       NAME                                         READY   UP-TO-DATE   AVAILABLE   AGE
seldon-system   deployment.apps/seldon-controller-manager    1/1     1            1           7d22h
kube-system     deployment.apps/seldon-spartakus-volunteer   1/1     1            1           7d22h
kube-system     deployment.apps/local-path-provisioner       1/1     1            1           21d
data            deployment.apps/kibana                       1/1     1            1           14d
kube-system     deployment.apps/metrics-server               1/1     1            1           21d
cert-manager    deployment.apps/cert-manager                 1/1     1            1           15d
data            deployment.apps/logstash                     1/1     1            1           14d
kube-system     deployment.apps/coredns                      1/1     1            1           21d
data            deployment.apps/mysql                        1/1     1            1           10h
data            deployment.apps/mqtt                         1/1     1            1           14d
data            deployment.apps/sentimentanalysis            1/1     1            1           14d
data            deployment.apps/prometheus                   1/1     1            1           14d
data            deployment.apps/basic-auth-plugin            1/1     1            1           14d
data            deployment.apps/nats                         1/1     1            1           14d
cert-manager    deployment.apps/cert-manager-cainjector      1/1     1            1           15d
data            deployment.apps/presto-data-worker           1/1     1            1           53m
cert-manager    deployment.apps/cert-manager-webhook         1/1     1            1           15d
data            deployment.apps/alertmanager                 1/1     1            1           14d
kube-system     deployment.apps/traefik                      1/1     1            1           21d
data            deployment.apps/presto-data-coordinator      1/1     1            1           53m
data            deployment.apps/queue-worker                 1/1     1            1           14d
data            deployment.apps/gateway                      1/1     1            1           14d
data            deployment.apps/faas-idler                   1/1     1            1           14d
data            deployment.apps/minio                        1/1     1            1           9d
data            deployment.apps/hive                         1/1     1            1           10h
default         deployment.apps/quality-default-0-quality    1/1     1            1           7d7h

NAMESPACE       NAME                                                    DESIRED   CURRENT   READY   AGE
seldon-system   replicaset.apps/seldon-controller-manager-99f687d8d     1         1         1       7d12h
kube-system     replicaset.apps/seldon-spartakus-volunteer-5b57b95596   1         1         1       7d12h
kube-system     replicaset.apps/local-path-provisioner-7ff9579c6        1         1         1       21d
data            replicaset.apps/kibana-67c68595b7                       1         1         1       14d
kube-system     replicaset.apps/metrics-server-7b4f8b595                1         1         1       21d
cert-manager    replicaset.apps/cert-manager-5597cff495                 1         1         1       15d
data            replicaset.apps/logstash-7b445484d8                     1         1         1       14d
kube-system     replicaset.apps/coredns-66c464876b                      1         1         1       21d
data            replicaset.apps/mysql-79ffd9d957                        1         1         1       10h
data            replicaset.apps/mqtt-cbdf9fb4                           1         1         1       14d
data            replicaset.apps/sentimentanalysis-9b98675f9             1         1         1       14d
data            replicaset.apps/prometheus-78dc788984                   1         1         1       14d
data            replicaset.apps/basic-auth-plugin-bc899c574             1         1         1       14d
data            replicaset.apps/nats-7d86c64647                         1         1         1       14d
cert-manager    replicaset.apps/cert-manager-cainjector-bd5f9c764       1         1         1       15d
data            replicaset.apps/presto-data-worker-678564cfc5           1         1         1       53m
cert-manager    replicaset.apps/cert-manager-webhook-5f57f59fbc         1         1         1       15d
data            replicaset.apps/alertmanager-6fcb5b9b7b                 1         1         1       14d
kube-system     replicaset.apps/traefik-5dd496474                       1         1         1       21d
data            replicaset.apps/presto-data-coordinator-64f7ffbb99      1         1         1       53m
data            replicaset.apps/queue-worker-5c76c4bd84                 1         1         1       14d
data            replicaset.apps/gateway-58fd85c86b                      1         1         1       14d
data            replicaset.apps/faas-idler-6df76476c9                   1         1         1       14d
data            replicaset.apps/minio-698d6d54c8                        1         1         1       9d
data            replicaset.apps/hive-dccc9f446                          1         1         1       10h
default         replicaset.apps/quality-default-0-quality-6d4664bd99    1         1         1       7d7h

NAMESPACE   NAME                             READY   AGE
data        statefulset.apps/nifi            1/2     14d
data        statefulset.apps/elasticsearch   1/1     14d
data        statefulset.apps/mlflow          1/1     8d
data        statefulset.apps/zookeeper       2/2     14d
data        statefulset.apps/kafka           2/2     14d

NAMESPACE     NAME                             COMPLETIONS   DURATION   AGE
kube-system   job.batch/helm-install-traefik   1/1           44s        21d

davar@carbon:~$ kubectl get all -n data
NAME                                           READY   STATUS    RESTARTS   AGE
pod/jupyter-notebook                           0/1     Unknown   0          98m
pod/kibana-67c68595b7-hgmlb                    1/1     Running   36         14d
pod/nifi-1                                     0/1     Pending   0          14d
pod/logstash-7b445484d8-tn4ww                  1/1     Running   36         14d
pod/mysql-79ffd9d957-xc6xx                     1/1     Running   5          10h
pod/mqtt-cbdf9fb4-c2grj                        1/1     Running   37         14d
pod/nifi-0                                     1/1     Running   36         14d
pod/sentimentanalysis-9b98675f9-bf6jw          1/1     Running   54         14d
pod/prometheus-78dc788984-m7q4z                1/1     Running   36         14d
pod/basic-auth-plugin-bc899c574-t55r2          1/1     Running   36         14d
pod/elasticsearch-0                            1/1     Running   37         14d
pod/nats-7d86c64647-lmktk                      1/1     Running   36         14d
pod/presto-data-worker-678564cfc5-z4699        1/1     Running   1          53m
pod/alertmanager-6fcb5b9b7b-c7hqb              1/1     Running   36         14d
pod/mlflow-0                                   1/1     Running   23         8d
pod/kafka-client-util                          1/1     Running   37         14d
pod/presto-data-coordinator-64f7ffbb99-5hlrb   1/1     Running   1          53m
pod/zookeeper-0                                1/1     Running   37         14d
pod/zookeeper-1                                1/1     Running   37         14d
pod/queue-worker-5c76c4bd84-dg9db              1/1     Running   55         14d
pod/gateway-58fd85c86b-5klq4                   2/2     Running   91         14d
pod/faas-idler-6df76476c9-6dhdp                1/1     Running   100        14d
pod/kafka-0                                    1/1     Running   47         14d
pod/kafka-1                                    1/1     Running   46         14d
pod/minio-698d6d54c8-xkvpq                     1/1     Running   35         9d
pod/hive-dccc9f446-6wsg2                       1/1     Running   10         10h

NAME                             TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                        AGE
service/zookeeper                ClusterIP   10.43.215.75    <none>        2181/TCP                       14d
service/zookeeper-headless       ClusterIP   None            <none>        2181/TCP,3888/TCP,2888/TCP     14d
service/kafka                    ClusterIP   10.43.166.24    <none>        9092/TCP                       14d
service/kafka-headless           ClusterIP   None            <none>        9092/TCP                       14d
service/mqtt                     ClusterIP   10.43.133.39    <none>        1883/TCP                       14d
service/elasticsearch            ClusterIP   10.43.243.188   <none>        9200/TCP                       14d
service/logstash                 ClusterIP   10.43.69.80     <none>        5044/TCP                       14d
service/kibana                   ClusterIP   10.43.142.124   <none>        80/TCP                         14d
service/nifi                     ClusterIP   None            <none>        8080/TCP,6007/TCP              14d
service/gateway                  ClusterIP   10.43.169.202   <none>        8080/TCP                       14d
service/nats                     ClusterIP   10.43.142.201   <none>        4222/TCP                       14d
service/basic-auth-plugin        ClusterIP   10.43.70.145    <none>        8080/TCP                       14d
service/alertmanager             ClusterIP   10.43.17.222    <none>        9093/TCP                       14d
service/prometheus               ClusterIP   10.43.21.25     <none>        9090/TCP                       14d
service/sentimentanalysis        ClusterIP   10.43.190.146   <none>        8080/TCP                       14d
service/minio-service            ClusterIP   10.43.248.161   <none>        9000/TCP                       9d
service/minio-service-headless   ClusterIP   None            <none>        9000/TCP                       9d
service/mlflow                   ClusterIP   10.43.60.15     <none>        5000/TCP                       8d
service/mysql-service            ClusterIP   10.43.161.77    <none>        3306/TCP                       10h
service/hive                     ClusterIP   10.43.178.75    <none>        10000/TCP,9083/TCP,10002/TCP   10h
service/presto-data              ClusterIP   10.43.226.235   <none>        80/TCP                         53m

NAME                                      READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/kibana                    1/1     1            1           14d
deployment.apps/logstash                  1/1     1            1           14d
deployment.apps/mysql                     1/1     1            1           10h
deployment.apps/mqtt                      1/1     1            1           14d
deployment.apps/sentimentanalysis         1/1     1            1           14d
deployment.apps/prometheus                1/1     1            1           14d
deployment.apps/basic-auth-plugin         1/1     1            1           14d
deployment.apps/nats                      1/1     1            1           14d
deployment.apps/presto-data-worker        1/1     1            1           53m
deployment.apps/alertmanager              1/1     1            1           14d
deployment.apps/presto-data-coordinator   1/1     1            1           53m
deployment.apps/queue-worker              1/1     1            1           14d
deployment.apps/gateway                   1/1     1            1           14d
deployment.apps/faas-idler                1/1     1            1           14d
deployment.apps/minio                     1/1     1            1           9d
deployment.apps/hive                      1/1     1            1           10h

NAME                                                 DESIRED   CURRENT   READY   AGE
replicaset.apps/kibana-67c68595b7                    1         1         1       14d
replicaset.apps/logstash-7b445484d8                  1         1         1       14d
replicaset.apps/mysql-79ffd9d957                     1         1         1       10h
replicaset.apps/mqtt-cbdf9fb4                        1         1         1       14d
replicaset.apps/sentimentanalysis-9b98675f9          1         1         1       14d
replicaset.apps/prometheus-78dc788984                1         1         1       14d
replicaset.apps/basic-auth-plugin-bc899c574          1         1         1       14d
replicaset.apps/nats-7d86c64647                      1         1         1       14d
replicaset.apps/presto-data-worker-678564cfc5        1         1         1       53m
replicaset.apps/alertmanager-6fcb5b9b7b              1         1         1       14d
replicaset.apps/presto-data-coordinator-64f7ffbb99   1         1         1       53m
replicaset.apps/queue-worker-5c76c4bd84              1         1         1       14d
replicaset.apps/gateway-58fd85c86b                   1         1         1       14d
replicaset.apps/faas-idler-6df76476c9                1         1         1       14d
replicaset.apps/minio-698d6d54c8                     1         1         1       9d
replicaset.apps/hive-dccc9f446                       1         1         1       10h

NAME                             READY   AGE
statefulset.apps/nifi            1/2     14d
statefulset.apps/elasticsearch   1/1     14d
statefulset.apps/mlflow          1/1     8d
statefulset.apps/zookeeper       2/2     14d
statefulset.apps/kafka           2/2     14d
davar@carbon:~$ helm list --all-namespaces
NAME               	NAMESPACE    	REVISION	UPDATED                                	STATUS  	CHART                     	APP VERSION
davar-data-openfaas	data         	1       	2020-11-27 16:48:59.150027039 +0200 EET	deployed	openfaas-6.2.0            	           
presto-data        	data         	1       	2020-12-11 20:26:14.817076845 +0200 EET	deployed	presto-1                  	           
seldon-core        	seldon-system	1       	2020-12-03 23:11:32.581974089 +0200 EET	deployed	seldon-core-operator-1.5.0	           
traefik            	kube-system  	1       	2020-11-20 06:03:58.611313978 +0000 UTC	deployed	traefik-1.81.0            	1.7.19     
davar@carbon:~$ 

davar@carbon:~$ kubectl get all -n default
NAME                                             READY   STATUS    RESTARTS   AGE
pod/dnsutils                                     1/1     Running   226        20d
pod/busybox                                      1/1     Running   230        20d
pod/quality-default-0-quality-6d4664bd99-pl28n   2/2     Running   64         7d7h

NAME                              TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)             AGE
service/kubernetes                ClusterIP   10.43.0.1      <none>        443/TCP             21d
service/quality-default-quality   ClusterIP   10.43.244.50   <none>        9000/TCP,9500/TCP   7d7h
service/quality-default           ClusterIP   10.43.68.44    <none>        8000/TCP,5001/TCP   7d7h

NAME                                        READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/quality-default-0-quality   1/1     1            1           7d7h

NAME                                                   DESIRED   CURRENT   READY   AGE
replicaset.apps/quality-default-0-quality-6d4664bd99   1         1         1       7d7h

davar@carbon:~$ kubectl get all -n seldon-system
NAME                                            READY   STATUS    RESTARTS   AGE
pod/seldon-controller-manager-99f687d8d-5nv74   1/1     Running   88         7d12h

NAME                             TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
service/seldon-webhook-service   ClusterIP   10.43.220.83   <none>        443/TCP   7d22h

NAME                                        READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/seldon-controller-manager   1/1     1            1           7d22h

NAME                                                  DESIRED   CURRENT   READY   AGE
replicaset.apps/seldon-controller-manager-99f687d8d   1         1         1       7d12h


```

### Apache Spark with S3(MinIO) as data source

```
kubectl apply -f ./003-data/5000-spark/10-spark-master-controller.yaml
kubectl apply -f ./003-data/5000-spark/20-spark-master-service.yaml
kubectl apply -f ./003-data/5000-spark/50-ingress.yaml
kubectl apply -f ./003-data/5000-spark/60-spark-worker-controller.yaml

```
Check Spark cluster:
```
$ kubectl get all -n data|grep spark
pod/spark-master-controller-4nljh       1/1     Running   8          67m
pod/spark-worker-controller-jrhjg       1/1     Running   0          46m
pod/spark-worker-controller-2pzk6       1/1     Running   0          46m
replicationcontroller/spark-master-controller   1         1         1       67m
replicationcontroller/spark-worker-controller   2         2         2       46m
service/spark-master-headless    ClusterIP   None            <none>        <none>                       46m
service/spark-master             ClusterIP   10.43.5.150     <none>        7077/TCP,8080/TCP            46m

$ kubectl get ing -n data|grep spark
spark-ingress      <none>   spark.data.davar.com     192.168.0.100   80, 443   44m
```


### GitLab (in-cluster CI/CD)  
```
kubectl apply -f ./003-data/2000-gitlab/00-namespace.yml
kubectl apply -f ./003-data/2000-gitlab/10-services.yml
kubectl apply -f ./003-data/2000-gitlab/20-configmap.yml
kubectl apply -f ./003-data/2000-gitlab/40-deployment.yml
kubectl apply -f ./003-data/2000-gitlab/50-ingress.yml
kubectl apply -f ./003-data/2000-gitlab/70-gitlab-admin-service-account.yaml
```
Integrate k8s cluster with GitLab (GitOps)

Ref: Deploy in-Cluster GitLab for K8s Development HOWTO (Developing for Kubernetes with k3s+GitLab): https://github.com/adavarski/k3s-GitLab-development 

### Argo CD (GitOps)

```
kubectl create ns argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```
We can configure argo to run deployments for us instead of kubectl apply -f. The goal here to always have our ML/DeepML deployment in sync with the representation of our deployment (YAML).

Note: GitOps

GitOps, a process popularized by Weaveworks, is another trending concept within the scope of Kubernetes CI/CD. GitOps involves the use of applications reacting to `git push events`. GitOps focuses primarily on Kubernetes clusters matching the state described by configuration residing in a Git repository. On a simplistic level, GitOps aims to replace `kubectl apply` with `git push`. Popular and well-supported GitOps implementations include GitLab, ArgoCD, Flux, and Jenkins X.




## Extend k3s cluster: Add k3s worker (bare-metal)

Install Ubuntu on some bare-metal server/workstation and setup resolvconf.service to use local DNS server.

```
$ sudo su -
$ apt update && apt upgrade -y
$ apt install -y apt-transport-https ca-certificates gnupg-agent software-properties-common
$ apt install -y linux-headers-$(uname -r)
$ export K3S_CLUSTER_SECRET="<PASTE VALUE>"
$ export K3S_URL="https://dev-k3s.davar.com:6443"
$ curl -sfL https://get.k3s.io | sh -s - agent 
$ scp root@192.168.0.101:/etc/rancher/k3s/k3s.yaml ~/.kube/k3s-config
$ sed -i "s/127.0.0.1/192.168.0.101/" ~/.kube/k3s-config
```
Fix k3s CoreDNS for local development to use local DNS server if needed.

##  Extend k3s cluster: Add k3s worker with NVIDIA Runtime for heavy ML workloads (bare-metal)

```

# Install Ubuntu 18.04 on some bare-metal server/workstation with an NVIDIA GeForce GPU (for example Gaming NVIDIA GeForce 1070 GPU)   .
$ sudo su
$ apt update && apt upgrade -y
$ apt install -y apt-transport-https \
ca-certificates gnupg-agent \
software-properties-common
$ apt install -y linux-headers-$(uname -r)
# Ceph block device kernel module (for Ceph support)
$ modprobe rbd
# NVIDIA GeForce GPU support. Install GPU drivers along with the nvidia-container-runtime plug-in for
containerd, the default container runtime for k3s.
$ apt install ubuntu-drivers-common
$ modprobe ipmi_devintf
$ add-apt-repository -y ppa:graphics-drivers
$ curl -s -L \
   https://nvidia.github.io/nvidia-docker/gpgkey \
   | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/
ubuntu18.04/nvidia-container-runtime.list | tee /etc/apt/
sources.list.d/nvidia-container-runtime.list   
$ apt-get install -y nvidia-driver-440
$ apt-get install -y nvidia-container-runtime
$ apt-get install -y nvidia-modprobe nvidia-smi
$ /sbin/modprobe nvidia
$ /sbin/modprobe nvidia-uvm  
$ reboot
$ nvidia-smi
# k3s with NVIDIA Runtime
sudo su -
export K3S_CLUSTER_SECRET="<PASTE VALUE>"
export K3S_URL="https://dev-k3s.davar.com:6443"
export INSTALL_K3S_SKIP_START=true
$ curl -sfL https://get.k3s.io | \
sh -s - agent 
$ mkdir -p /var/lib/rancher/k3s/agent/etc/containerd/
$ cat <<"EOF" > \
/var/lib/rancher/k3s/agent/etc/containerd/config.toml
[plugins.opt]
  path = "/var/lib/rancher/k3s/agent/containerd"
[plugins.cri]
  stream_server_address = "127.0.0.1"
  stream_server_port = "10010"
  sandbox_image = "docker.io/rancher/pause:3.1"
[plugins.cri.containerd.runtimes.runc]
  runtime_type = "io.containerd.runtime.v1.linux"
[plugins.linux]
  runtime = "nvidia-container-runtime"
EOF
$ systemctl start k3s
$ kubectl label node gpu-metal kubernetes.io/role=gpu
# Edit k8s Geth miners yaml ( ./cluster-davar-eth/100-eth/40-miner/30-deployment.yml) to run Pods on gpu-metal using label:gpu

```

Fix k3s CoreDNS for local development to use local DNS server if needed.


## Demo1: [DataProcessing: Serverless:OpenFaaS+ETL:Apache Nifi](https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo1-DataProcessing-Serverless-ETL/)

## Demo2: [DataProcessing-MinIO](https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo2-DataProcessing-MinIO/)

## Demo3: [AutoML:MLFlow+Seldon Core](https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo3-AutoML-MLFlow-SeldonCore)

## Demo4: [DeepML with TensorFlow](https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo4-DeepML-TensorFlow)

## Demo5: [BigData:MinIO Data Lake with Hive/Presto SQL-Engines](https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto)

## Demo6: [Spark with S3(MinIO) for large-scale big data processing and ML](https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo6-Spark-ML)

## Clean environment

```
kubectl delete -f ./003-data/000-namespace/00-namespace.yml
```
Note: all resources/objects into data namespace will be auto-removed by k8s.


# minikube:
```
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && mv ./minikube /usr/local/bin/
minikube start --cpus 2 --memory 6150 --insecure-registry="docker.infra.example.com"

```
Deploy in-Cluster GitLab for K8s Development HOWTO (Developing for Kubernetes with minikube+GitLab): https://github.com/adavarski/minikube-gitlab-development

# kubespray (HA: 2 masters)
```
$ git clone https://github.com/kubernetes-sigs/kubespray
$ sudo yum install python-pip; sudo pip install --upgrade pip; 
$ sudo pip install -r requirements.txt; vagrant up

On all k8s nodes fix docker networking: 

/etc/docker/daemon.json

{
  "insecure-registries" : ["docker.infra.example.com"],
  "bip": "10.30.0.1/16",
  "default-address-pools":
  [
     {"base":"10.20.0.0/16","size":24}
  ]
}

$ vagrant halt; vagrant up

Kubectl: 

$ vagrant ssh k8s-1 -c "sudo cat /etc/kubernetes/admin.conf" > k8s-cluster.conf
$ export KUBECONFIG=./k8s-cluster.conf 
$ kubectl version
$ kubectl cluster-info
Kubernetes master is running at https://172.17.8.101:6443
coredns is running at https://172.17.8.101:6443/api/v1/namespaces/kube-system/services/coredns:dns/proxy
kubernetes-dashboard is running at https://172.17.8.101:6443/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy

$ kubectl get nodes -L beta.kubernetes.io/arch -L beta.kubernetes.io/os -L beta.kubernetes.io/instance-type
NAME    STATUS   ROLES    AGE    VERSION   ARCH    OS      INSTANCE-TYPE
k8s-1   Ready    master   209d   v1.16.3   amd64   linux   
k8s-2   Ready    master   209d   v1.16.3   amd64   linux   
k8s-3   Ready    <none>   209d   v1.16.3   amd64   linux  

$ kubectl get pods -o wide --sort-by="{.spec.nodeName}" --all-namespaces

Helm:

$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
$ ./get_helm.sh
$ helm repo add stable https://kubernetes-charts.storage.googleapis.com/
$ helm repo add incubator http://storage.googleapis.com/kubernetes-charts-incubator
$ helm repo add hashicorp https://helm.releases.hashicorp.com
$ helm search repo stable
$ helm search repo hashicorp/consul
$ helm install consul hashicorp/consul --set global.name=consul
$ helm install incubator/kafka --set global.name=kafka
$ helm install stable/postgresql ... ; etc. etc.

$ helm ls
NAME                    	NAMESPACE	REVISION	UPDATED                                 	STATUS  	CHART              	APP VERSION
consul-1594280853       	default  	1       	2020-07-09 10:47:36.997366559 +0300 EEST	deployed	consul-3.9.6       	1.5.3      
fluent-bit-1594282992   	default  	1       	2020-07-09 11:23:16.306895607 +0300 EEST	deployed	fluent-bit-2.8.17  	1.3.7      
grafana-1594282747      	default  	1       	2020-07-09 11:19:10.878858677 +0300 EEST	deployed	grafana-5.3.5      	7.0.3      
kibana-1594282930       	default  	1       	2020-07-09 11:22:14.332304833 +0300 EEST	deployed	kibana-3.2.6       	6.7.0      
logstash-1594282961     	default  	1       	2020-07-09 11:22:45.049385698 +0300 EEST	deployed	logstash-2.4.0     	7.1.1      
postgresql-1594282655   	default  	1       	2020-07-09 11:17:38.550513366 +0300 EEST	deployed	postgresql-8.6.4   	11.7.0     
telegraf-1594282866     	default  	1       	2020-07-09 11:21:09.59958005 +0300 EEST 	deployed	telegraf-1.6.1     	1.12       
...
...

Note1: If we have working docker-compose based env, it’s easy to migrate to k8s dev env (minikube:local; kubespary:local, clouds; KOPS: AWS; etc.) 
->  kompose convert

curl -L https://github.com/kubernetes/kompose/releases/download/v1.21.0/kompose-linux-amd64 -o kompose && chmod +x kompose && sudo mv ./kompose /usr/local/bin/kompose 
./kompose convert (docker-copmose.yml)

Note2: Helm (k8s) + CI/CD (Jenkins) continious deployment. 

Note3: If we don’t use k8s we have to write our TF modules for SaaS (PoC: AWS) to have IaC based deployment for all services: ELK, Kafka, Consul, Grafana, Sensu, InfluxDB, etc. , etc. —> and have IaC: TF modules source @GitHub.

Examples TF: 
https://github.com/phiroict/terraform-aws-kafka-cluster; https://github.com/dwmkerr/terraform-consul-cluster, etc.

Note4.It's beter to use k8s Operators (ref: https://github.com/adavarski/k8s-operators-playground) than Helm Cahrts
```








