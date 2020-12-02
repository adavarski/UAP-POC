# Local & AWS SaaS development k8s environment

## AWS (build k8s cluster with KOPS)

KOPS is based on Terraform and is working very well for AWS k8s deployments. After AWS k8s cluster has been deployed, you can use [003-data/](https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/003-data) as base or Helm Charts or k8s Operators for SaaS services deployment @ k8s cluster (create Helm Charts: Consul cluster, Kafka cluster, Elasticsearch cluster, etc. based on stable Helm charts for all needed SaaS services: Kafka, ELK, Postgres, Consul, Grafana, Sensu, InfluxDB, etc., etc. →  Ref:https://artifacthub.io/; https://github.com/artifacthub/hub https://github.com/helm/charts (obsolete) ; etc. Better is to create k8s Operators for all needed SaaS services (ref: https://github.com/adavarski/k8s-operators-playground) than Helm Charts, based on https://github.com/operator-framework/community-operators. There are many k8s Operators @ https://operatorhub.io/ for example https://operatorhub.io/operator/postgres-operator, https://operatorhub.io/operator/elastic-cloud-eck, https://operatorhub.io/operator/banzaicloud-kafka-operator, etc. so create own based on them. 

## Local k8s development  (minikube, kubespray, k3s, etc.) 

For setting up Kubernetes local development environment, there are three recommended methods

    - k3s (default) https://k3s.io/
    - minikube https://minikube.sigs.k8s.io/docs/
    - kubespary https://kubespray.io/#/

Note: Of the three (k3s & minikube & kubespay), k3s tends to be the most viable. It is closer to a production style deployment. 

### minikube:
```
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && mv ./minikube /usr/local/bin/
minikube start --cpus 2 --memory 6150 --insecure-registry="docker.infra.example.com"

```
Deploy in-Cluster GitLab for K8s Development HOWTO (Developing for Kubernetes with minikube+GitLab): https://github.com/adavarski/minikube-gitlab-development

### kubespray (HA: 2 masters)
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

### k3s

k3s is 40MB binary that runs “a fully compliant production-grade Kubernetes distribution” and requires only 512MB of RAM. k3s is a great way to wrap applications that you may not want to run in a full production Cluster but would like to achieve greater uniformity in systems deployment, monitoring, and management across all development operations.

# k3s: (Default) k8s local development environment HOWTO 

k3s is deafult k8s developlent environment, because k3s is closer to a production style deployment, than minikube & kubespary .


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

### Crate eth namespace: data

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

## SaaS deploy 

### Monitoring

```
git clone git@github.com:coreos/kube-prometheus.git
cd kube-prometheus
# Create the namespace and CRDs, and then wait for them to be availble before creating the remaining resources
kubectl create -f manifests/setup
until kubectl get servicemonitors --all-namespaces ; do date; sleep 1; echo ""; done
kubectl create -f manifests/
$ kubectl --namespace monitoring port-forward svc/grafana 3000
````
Open http://localhost:3000 on a local workstation, and log in to Grafana with the default administrator credentials, username: admin, password: admin. Explore the prebuilt dashboards for monitoring many aspects of the Kubernetes cluster, including Nodes, Namespaces, and Pods.


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

### ETL (Routing and Transformation)

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

## Check k8s development cluster:

```
$ kubectl get all --all-namespaces
NAMESPACE             NAME                                          READY   STATUS      RESTARTS   AGE
kube-system           pod/helm-install-traefik-fbmkt                0/1     Completed   0          12d
gitlab-managed-apps   pod/install-helm                              0/1     Error       0          11d
data                  pod/nifi-1                                    0/1     Pending     0          5d6h
cert-manager          pod/cert-manager-5597cff495-rrl52             1/1     Running     19         6d23h
data                  pod/logstash-7b445484d8-tn4ww                 1/1     Running     11         5d9h
kube-system           pod/local-path-provisioner-7ff9579c6-88rrd    1/1     Running     61         12d
cert-manager          pod/cert-manager-cainjector-bd5f9c764-z7bh6   1/1     Running     34         6d23h
kube-system           pod/coredns-66c464876b-lpfv4                  1/1     Running     26         11d
kube-system           pod/metrics-server-7b4f8b595-964g7            1/1     Running     21         12d
data                  pod/nifi-0                                    1/1     Running     11         5d6h
data                  pod/nats-7d86c64647-lmktk                     1/1     Running     11         5d5h
data                  pod/kibana-67c68595b7-hgmlb                   1/1     Running     11         5d8h
cert-manager          pod/cert-manager-webhook-5f57f59fbc-49jk7     1/1     Running     18         6d23h
data                  pod/elasticsearch-0                           1/1     Running     12         5d9h
kube-system           pod/svclb-traefik-w9lq6                       2/2     Running     42         12d
data                  pod/prometheus-78dc788984-m7q4z               1/1     Running     11         5d5h
kube-system           pod/traefik-5dd496474-xbdg2                   1/1     Running     25         12d
data                  pod/mqtt-cbdf9fb4-c2grj                       1/1     Running     12         5d10h
data                  pod/queue-worker-5c76c4bd84-dg9db             1/1     Running     17         5d5h
data                  pod/kafka-client-util                         1/1     Running     12         5d10h
data                  pod/sentimentanalysis-9b98675f9-bf6jw         1/1     Running     18         5d5h
data                  pod/basic-auth-plugin-bc899c574-t55r2         1/1     Running     11         5d5h
data                  pod/alertmanager-6fcb5b9b7b-c7hqb             1/1     Running     11         5d5h
data                  pod/zookeeper-0                               1/1     Running     12         5d10h
data                  pod/zookeeper-1                               1/1     Running     12         5d10h
data                  pod/gateway-58fd85c86b-5klq4                  2/2     Running     28         5d5h
data                  pod/faas-idler-6df76476c9-6dhdp               1/1     Running     31         5d5h
data                  pod/kafka-0                                   1/1     Running     14         5d10h
data                  pod/kafka-1                                   1/1     Running     14         5d10h
data                  pod/minio-698d6d54c8-xkvpq                    1/1     Running     2          12h
default               pod/dnsutils                                  1/1     Running     126        11d
default               pod/busybox                                   1/1     Running     130        11d

NAMESPACE      NAME                             TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                      AGE
default        service/kubernetes               ClusterIP      10.43.0.1       <none>          443/TCP                      12d
kube-system    service/metrics-server           ClusterIP      10.43.139.93    <none>          443/TCP                      12d
kube-system    service/traefik-prometheus       ClusterIP      10.43.78.216    <none>          9100/TCP                     12d
kube-system    service/kube-dns                 ClusterIP      10.43.0.10      <none>          53/UDP,53/TCP,9153/TCP       12d
cert-manager   service/cert-manager             ClusterIP      10.43.162.66    <none>          9402/TCP                     6d23h
cert-manager   service/cert-manager-webhook     ClusterIP      10.43.202.9     <none>          443/TCP                      6d23h
data           service/zookeeper                ClusterIP      10.43.215.75    <none>          2181/TCP                     5d10h
data           service/zookeeper-headless       ClusterIP      None            <none>          2181/TCP,3888/TCP,2888/TCP   5d10h
data           service/kafka                    ClusterIP      10.43.166.24    <none>          9092/TCP                     5d10h
data           service/kafka-headless           ClusterIP      None            <none>          9092/TCP                     5d10h
data           service/mqtt                     ClusterIP      10.43.133.39    <none>          1883/TCP                     5d10h
data           service/elasticsearch            ClusterIP      10.43.243.188   <none>          9200/TCP                     5d9h
data           service/logstash                 ClusterIP      10.43.69.80     <none>          5044/TCP                     5d9h
data           service/kibana                   ClusterIP      10.43.142.124   <none>          80/TCP                       5d8h
data           service/nifi                     ClusterIP      None            <none>          8080/TCP,6007/TCP            5d6h
data           service/gateway                  ClusterIP      10.43.169.202   <none>          8080/TCP                     5d5h
data           service/nats                     ClusterIP      10.43.142.201   <none>          4222/TCP                     5d5h
data           service/basic-auth-plugin        ClusterIP      10.43.70.145    <none>          8080/TCP                     5d5h
data           service/alertmanager             ClusterIP      10.43.17.222    <none>          9093/TCP                     5d5h
data           service/prometheus               ClusterIP      10.43.21.25     <none>          9090/TCP                     5d5h
data           service/sentimentanalysis        ClusterIP      10.43.190.146   <none>          8080/TCP                     5d5h
data           service/minio-service            ClusterIP      10.43.248.161   <none>          9000/TCP                     5h33m
kube-system    service/traefik                  LoadBalancer   10.43.100.221   192.168.0.101   80:31768/TCP,443:30058/TCP   12d
data           service/minio-service-headless   ClusterIP      None            <none>          9000/TCP                     103m

NAMESPACE     NAME                           DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
kube-system   daemonset.apps/svclb-traefik   1         1         1       1            1           <none>          12d

NAMESPACE      NAME                                      READY   UP-TO-DATE   AVAILABLE   AGE
cert-manager   deployment.apps/cert-manager-cainjector   1/1     1            1           6d23h
data           deployment.apps/logstash                  1/1     1            1           5d9h
kube-system    deployment.apps/coredns                   1/1     1            1           12d
kube-system    deployment.apps/local-path-provisioner    1/1     1            1           12d
cert-manager   deployment.apps/cert-manager              1/1     1            1           6d23h
kube-system    deployment.apps/metrics-server            1/1     1            1           12d
data           deployment.apps/nats                      1/1     1            1           5d5h
data           deployment.apps/kibana                    1/1     1            1           5d8h
cert-manager   deployment.apps/cert-manager-webhook      1/1     1            1           6d23h
data           deployment.apps/prometheus                1/1     1            1           5d5h
kube-system    deployment.apps/traefik                   1/1     1            1           12d
data           deployment.apps/mqtt                      1/1     1            1           5d10h
data           deployment.apps/queue-worker              1/1     1            1           5d5h
data           deployment.apps/sentimentanalysis         1/1     1            1           5d5h
data           deployment.apps/basic-auth-plugin         1/1     1            1           5d5h
data           deployment.apps/alertmanager              1/1     1            1           5d5h
data           deployment.apps/gateway                   1/1     1            1           5d5h
data           deployment.apps/faas-idler                1/1     1            1           5d5h
data           deployment.apps/minio                     1/1     1            1           12h

NAMESPACE      NAME                                                DESIRED   CURRENT   READY   AGE
cert-manager   replicaset.apps/cert-manager-cainjector-bd5f9c764   1         1         1       6d23h
data           replicaset.apps/logstash-7b445484d8                 1         1         1       5d9h
kube-system    replicaset.apps/coredns-66c464876b                  1         1         1       12d
kube-system    replicaset.apps/local-path-provisioner-7ff9579c6    1         1         1       12d
cert-manager   replicaset.apps/cert-manager-5597cff495             1         1         1       6d23h
kube-system    replicaset.apps/metrics-server-7b4f8b595            1         1         1       12d
data           replicaset.apps/nats-7d86c64647                     1         1         1       5d5h
data           replicaset.apps/kibana-67c68595b7                   1         1         1       5d8h
cert-manager   replicaset.apps/cert-manager-webhook-5f57f59fbc     1         1         1       6d23h
data           replicaset.apps/prometheus-78dc788984               1         1         1       5d5h
kube-system    replicaset.apps/traefik-5dd496474                   1         1         1       12d
data           replicaset.apps/mqtt-cbdf9fb4                       1         1         1       5d10h
data           replicaset.apps/queue-worker-5c76c4bd84             1         1         1       5d5h
data           replicaset.apps/sentimentanalysis-9b98675f9         1         1         1       5d5h
data           replicaset.apps/basic-auth-plugin-bc899c574         1         1         1       5d5h
data           replicaset.apps/alertmanager-6fcb5b9b7b             1         1         1       5d5h
data           replicaset.apps/gateway-58fd85c86b                  1         1         1       5d5h
data           replicaset.apps/faas-idler-6df76476c9               1         1         1       5d5h
data           replicaset.apps/minio-698d6d54c8                    1         1         1       12h

NAMESPACE   NAME                             READY   AGE
data        statefulset.apps/nifi            1/2     5d6h
data        statefulset.apps/elasticsearch   1/1     5d9h
data        statefulset.apps/zookeeper       2/2     5d10h
data        statefulset.apps/kafka           2/2     5d10h

NAMESPACE     NAME                             COMPLETIONS   DURATION   AGE
kube-system   job.batch/helm-install-traefik   1/1           44s        12d

$ kubectl get all -n data
NAME                                    READY   STATUS    RESTARTS   AGE
pod/nifi-1                              0/1     Pending   0          5d6h
pod/logstash-7b445484d8-tn4ww           1/1     Running   11         5d9h
pod/nifi-0                              1/1     Running   11         5d6h
pod/nats-7d86c64647-lmktk               1/1     Running   11         5d5h
pod/kibana-67c68595b7-hgmlb             1/1     Running   11         5d8h
pod/elasticsearch-0                     1/1     Running   12         5d9h
pod/prometheus-78dc788984-m7q4z         1/1     Running   11         5d5h
pod/mqtt-cbdf9fb4-c2grj                 1/1     Running   12         5d10h
pod/queue-worker-5c76c4bd84-dg9db       1/1     Running   17         5d5h
pod/kafka-client-util                   1/1     Running   12         5d10h
pod/sentimentanalysis-9b98675f9-bf6jw   1/1     Running   18         5d5h
pod/basic-auth-plugin-bc899c574-t55r2   1/1     Running   11         5d5h
pod/alertmanager-6fcb5b9b7b-c7hqb       1/1     Running   11         5d5h
pod/zookeeper-0                         1/1     Running   12         5d10h
pod/zookeeper-1                         1/1     Running   12         5d10h
pod/gateway-58fd85c86b-5klq4            2/2     Running   28         5d5h
pod/faas-idler-6df76476c9-6dhdp         1/1     Running   31         5d5h
pod/kafka-0                             1/1     Running   14         5d10h
pod/kafka-1                             1/1     Running   14         5d10h
pod/minio-698d6d54c8-xkvpq              1/1     Running   2          12h

NAME                             TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                      AGE
service/zookeeper                ClusterIP   10.43.215.75    <none>        2181/TCP                     5d10h
service/zookeeper-headless       ClusterIP   None            <none>        2181/TCP,3888/TCP,2888/TCP   5d10h
service/kafka                    ClusterIP   10.43.166.24    <none>        9092/TCP                     5d10h
service/kafka-headless           ClusterIP   None            <none>        9092/TCP                     5d10h
service/mqtt                     ClusterIP   10.43.133.39    <none>        1883/TCP                     5d10h
service/elasticsearch            ClusterIP   10.43.243.188   <none>        9200/TCP                     5d9h
service/logstash                 ClusterIP   10.43.69.80     <none>        5044/TCP                     5d9h
service/kibana                   ClusterIP   10.43.142.124   <none>        80/TCP                       5d8h
service/nifi                     ClusterIP   None            <none>        8080/TCP,6007/TCP            5d6h
service/gateway                  ClusterIP   10.43.169.202   <none>        8080/TCP                     5d5h
service/nats                     ClusterIP   10.43.142.201   <none>        4222/TCP                     5d5h
service/basic-auth-plugin        ClusterIP   10.43.70.145    <none>        8080/TCP                     5d5h
service/alertmanager             ClusterIP   10.43.17.222    <none>        9093/TCP                     5d5h
service/prometheus               ClusterIP   10.43.21.25     <none>        9090/TCP                     5d5h
service/sentimentanalysis        ClusterIP   10.43.190.146   <none>        8080/TCP                     5d5h
service/minio-service            ClusterIP   10.43.248.161   <none>        9000/TCP                     5h33m
service/minio-service-headless   ClusterIP   None            <none>        9000/TCP                     103m

NAME                                READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/logstash            1/1     1            1           5d9h
deployment.apps/nats                1/1     1            1           5d5h
deployment.apps/kibana              1/1     1            1           5d8h
deployment.apps/prometheus          1/1     1            1           5d5h
deployment.apps/mqtt                1/1     1            1           5d10h
deployment.apps/queue-worker        1/1     1            1           5d5h
deployment.apps/sentimentanalysis   1/1     1            1           5d5h
deployment.apps/basic-auth-plugin   1/1     1            1           5d5h
deployment.apps/alertmanager        1/1     1            1           5d5h
deployment.apps/gateway             1/1     1            1           5d5h
deployment.apps/faas-idler          1/1     1            1           5d5h
deployment.apps/minio               1/1     1            1           12h

NAME                                          DESIRED   CURRENT   READY   AGE
replicaset.apps/logstash-7b445484d8           1         1         1       5d9h
replicaset.apps/nats-7d86c64647               1         1         1       5d5h
replicaset.apps/kibana-67c68595b7             1         1         1       5d8h
replicaset.apps/prometheus-78dc788984         1         1         1       5d5h
replicaset.apps/mqtt-cbdf9fb4                 1         1         1       5d10h
replicaset.apps/queue-worker-5c76c4bd84       1         1         1       5d5h
replicaset.apps/sentimentanalysis-9b98675f9   1         1         1       5d5h
replicaset.apps/basic-auth-plugin-bc899c574   1         1         1       5d5h
replicaset.apps/alertmanager-6fcb5b9b7b       1         1         1       5d5h
replicaset.apps/gateway-58fd85c86b            1         1         1       5d5h
replicaset.apps/faas-idler-6df76476c9         1         1         1       5d5h
replicaset.apps/minio-698d6d54c8              1         1         1       12h

NAME                             READY   AGE
statefulset.apps/nifi            1/2     5d6h
statefulset.apps/elasticsearch   1/1     5d9h
statefulset.apps/zookeeper       2/2     5d10h
statefulset.apps/kafka           2/2     5d10h

$ kubectl get secrets -n data
NAME                                         TYPE                                  DATA   AGE
default-token-q6h9p                          kubernetes.io/service-account-token   3      5d10h
sysop-basic-auth                             Opaque                                1      5d10h
basic-auth                                   Opaque                                2      5d5h
apk8s-data-openfaas-controller-token-5jw72   kubernetes.io/service-account-token   3      5d5h
apk8s-data-openfaas-prometheus-token-spldb   kubernetes.io/service-account-token   3      5d5h
sh.helm.release.v1.apk8s-data-openfaas.v1    helm.sh/release.v1                    1      5d5h
data-production-tls                          kubernetes.io/tls                     3      5d8h
minio-production-tls                         kubernetes.io/tls                     3      5h45m

$ kubectl get certificates -n data
NAME                   READY   SECRET                 AGE
data-production-tls    True    data-production-tls    5d8h
minio-production-tls   True    minio-production-tls   5h33m

$ kubectl get ingress --all-namespaces
Warning: extensions/v1beta1 Ingress is deprecated in v1.14+, unavailable in v1.22+; use networking.k8s.io/v1 Ingress
NAMESPACE   NAME               CLASS    HOSTS                    ADDRESS         PORTS     AGE
data        kibana             <none>   kib.data.davar.com       192.168.0.101   80, 443   5d8h
data        nifi               <none>   nifi.data.davar.com      192.168.0.101   80, 443   5d6h
data        openfaas-ingress   <none>   gateway.openfaas.local                   80        5d5h
data        faas               <none>   faas.data.davar.com      192.168.0.101   80, 443   5d5h
data        minio-ingress      <none>   minio.data.davar.com     192.168.0.101   80, 443   5h32m

```

### GitLab (in-cluster CI/CD)

Deploy in-Cluster GitLab for K8s Development HOWTO (Developing for Kubernetes with k3s+GitLab): https://github.com/adavarski/k3s-GitLab-development 

## k8s Operators

For k8s Operators creation refer to HOWTO: https://github.com/adavarski/k8s-operators-playground

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

## Demo1: [DataProcessing: Serverless:OpenFaaS+ETL:Apache Nifi](https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo1-DataProcessing-Serverless-ETL/)

## Demo2: [Demo2-DataProcessing-MinIO](https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo2-DataProcessing-MinIO/)

## Clean environment

```
kubectl delete -f ./003-data/000-namespace/00-namespace.yml
```
Note: all resources/objects into data namespace will be auto-removed by k8s.

