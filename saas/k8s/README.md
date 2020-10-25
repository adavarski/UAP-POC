

## k8s & helm for local SaaS dev k8s-based env and for AWS dev k8s-based envs

### AWS (we will do R&D when we have separate AWS account for SaaS : build k8s cluster with KOPS)

KOPS is based on Terraform and is working very well for AWS k8s deployments. After AWS k8s cluster has been deployed, we can use Helm for SaaS services (Mission Control:NOC) @ k8s cluster (we have try to write our own Helm charts for all services: Consul cluster, Kafka cluster, etc. based on stable Helm charts for all needed SaaS services: ELK, Kafka, Consul, Grafana, Sensu, InfluxDB, etc., etc. →  Ref:https://hub.helm.sh/charts; https://github.com/helm/charts/tree/master/stable ; etc.

### Local k8s dev  (minikube, kubespray. etc.) 

For local dev: Use minikube/kubespray or what you like )

#### minikube:
```
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && chmod +x minikube && mv ./minikube /usr/local/bin/
minikube start --cpus 2 --memory 6150 --insecure-registry="docker.infra.example.com"

...

```

#### kubespray (HA: 2 masters)
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
```




