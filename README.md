# PaaS/SaaS data-driven/data-science platform POC/Development environmints:

Vagrant/Docker/k8s/AWS IaC-based PaaS/PaaS infrastucture/environments (POC): Terraform for AWS infrastructure provisioning, Vagrant/Docker/k8s for Dev environments and Ansible for Configuration Management (CM). 

Used stacks and products for PaaS/SaaS POC: Monitoring stacks (Prometheus-based, TIG:Telegraf+InfluxDB+Grafana, Sensu, Zabbix); Indexing and Analytics/Debugging and Log management stacks (ELK/EFK); Pipeline: Messaging/Kafka stack (Kafka cluster, Zookeper cluster, Kafka Replicator, Kafka Connect, Schema Registry); Routing and Transformation (Serverless:OpenFaaS; ETL:Apache NiFi), Data Lake (MinIO s3-compatable Object Storage); DWHs (HIVE with MinIO:s3, Cassandra, Presto); Blockchein (Ethereum for Smart Contacts); Machine Learning (TensorFlow, Model Development with AutoML: Kubeflow(MLflow, etc.); AI (Seldon Core); GitLab/Jenkins In-Platform CI/CD; Identity and Access Management (IAM:Keycloak); JupyterHub/JupyterLab for data science; Consul cluster; Postgres cluster with Patroni; etc.

### POC/DEV environments used:

- docker-compose based dev env (Note: default dev env) 
- Vagrant based dev env (Vagrant: IaC for simulating public/private cloud envs -> AWS, Azure, GCP, OCP, OpenStack, etc. Very flexible, ruby syntax, better than OpenStack (KVM) for DEV, you can use Vagrant with VBOX/KVM/etc. + Packer (create Vagrant boxes for Vbox/KVM/etc. like AWS AMIs creation with Packer) + Ansible for Configuration Management
- k8s (k8s-local: k3s, minikube, kubespray)
- AWS —> Terraform for provisioning infrastructure && Ansible for Configuration Management(CM). KOPS for k8s clusters deploy on AWS, and k8s Operators/Helm Charts/YAML manifests for creating k8s deployments(PasS&SaaS services).  

### PaaS/SaaS objectives:

Platform as a Service (PaaS) will be data-driven and data-science platform allowing end user to develop, run, and manage applications without the complexity of building and maintaining the infrastructure.

Software as a Service (SaaS) will be "on-demand software", accessed/used by end users using a web browser.

