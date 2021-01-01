## PaaS/SaaS data-driven Analytics/ML/DeepML platform for data analyst/data engineer/data scientist playground (R&D/MVP/POC/environmints):

Summary: Vagrant/Docker/k8s/AWS IaC-based PaaS/SaaS infrastucture/environments (POC): Terraform for AWS infrastructure provisioning, Vagrant/Docker/k8s for Dev environments and Ansible for Configuration Management (CM). 

### Used stacks and products for PaaS/SaaS POC
VPN (WireGuard, k8s:Kilo); Monitoring stacks (Prometheus-based, TIG:Telegraf+InfluxDB+Grafana, Sensu, Zabbix); Indexing and Analytics/Debugging and Log management stacks (ELK/EFK); Pipeline: Messaging/Kafka stack (Kafka cluster, Zookeper cluster, Kafka Replicator, Kafka Connect, Schema Registry); Routing and Transformation (Serverless:OpenFaaS; ETL:Apache NiFi), Data Lake/Big Data (MinIO s3-compatable Object Storage); DWHs (HIVE with MinIO:s3, Cassandra, Presto); Blockchein (Ethereum for Smart Contacts); Apache Spark for large-scale data processing and data analytics; Machine Learning/Deep Learning/AutoML (TensorFlow, k8s:Model Development with AutoML: Kubeflow(MLflow, etc.) and k8s:AI Model Deployment (Seldon Core), Spark ML with S3(MinIO) as Data Source); GitLab/Jenkins In-Platform CI/CD (including GitOps); Identity and Access Management (IAM:Keycloak); JupyterHub/JupyterLab for data science; Consul cluster; HashiCorp Vault cluster; Postgres cluster with Patroni; k8s Persistent Volumes (Rook Ceph, Gluster); etc.


### PaaS/SaaS POC/Development environments used:

- docker-compose based dev env 
- Vagrant based dev env (Vagrant: IaC for simulating public/private cloud envs -> AWS, Azure, GCP, OCP, OpenStack, etc. Very flexible, ruby syntax, better than OpenStack (KVM) for DEV, you can use Vagrant with VBOX/KVM/etc. + Packer (create Vagrant boxes for Vbox/KVM/etc. like AWS AMIs creation with Packer) + Ansible for Configuration Management
- k8s (k8s-local: k3s, minikube, kubespray) `Note: Default development environment` 
- AWS —> Terraform for provisioning infrastructure && Ansible for Configuration Management(CM). KOPS for k8s clusters deploy on AWS, and k8s Operators/Helm Charts/YAML manifests for creating k8s deployments(PasS&SaaS services).  

## PaaS/SaaS objectives:

Platform as a Service (PaaS) will be data-driven and data-science platform allowing end user to develop, run, and manage applications without the complexity of building and maintaining the infrastructure.

Software as a Service (SaaS) will be "on-demand software", accessed/used by end users using a web browser.

For production we will use k8s:

## k8s-based data-driven Analytics/ML/DeepML SaaS platform for data analyst/data engineer/data scientist playground (R&D/MVP/POC/environmints)

Summary: k8s-based Analytics/ML/DeepML SaaS using Big Data/Data Lake: MinIO (s3-compatable object storage) with Hive(s3) SQL-Engine/DWHs (instead of Snowflake as big data platform for example), Apache Spark(Hive for metadata)/Delta Lake(lakehouses)/Jupyter/etc. (instead of Databricks for example) + Kafka stack + ELK/EFK + Serverless(OpenFaaS) + ETL(Apache NiFi) + ML/DeepML/AutoML + Jupyter + GitOps. For building Analytics/ML SaaS platform we can also use cloud-native SaaSs as reference (or build SaaS based on cloud-native SaaSs): Snowflake(SaaS/DWaaS) as big data solution on a single data platform (DWH, S3, etc.) + Databricks(cloud-based big data processing SaaS) + AWS S3/MKS/SQS/ELK/Lambda/etc.
