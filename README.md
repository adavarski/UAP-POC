# SaaS POC/DEV environmints:

docker/Vagrant/k8s/AWS IaC-based SaaS infrastucture/environments POC: Terraform for AWS infrastructure provisioning, Vagrant/Docker for Dev environments and Ansible for CM. Used stacks and products for SaaS POC: Monitoring stacks (TIG:Telegraf+InfluxDB+Grafana, Sensu, Zabbix); Debugging and Log management stacks (ELK/EFK); Messaging/Kafka stack (Kafka cluster, Zookeper cluster, Kafka Replicator, Kafka Connect, Schema Registry); Consul cluster; Postgres cluster with Patroni; etc.

Note:
- docker-compose based dev env (Note: default dev env) 
- Vagrant based dev env (Vagrant: IaC for simulating public/private cloud envs -> AWS, Azure, GCP, OCP, OpenStack, etc. Very flexible, ruby syntax, better than OpenStack (KVM) for DEV, you can use Vagrant with VBOX/KVM/etc. + Packer (create Vagrant boxes for Vbox/KVM/etc. like AWS AMIs creation with Packer)
- k8s (k8s-local: minikube, kubespray)
- AWS (AWS account is needed) —> Terraform for provisioning infrastructure && Ansible for Configuration Management(CM). 

