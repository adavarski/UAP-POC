#### TBD: Building AWS k8s (with KOPS) Infrastructure with Terraform Modules  and using k8s Operators/HELM charts for SaaS-NOC services deployment (including GitLab or Jenkins for in-cluster CI/CD, private docker registry and helm private repo): 

KOPS is based on Terraform and is working very well for AWS k8s deployments. After AWS k8s cluster has been deployed, you can use https://github.com/adavarski/SaaS-Vagrant-Docker-k8s-AWS-POC/tree/main/saas/k8s/003-data (ref:https://github.com/adavarski/SaaS-Vagrant-Docker-k8s-AWS-POC/tree/main/saas/k8s) as base or Helm Charts or k8s Operators for SaaS services deployment @ k8s cluster (create Helm Charts: Consul cluster, Kafka cluster, Elasticsearch cluster, etc. based on stable Helm charts for all needed SaaS services: Kafka, ELK, Postgres, Consul, Grafana, Sensu, InfluxDB, etc., etc. → Ref:https://artifacthub.io/; https://github.com/artifacthub/hub https://github.com/helm/charts (obsolete) ; etc. Better is to create k8s Operators for all needed SaaS services (ref: https://github.com/adavarski/k8s-operators-playground) than Helm Charts, based on https://github.com/operator-framework/community-operators. There are many k8s Operators @ https://operatorhub.io/ for example https://operatorhub.io/operator/postgres-operator, https://operatorhub.io/operator/elastic-cloud-eck, https://operatorhub.io/operator/banzaicloud-kafka-operator, etc. so create own based on them.

Note1: KOPS for k8s clusters deploy on AWS, and k8s Operators/Helm Charts/YAML manifests for creating k8s deployments(PasS&SaaS services).

Note2: GitLab for k8s in-platform/in-cluster CI/CD (Easy k8s integration, Private Docker Registry, etc.). 

Deploy in-Cluster GitLab for K8s Development HOWTO (Developing for Kubernetes with k3s+GitLab): https://github.com/adavarski/k3s-GitLab-development

Note3: For k8s Operators development refer to HOWTO: https://github.com/adavarski/k8s-operators-playground

