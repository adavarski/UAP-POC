# DEV infrastructure provisioning

## Overview

## Usage
```
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=us-east-2
```

Initialize the Terraform:

```
terraform init
```

View the changes:

```
terraform plan -var "parent_zone=noc.infra.example.com" 
```

Launch the resources:

```
terraform apply -var "parent_zone=noc.infra.example.com"
```
Show resource details:

```
terraform show
```

Checks inside VMs:

```
$ ssh -i ./saas.pem centos@ec2-54-245-27-76.us-east-2.compute.amazonaws.com

```

Destroy resources
```
$ terraform destroy -var "parent_zone=noc.infra.example.com" 

```


For Jenkinsfile CloudBees AWS Credentials" J.plugin and setup J.Credentials (ID: AWS_SAAS_DEV)

Note: Generate current graph infra example with "terraform graph | dot -Tpng > SaaS-DEV-TF-graph.png"

<img src="https://github.com/adavarski/SaaS-FULL-POC/blob/main/saas/aws/terraform/SaaS-DEV-TF-graph.png?raw=true" width="650">

