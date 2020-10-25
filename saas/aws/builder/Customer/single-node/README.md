## Singe-node for testing

### Pre

Create new AMI with packer (BASE_IMAGE="ami-01e36b7901e884a10”, region": "us-east-2”, subnet_id": "subnet-XXXX”)

```
$ aws ec2 describe-images --owners XXXXX --query 'Images[*].[CreationDate,Name,ImageId]' --filters "Name=name,Values=saas*2020*" --region us-east-2 --output table | sort -r
```
Edit variables.tf for new ami :
```
variable "ami" {
  description = "AWS AMI builded with packer"

  default = "ami-0a0db264dd0c3ab6d"
}
```

### Usage
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
terraform plan -var "parent_zone=saas.infra.example.com"
```

Launch the resources:

```
terraform apply -var "parent_zone=saas.infra.example.com"
```

Show resource details:

```
terraform show
```
Destroy 
```
terraform destroy -var "parent_zone=saas.infra.example.com"
```

```
$ aws ec2 describe-instances   --filter 'Name=instance-state-name,Values=running' |   jq -r '.Reservations[].Instances[] | [.InstanceId, .PrivateIpAddress, .Tags[].Value] | @json'
["i-07f9d7cae33ca0d0c","172.31.35.233","Instance for SaaS testing"]
$ chmod 600 saas.pem
$ ssh -i ./saas.pem centos@ec2-3-21-167-11.us-east-2.compute.amazonaws.com 

http://ec2-3-21-167-11.us-east-2.compute.amazonaws.com
```
For Jenkinsfile-DEMO install "CloudBees AWS Credentials" J.plugin and setup J.Credentials (ID: AWS_SAAS)

