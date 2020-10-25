variable "region" {
  description = "AWS region to host your infrastructure"
  default     = "us-east-2"
}

variable "key_name" {
  description = "Private key name to use with instance"
  default     = "saas"
}

variable "instance_type" {
  description = "AWS instance type"
  default     = "t3.large"
}

variable "ami" {
  description = "AWS AMI builded with packer"

  # Centos 7 
  default = "ami-0a0db264dd0c3ab6d"
}

