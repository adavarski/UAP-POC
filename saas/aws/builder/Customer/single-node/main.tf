# Use AWS Terraform provider
provider "aws" {
  region = "us-east-2"
}

data "aws_availability_zones" "all" {
}

data "template_file" "myuserdata" {
  template = "${file("${path.cwd}/user-data.tpl")}"
}

# Create EC2 instance
resource "aws_instance" "demo-tf" {
  ami                    = var.ami
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.demo-tf.id]
  source_dest_check      = false
  instance_type          = var.instance_type
  user_data = "${data.template_file.myuserdata.template}"
  tags = {
    Name  = "Instance for SaaS testing"
  }
}

variable "parent_zone" {}


locals {
  fully_qualified_parent_zone = "${var.parent_zone}."
}

data "aws_route53_zone" "parent" {
  name = local.fully_qualified_parent_zone
}

resource "aws_route53_record" "single-node" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "single-node"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.demo-tf.public_ip]
}


# Create Security Group for EC2
resource "aws_security_group" "demo-tf" {
  name = "terraform-demo-sg"

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }


  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

}


