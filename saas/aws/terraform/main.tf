#---------------------------------------------------------------
# Info
# autor: davar
# email: 
# version: 0.1.0
# Descrption: SaaS DEV : Full TF (VPC & EC2 & etc.) @ AWS
#---------------------------------------------------------------
#
#
# terraform --help   
# terraform providers 
# terraform init     
# terraform validate
# terraform plan   
# terraform apply 
# terraform show 
# terraform destroy  
#---------------------------------------------------------------

#---------------------------------------------------------------
#                           PROVEDOR
#---------------------------------------------------------------

provider "aws" {
    profile = "default"
    region = "us-east-2"
}

#---------------------------------------------------------------
#                             VPC
#---------------------------------------------------------------

resource "aws_vpc" "VPC_MC_DEV" {
  cidr_block           = var.vpcCIDRblock
  instance_tenancy     = var.instanceTenancy 
  enable_dns_support   = var.dnsSupport 
  enable_dns_hostnames = var.dnsHostNames
tags = {
    Name = "VPC MC DEV"
}
}

#--------------------------------------------------------------
#                           SUBNETS
#--------------------------------------------------------------
#                 SUBNET1 - PUBLIC - LB 
#--------------------------------------------------------------

resource "aws_subnet" "Pub_subnet1" {
  vpc_id                  = aws_vpc.VPC_MC_DEV.id
  cidr_block              = var.publicsCIDRblock
  map_public_ip_on_launch = var.mapPublicIP 
  availability_zone       = var.availabilityZone
tags = {
   Name = "Sub-Pub1"
}
}

#--------------------------------------------------------------
#                 SUBNET2 - PRIVATE - MC SERVICES
#--------------------------------------------------------------

resource "aws_subnet" "Priv_subnet1" {
  vpc_id                  = aws_vpc.VPC_MC_DEV.id
  cidr_block              = var.privateCIDRblock
  map_public_ip_on_launch = var.mapPublicIP 
  availability_zone       = var.availabilityZone
tags = {
   Name = "Sub-Priv1"
}
}

#--------------------------------------------------------------
#                 SUBNET3 - PRIVATE - MONITORING
#--------------------------------------------------------------

resource "aws_subnet" "Priv_subnet2" {
  vpc_id                  = aws_vpc.VPC_MC_DEV.id
  cidr_block              = var.private2CIDRblock
  map_public_ip_on_launch = var.mapPublicIP 
  availability_zone       = var.availabilityZone
tags = {
   Name = "Sub-Priv2"
}
}

#--------------------------------------------------------------
#                            NACL
#--------------------------------------------------------------

resource "aws_network_acl" "DevOps_MC_NACL" {
  vpc_id = aws_vpc.VPC_MC_DEV.id
  subnet_ids = [ aws_subnet.Pub_subnet1.id, aws_subnet.Priv_subnet1.id, aws_subnet.Priv_subnet2.id ]
  #-----------------------------------------------------------
  #                    EPHEMERAL PORTS
  #-----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 1024
    to_port    = 65535
  }

   egress {
    protocol   = "tcp"
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 1024
    to_port    = 65535
  }  
  #-----------------------------------------------------------
  #                         HTTP
  #-----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 110
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 80
    to_port    = 80
  }

   egress {
    protocol   = "tcp"
    rule_no    = 110
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 80 
    to_port    = 80
  }
  #-----------------------------------------------------------
  #                         HTTPS
  #-----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 111
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 443
    to_port    = 443
  }

   egress {
    protocol   = "tcp"
    rule_no    = 111
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 443
    to_port    = 443
  }
   

  #----------------------------------------------------------
  #                          SSH
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 120
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 22
    to_port    = 22
  }

   egress {
    protocol   = "tcp"
    rule_no    = 120
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 22 
    to_port    = 22
  } 
  
  #----------------------------------------------------------
  #                          ORCHESTRATOR.WEB
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 130
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 11500
    to_port    = 11500
  }

   egress {
    protocol   = "tcp"
    rule_no    = 130
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 11500
    to_port    = 11500
  }

  #----------------------------------------------------------
  #                          ORCHESTRATOR.API
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 140
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 11501
    to_port    = 11501
  }

   egress {
    protocol   = "tcp"
    rule_no    = 140
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 11501
    to_port    = 11501
  }

  #----------------------------------------------------------
  #                          GRAFANA
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 150
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 3000
    to_port    = 3000
  }

   egress {
    protocol   = "tcp"
    rule_no    = 150
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 3000
    to_port    = 3000
  }

  #----------------------------------------------------------
  #                          KIBANA
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 160
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 5601
    to_port    = 5601
  }

   egress {
    protocol   = "tcp"
    rule_no    = 160
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 5601
    to_port    = 5601
  }

  #----------------------------------------------------------
  #                          ELASTICSEARCH
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 170
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9200
    to_port    = 9200
  }

   egress {
    protocol   = "tcp"
    rule_no    = 170
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9200
    to_port    = 9200
  }

  ingress {
    protocol   = "tcp"
    rule_no    = 180
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9300
    to_port    = 9300
  }

   egress {
    protocol   = "tcp"
    rule_no    = 180
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9300
    to_port    = 9300
  }


  #----------------------------------------------------------
  #                          INFLUXDB
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 190
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 8086
    to_port    = 8086
  }

   egress {
    protocol   = "tcp"
    rule_no    = 190
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 8086
    to_port    = 8086
  }

  #----------------------------------------------------------
  #                          ZOOKEEPER
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 200
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 2888
    to_port    = 2888
  }

   egress {
    protocol   = "tcp"
    rule_no    = 200
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 2888
    to_port    = 2888
  }

  ingress {
    protocol   = "tcp"
    rule_no    = 210
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 3888
    to_port    = 3888
  }

   egress {
    protocol   = "tcp"
    rule_no    = 210
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 3888
    to_port    = 3888
  }

  ingress {
    protocol   = "tcp"
    rule_no    = 220
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 2181
    to_port    = 2181
  }

   egress {
    protocol   = "tcp"
    rule_no    = 220
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 2181
    to_port    = 2181
  }

  #----------------------------------------------------------
  #                          KAFKA
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 230
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9092
    to_port    = 9092
  }

   egress {
    protocol   = "tcp"
    rule_no    = 230
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9092
    to_port    = 9092
  }

  ingress {
    protocol   = "tcp"
    rule_no    = 240
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9094
    to_port    = 9094
  }

   egress {
    protocol   = "tcp"
    rule_no    = 240
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 9094
    to_port    = 9094
  }

  #----------------------------------------------------------
  #                          POSTGRES
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 240
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 8008
    to_port    = 8008
  }

   egress {
    protocol   = "tcp"
    rule_no    = 240
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 8008
    to_port    = 8008
  }

  ingress {
    protocol   = "tcp"
    rule_no    = 250
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 5432
    to_port    = 5432
  }

   egress {
    protocol   = "tcp"
    rule_no    = 250
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 5432
    to_port    = 5432
  }

  #----------------------------------------------------------
  #                          KAFKA:SCHEMA REGISTRY
  #----------------------------------------------------------
  ingress {
    protocol   = "tcp"
    rule_no    = 260
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 8081
    to_port    = 8081
  }

   egress {
    protocol   = "tcp"
    rule_no    = 260
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 8081
    to_port    = 8081
  }

tags = {
    Name = "NACL-MC/NOC"
}
}


#--------------------------------------------------------------
#                      INTERNET GATEWAY
#--------------------------------------------------------------

resource "aws_internet_gateway" "IGW_MC_DEV" {
 vpc_id = aws_vpc.VPC_MC_DEV.id
 tags = {
        Name = "IG - MC DEV"
}
} 

#--------------------------------------------------------------
#                   ROUTE PUBLIC SUBNET PUB
#--------------------------------------------------------------

resource "aws_route_table" "Pub1_rt" {
 vpc_id = aws_vpc.VPC_MC_DEV.id
 tags = {
        Name = "RT_Pub1"
}
} 

#--------------------------------------------------------------
#                  ROUTE PRIVATE SUBNET PRIV 1
#--------------------------------------------------------------

resource "aws_route_table" "Priv1_rt" {
 vpc_id = aws_vpc.VPC_MC_DEV.id
 tags = {
        Name = "RT_Priv1"
}
} 

#--------------------------------------------------------------
#                  ROUTE PRIVATE SUBNET PRIV 2 Public
#--------------------------------------------------------------

resource "aws_route_table" "Priv2_rt" {
 vpc_id = aws_vpc.VPC_MC_DEV.id
 tags = {
        Name = "RT_Priv2"
}
} 

#--------------------------------------------------------------
#                   ROUTE FOR INTENET SUBNET1
#--------------------------------------------------------------

resource "aws_route" "access_internet" {
  route_table_id         = aws_route_table.Pub1_rt.id
  destination_cidr_block = var.publicdestCIDRblock
  gateway_id             = aws_internet_gateway.IGW_MC_DEV.id
}

# Rotas temporárias
resource "aws_route" "access_internet2" {
  route_table_id         = aws_route_table.Priv1_rt.id
  destination_cidr_block = var.publicdestCIDRblock
  gateway_id             = aws_internet_gateway.IGW_MC_DEV.id
}

resource "aws_route" "access_internet3" {
  route_table_id         = aws_route_table.Priv2_rt.id
  destination_cidr_block = var.publicdestCIDRblock
  gateway_id             = aws_internet_gateway.IGW_MC_DEV.id
}

#--------------------------------------------------------------
#                  Associating routing a subnet
#--------------------------------------------------------------

# Associating a subnet 1 - Public
resource "aws_route_table_association" "Pub_associating" {
  subnet_id      = aws_subnet.Pub_subnet1.id
  route_table_id = aws_route_table.Pub1_rt.id
}
# Associating a subnet 2 - Private
resource "aws_route_table_association" "Priv1_associating" {
  subnet_id      = aws_subnet.Priv_subnet1.id
  route_table_id = aws_route_table.Priv1_rt.id
}
# Associating a subnet 3 - Private
resource "aws_route_table_association" "Priv2_associating" {
  subnet_id      = aws_subnet.Priv_subnet2.id
  route_table_id = aws_route_table.Priv2_rt.id
}

#--------------------------------------------------------------
#                      SECURITY GROUP
#--------------------------------------------------------------
#                    SG 1 - BALANCER
#--------------------------------------------------------------

resource "aws_security_group" "srv_lb_sg" {
  name           = "lb_sg"
  description    = "SG for external access - HAproxy"
  vpc_id         = aws_vpc.VPC_MC_DEV.id

  tags = {
        Name = "srv_lb_sg"
    } 
  
  # All policies temporarily released
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = var.ingressCIDRblock
  }   

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = var.egressCIDRblock
  }

}

#--------------------------------------------------------------
#                     SG 2 - SERVICES
#--------------------------------------------------------------

resource "aws_security_group" "srv_svc_sg" {
  name           = "svc_sg"
  description    = "SG for MC servcies"
  vpc_id         = aws_vpc.VPC_MC_DEV.id

  tags = {
        Name = "srv_svc_sg"
    } 
  # All policies temporarily released
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = var.ingressCIDRblock
  # security_groups = [aws_security_group.srv_lb_sg.id, aws_security_group.srv_mt_sg.id]
  }   

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = var.egressCIDRblock
  # security_groups = [aws_security_group.srv_lb_sg.id, aws_security_group.srv_mt_sg.id]
  }

}

#--------------------------------------------------------------
#                     SG 3 - MONITORING
#--------------------------------------------------------------

resource "aws_security_group" "srv_mt_sg" {
  name           = "mt_sg"
  description    = "SG for monitoring services - Grafana, Kibana, etc."
  vpc_id         = aws_vpc.VPC_MC_DEV.id

  tags = {
        Name = "srv_mt_sg"
    } 
  
  # All policies temporarily released
  ingress {
    description = "All policies temporarily released"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = var.ingressCIDRblock
   # security_groups = [aws_security_group.srv_lb_sg.id]
   # security_groups = [aws_security_group.srv_lb_sg.id, aws_security_group.srv_svc_sg.id]
    
  }   

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = var.egressCIDRblock
  # security_groups = [aws_security_group.srv_lb_sg.id]
  # security_groups = [aws_security_group.srv_lb_sg.id, aws_security_group.srv_svc_sg.id]
  }

}

#--------------------------------------------------------------
#                       INSTANCES EC2 
#--------------------------------------------------------------
#                  INSTQNCE 1 - BALANCER
#--------------------------------------------------------------
# HAproxy
#--------------------------------------------------------------
resource "aws_instance" "lb_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group 
    vpc_security_group_ids = [aws_security_group.srv_lb_sg.id]
    subnet_id              = aws_subnet.Pub_subnet1.id
    
    # Tags instance
    tags = {
        Name = "SRV_LB"
    }   
}

#--------------------------------------------------------------
#                  INSTNCES - MC SERVICES
#--------------------------------------------------------------
# Servises:
# InfraAPI
# Orchestrator
# Consul cluster
# Postgres cluster with patroni[consul]
# Messaging/Kafka stack
# etc.
#--------------------------------------------------------------
resource "aws_instance" "svc_dcs01_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Consul01"
    }
}

resource "aws_instance" "svc_dcs02_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Consul02"
    }
}

resource "aws_instance" "svc_dcs03_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Consul03"
    }
}


resource "aws_instance" "svc_pg01_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group 
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id
    
    # Tags
    tags = {
        Name = "SRV_SVC_PG01"
    }   
}

resource "aws_instance" "svc_pg02_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group    
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id
   
    # Tags 
    tags = {
        Name = "SRV_SVC_PG02"
    }
}

resource "aws_instance" "svc_pg03_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_PG03"
    }
}

resource "aws_instance" "svc_zoo01_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = { 
        Name = "SRV_SVC_Zoo01"
    }
}

resource "aws_instance" "svc_zoo02_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Zoo02"
    }
}

resource "aws_instance" "svc_zoo03_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Zoo03"
    }
}

resource "aws_instance" "svc_kafka01_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Kafka01"
    }
}

resource "aws_instance" "svc_kafka02_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Kafka02"
    }              
}

resource "aws_instance" "svc_kafka03_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_Kafka03"
    }              
}

resource "aws_instance" "svc_schemaregistry_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_SchemaRegistry"
    }
}

resource "aws_instance" "svc_kafkaconnect_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_KafkaConnect"
    }
}

resource "aws_instance" "svc_kafkareplicator_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Security group
    vpc_security_group_ids = [aws_security_group.srv_svc_sg.id]
    subnet_id              = aws_subnet.Priv_subnet1.id

    # Tags
    tags = {
        Name = "SRV_SVC_KafkaReplicator"
    }
}


#--------------------------------------------------------------
#                  INSTANCES - MONITORING/LOGs
#--------------------------------------------------------------
# Monitoring/Debbuging&Logs servcies:
# TIG, SENSU, ZABBIX
# ELK/EFK
#--------------------------------------------------------------

resource "aws_instance" "mt_influxdb_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup  
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id
    
    # Tags 
    tags = {
        Name = "SRV_MT_InfluxDB"
    }   
}

resource "aws_instance" "mt_grafana_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id

    # Tags
    tags = {
        Name = "SRV_MT_Grafana"
    }
}

resource "aws_instance" "mt_sensu_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id

    # Tags
    tags = {
        Name = "SRV_MT_Sensu"
    }
}

resource "aws_instance" "mt_elasticnode1_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id

    # Tags
    tags = {
        Name = "SRV_MT_Elasticnode1"
    }
}

resource "aws_instance" "mt_elasticnode2_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id

    # Tags
    tags = {
        Name = "SRV_MT_Elasticnode2"
    }
}

resource "aws_instance" "mt_elasticnode3_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id

    # Tags
    tags = {
        Name = "SRV_MT_Elasticnode3"
    }
}

resource "aws_instance" "mt_kibana_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id

    # Tags
    tags = {
        Name = "SRV_MT_Kibana"
    }
}

resource "aws_instance" "mt_logstash_ec2" {
    instance_type          = var.typeInstance
    ami                    = var.ami
    key_name               = var.key_name

    # Grup
    vpc_security_group_ids = [aws_security_group.srv_mt_sg.id]
    subnet_id              = aws_subnet.Priv_subnet2.id

    # Tags
    tags = {
        Name = "SRV_MT_Logstash"
    }
}


#--------------------------------------------------------------
#                  ROUTE53 - LBs, MC Servcies, MONITORING/LOGs
#--------------------------------------------------------------
# LBs:
# MC SERVICES:
# MONITORING/LOGS: 
#--------------------------------------------------------------

variable "parent_zone" {}


locals {
  fully_qualified_parent_zone = "${var.parent_zone}."
}

data "aws_route53_zone" "parent" {
  name = local.fully_qualified_parent_zone
}

resource "aws_route53_record" "route53_lb_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "lb_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.lb_ec2.public_ip]
}

resource "aws_route53_record" "route53_dcs01_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_dcs01_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_dcs01_ec2.public_ip]
}

resource "aws_route53_record" "route53_dcs02_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_dcs02_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_dcs02_ec2.public_ip]
}

resource "aws_route53_record" "route53_dcs03_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_dcs03_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_dcs03_ec2.public_ip]
}


resource "aws_route53_record" "route53_svc_pg01_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_pg01_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_pg01_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_pg02_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_pg02_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_pg02_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_pg03_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_pg03_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_pg02_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_zoo01_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_zoo01_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_zoo01_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_zoo02_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_zoo02_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_zoo02_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_zoo03_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_zoo03_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_zoo03_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_kafka01_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_kafka01_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_kafka01_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_kafka02_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_kafka02_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_kafka02_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_kafka03_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_kafka03_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_kafka03_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_schemaregistry_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_schemaregistry_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_schemaregistry_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_kafkaconnect_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_kafkaconnect_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_kafkaconnect_ec2.public_ip]
}

resource "aws_route53_record" "route53_svc_kafkareplicator_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "svc_kafkareplicator_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.svc_kafkareplicator_ec2.public_ip]
}

resource "aws_route53_record" "route53_mt_elasticnode1_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "mt_elasticnode1_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.mt_elasticnode1_ec2.public_ip]
}

resource "aws_route53_record" "route53_mt_elasticnode2_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "mt_elasticnode2_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.mt_elasticnode2_ec2.public_ip]
}

resource "aws_route53_record" "route53_mt_elasticnode3_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "mt_elasticnode3_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.mt_elasticnode3_ec2.public_ip]
}


resource "aws_route53_record" "route53_mt_kibana_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "mt_kibana_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.mt_kibana_ec2.public_ip]
}

resource "aws_route53_record" "route53_mt_logstash_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "mt_logstash_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.mt_logstash_ec2.public_ip]
}

resource "aws_route53_record" "route53_mt_influxdb_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "mt_influxdb_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.mt_influxdb_ec2.public_ip]
}

resource "aws_route53_record" "route53_mt_grafana_ec2" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "mt_grafana_ec2"
  type    = "A"
  ttl     = "300"
  records = [aws_instance.mt_grafana_ec2.public_ip]
}



#--------------------------------------------------------------
#                  AWS ELBs - > EC2, Route53 
#--------------------------------------------------------------
# LBs:
# orchestrator.web; orchestrator.api, grafana, kibana 
# 
#--------------------------------------------------------------

resource "aws_elb" "grafana-elb" {
  name               = "grafana-elb"
  security_groups    = ["${aws_security_group.srv_lb_sg.id}"]
  subnets = ["${aws_subnet.Pub_subnet1.id}"]
  health_check {
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 3
    interval            = 30
    target              = "HTTP:3000/"
  }

  listener {
    lb_port           = 3000
    lb_protocol       = "http"
    instance_port     = "3000"
    instance_protocol = "http"
  }
}

# Create a new load balancer attachment
resource "aws_elb_attachment" "grafana-elb" {
  elb      = "aws_elb.grafana-elb.id"
  instance = "aws_instance.mt_grafana_ec2.id"
}

resource "aws_route53_record" "grafana" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "grafana-elb"
  type    = "A"

  alias {
    name                   = aws_elb.grafana-elb.dns_name
    zone_id                = aws_elb.grafana-elb.zone_id
    evaluate_target_health = true
  }
}

resource "aws_elb" "kibana-elb" {
  name               = "kibana-elb"
  security_groups    = ["${aws_security_group.srv_lb_sg.id}"]
  subnets = ["${aws_subnet.Pub_subnet1.id}"]
  health_check {
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 3
    interval            = 30
    target              = "HTTP:5601/"
  }

  listener {
    lb_port           = 5601
    lb_protocol       = "http"
    instance_port     = "5601"
    instance_protocol = "http"
  }
}

# Create a new load balancer attachment
resource "aws_elb_attachment" "kibana-elb" {
  elb      = "aws_elb.grafana-elb.id"
  instance = "aws_instance.mt_kibana_ec2.id"
}

resource "aws_route53_record" "kibana" {
  zone_id = data.aws_route53_zone.parent.id
  name    = "kibana-elb"
  type    = "A"

  alias {
    name                   = aws_elb.kibana-elb.dns_name
    zone_id                = aws_elb.kibana-elb.zone_id
    evaluate_target_health = true
  }
}
