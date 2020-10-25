# DNS Public BALANCER (IPV4)
output "lb_address" {
  value = ["${aws_instance.lb_ec2.*.public_dns}"]
}

# IP Public BALANCER
output "lb_ip" {
  value = ["${aws_instance.lb_ec2.*.public_ip}"]
}

# DNS Public SERVISES:pg01 (IPV4)
output "svc_address" {
  value = ["${aws_instance.svc_pg01_ec2.*.public_dns}"]
}

# IP Public SERVICES: pg01
output "svc_ip" {
  value = ["${aws_instance.svc_pg01_ec2.*.public_ip}"]
}


# DNS Public MONITORING:TIG:grafana (IPV4)
output "mt_address_grafana" {
  value = ["${aws_instance.mt_grafana_ec2.*.public_dns}"]
}

# IP MONITORING:TIG:grafana
output "mt_ip_grafana" {
  value = ["${aws_instance.mt_grafana_ec2.*.public_ip}"]
}

# IP MONITORING:ELK:kibana
output "mt_ip_kibana" {
  value = ["${aws_instance.mt_kibana_ec2.*.public_ip}"]
}

# DNS Public ELB:kibana (IPV4)
output "elb_address_kibana" {
  value = ["${aws_elb.kibana-elb.dns_name}"]
}

