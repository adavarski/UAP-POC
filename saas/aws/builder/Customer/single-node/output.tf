output "instance_ips" {
  value = [aws_instance.demo-tf.*.public_ip]
}

output "ip" {
  value       = aws_instance.demo-tf.public_dns
  description = "The URL of the server instance."
}

