#cloud-config
runcmd:
# get/setup instance name
 - hostnamectl set-hostname `curl -s http://169.254.169.254/latest/meta-data/public-hostname`
# configure 
 - sudo su - devops -c 'secure-host.xx'
