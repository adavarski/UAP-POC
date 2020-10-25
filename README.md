# SaaS DEV environmints:

- docker-compose based dev env (Note: default dev env) 
- Vagrant based dev env (Vagrant: IaC for simulating public/private cloud envs (AWS, Azure, GCP, OCP, OpenStack, etc.) : very flexible, ruby syntax, better than OpenStack (KVM) for DEV, you can use vagrant with VBOX/KVM/etc. and with packer (create Vagrant boxes for Vbox/KVM/etc. like AWS AMIs creation with packer)
- k8s (k8s-local: minikube, kubespray)
- AWS (new SAAS account) —> TF for provisioning infrastructure && ansible for CM 

