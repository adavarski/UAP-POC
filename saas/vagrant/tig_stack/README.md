## TIG
======================================

### Overview

This repo it helps you quickly spin up a TIG monitoring stack (VMs: influxdb, grafana, telegraf), and setup TIG monitoring stack with ansible.

Note: All packages are from CentOS 7

### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started
```
yum install ansible (MacOS X: brew install ansible)
vagrant up (MacOS X: sudo vagrant up)

```
```
[root@dl360p30 TIG]# vagrant status
Current machine states:

influxdb                  running (virtualbox)
grafana                   running (virtualbox)
telegraf                  running (virtualbox)

```
### Import Grafana dashboards: example https://grafana.com/grafana/dashboards/10581

<img src="https://github.com/adavarski/SaaS-FULL-POC/blob/main/saas/vagrant/tig_stack/diagrams/TIG-vagrant-grafana-import-dashboard-host-10581.png?raw=true" width="650">

<img src="https://github.com/ExampleInc//TIG-vagrant-grafana.png?raw=true" width="650">



### References
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)
