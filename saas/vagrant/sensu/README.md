
## Sensu 
======================================

### Overview

This repo it helps you quickly spin up a sensu server and a sensu client, and setup sensu monitoring with ansible.

Note: All packages are from CentOS 7

### Dependencies
1. [Virtualbox](https://www.virtualbox.org/wiki/Downloads)
2. [Vagrant](http://www.vagrantup.com/downloads.html)

### Getting started
```
yum install ansible
ansible-galaxy collection install sensu.sensu_go
vagrant up

```
[root@dl360p30 sensu]# vagrant status
Current machine states:

sensu                     running (virtualbox)
client                    running (virtualbox)

```
### References
* [Vagrant](http://vagrantup.com)
* [VirtualBox](http://www.virtualbox.org)
