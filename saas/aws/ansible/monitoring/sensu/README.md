### SaaS Sensu Monitoring

All SaaS infrastructure we have has to be monitored. We have to monitor AWS VMs and NOC services are UP and running and as much as possible all servers metrics, depending of server role: CPU/Memory/Swap/Disk IO/Free disk space/processes&open files/security metrics/services/etc. If there is some problem/issue e-mail/slack message/etc. will be sended by Infrastructure monitoring system (Sensu) to NOC/SRE/DevOps/Infrastructure teams to prevent infrastructure failures and issues.


### HOWTO Install & Configure

1.Create ansible host on some of on-prem infra servers or create AWS VM for ansible host with terraform and install ansible 

```
# yum install ansible -y

```
3. Copy all files from this git repo directory to @ansible host:/etc/ansible and configure ansible.cfg, ansible inventory (hosts), variables and ansible playbooks (sensu-infra-monitoring.yaml, etc.)

4. Install Sensu Go Ansible Collection 

```
ansible-galaxy collection install sensu.sensu_go
```

5. Create AWS VM with terraform for sensu server and create Route53 DNS record.

DNS: monitoring-sensu.example.com XXX.XXX.XXX.XXX (Terraform) 

5. Install sensu server on AWS sensu monitoring VM (IP:XXX.XXX.XXX.XXX)

On ansible host:
```
# cd ~; ssh-keygen -t rsa; cd .ssh; ssh-copy-id -i id_rsa.pub XXX.XXX.XXX.XXX
# cd /etc/ansible
# ansible-playbook -i hosts -l backends sensu-infra-monitoring.yaml 
```

So will have sensu installed: @monitoring-sensu.example.com (Dashboard is available at http://monitoring-sensu.e.com:3000)

Note: Change sensu administrator password and create users for NOC/SRE team.

5.Add AWS VMs/services for monitoring 

On ansible host:

```
Example1 : Add dev-infra host group for monitoring.

Install agents on dev-infra AWS VMs 

cd ~/.ssh

# ssh-copy-id -i id_rsa.pub AAA.AAA.AAA.AAA
# ssh-copy-id -i id_rsa.pub BBB.BBB.BBB.BBB
# ssh-copy-id -i id_rsa.pub CCC.CCC.CCC.CCC

# cd /etc/ansible

# ansible-playbook -i hosts -l dev-infra --extra-vars "@variables" sensu-infra-monitoring.yaml 

Example2: Add dev-infra @sensu
for i in `awk '/dev-infra/{flag=1;next}/dev-infra/{flag=0}flag' hosts`;do ssh-copy-id -i ~/.ssh/id_rsa.pub $i;done
ansible-playbook -i hosts -l dev-infraa sensu-infra-monitoring.yaml


Example3: Add new host @dev-infra host group

- add host to /etc/ansible/hosts : host group:  dev-infra

[dev-infra]
....
DDD.DDD.DDD.DDD

- ssh-copy-id -i id_rsa.pub DDD.DDD.DDD.DDD

- Install agent on server: 

ansible-playbook -i hosts --limit "DDD.DDD.DDD.DDD" --extra-vars "@variables" sensu-infra-monitoring.yaml.yaml
```


