### SaaS NOC/MC TIG monitoring

All SaaS MC/NOC AWS infrastructure we have has to be monitored. We have to monitor AWS VMs and NOC services are UP and running and as much as possible all servers metrics, depending of server role: CPU/Memory/Swap/Disk IO/Free disk space/processes&open files/security metrics/services/etc. If there is some problem/issue e-mail/slack message/etc. will be sended by Infrastructure monitoring system (TIG stack) to NOC/SRE/DevOps/Infrastructure teams to prevent infrastructure failures and issues.


### HOWTO Install & Configure

1.Create ansible host on some of on-prem infra servers or create AWS VM for ansible host with terraform and install ansible 

```
# yum install ansible -y

```
2. Copy all files from this git repo directory to @ansible host:/etc/ansible and configure ansible.cfg, ansible inventory (hosts), variables and ansible playbooks (tig-infra-monitoring.yaml, etc.)

3. Create AWS VM with terraform for influxdb and grafana and create Route53 DNS record.

DNS: monitoring-grafana.noc.infra.saas.com XXX.XXX.XXX.XXX (Terraform) 

4. Install TIG (influxDB, grafana) servers on AWS for monitoring VM (IP:influxdb:XXX.XXX.XXX.XXX, IP:influxdb:YYY.YYY.YYY.YYY)

On ansible host:
```
# cd ~; ssh-keygen -t rsa; cd .ssh; ssh-copy-id -i id_rsa.pub XXX.XXX.XXX.XXX; ssh-copy-id -i id_rsa.pub YYY.YYY.YYY.YYY
# cd /etc/ansible
# ansible-playbook -i hosts -l influxdb tig-infra-monitoring.yaml 
# ansible-playbook -i hosts -l grafana tig-infra-monitoring.yaml
```

So will have grafana installed: @monitoring-grafana.noc.infra.saas.com (Dashboard is available at http://monitoring-grafana.noc.infra.saas.com:3000)

Note: Change grafana administrator password and create users for NOC/SRE team.

5.Add NOC/MC AWS VMs/services for monitoring (telegraf):

On ansible host:

```
Example1 : Add noc-dev-infraapi host group for monitoring.

Install telegraf agents on telegraf AWS VMs 

cd ~/.ssh

# ssh-copy-id -i id_rsa.pub AAA.AAA.AAA.AAA
# ssh-copy-id -i id_rsa.pub BBB.BBB.BBB.BBB
# ssh-copy-id -i id_rsa.pub CCC.CCC.CCC.CCC

# cd /etc/ansible

# ansible-playbook -i hosts -l telegraf --extra-vars "@variables" tig-infra-monitoring.yaml 

Example3: Add new host @telegraf host group

- add host to /etc/ansible/hosts[telegraf] : host group:  telegraf

[telegraf]
....
DDD.DDD.DDD.DDD

- ssh-copy-id -i id_rsa.pub DDD.DDD.DDD.DDD

- Install agent on server: 

ansible-playbook -i hosts --limit "DDD.DDD.DDD.DDD" --extra-vars "@variables" tig-infra-monitoring.yaml.yaml
```


Smokeping AWS NOC: http://monitoring-sensu.noc.infra.saas.com:8888
