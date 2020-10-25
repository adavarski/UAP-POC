### TBD: ansible playbooks & roles for MC/NOC services
ansible-playbook --inventory mc_dev --extra-vars "@variables" 00_all.yml

ansible-playbook -vvvv -i mc_dev 00_all.yml > failure.txt

ansible-playbook -i mc_dev 00_all.yml --tags=consul
