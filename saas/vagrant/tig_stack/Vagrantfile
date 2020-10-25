# -*- mode: ruby -*-
# vim: set ft=ruby :
home = ENV['HOME']

MACHINES = {
  :influxdb => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.61', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "monitoring-net"},
            ]
  },
  :grafana => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.62', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "monitoring-net"},
            ]
  },
  :telegraf => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.63', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "monitoring-net"},
            ]
  },  
}

Vagrant.configure("2") do |config|

  config.vm.define "influxdb" do |c|
    c.vm.hostname = "influxdb"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2251, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 8086, host: 8086, id: "influxdb", host_ip: '127.0.0.1'
  end
  config.vm.define "grafana" do |c|
    c.vm.hostname = "grafana"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2252, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 3000, host: 3000, id: "grafana", host_ip: '127.0.0.1'
  end
  config.vm.define "telegraf" do |c|
    c.vm.hostname = "telegraf"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2253, id: "ssh", host_ip: '127.0.0.1'
  end


  MACHINES.each do |boxname, boxconfig|

    config.vm.define boxname do |box|

        box.vm.box = boxconfig[:box_name]
        box.vm.box_check_update = false

        boxconfig[:net].each do |ipconf|
          box.vm.network "private_network", ipconf
        end

        if boxconfig.key?(:public)
          box.vm.network "public_network", boxconfig[:public]
        end

        box.vm.provider "virtualbox" do |v|
          v.customize ["modifyvm", :id, "--audio", "none"]
          v.memory = "2048"
          v.cpus = "1"
        end

        box.vm.provision "shell", inline: <<-SHELL
                mkdir -p ~root/.ssh
                cp ~vagrant/.ssh/auth* ~root/.ssh
                sed -i 's/^PasswordAuthentication no/#PasswordAuthentication no/g' /etc/ssh/sshd_config
                sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config
                systemctl restart sshd
        SHELL

        box.vm.provision "ansible" do |ansible|
          ansible.verbose = "v"
          ansible.playbook = "provisioning/00_all.yml"
          ansible.inventory_path = "provisioning/hosts_vagrant"
          ansible.extra_vars = "provisioning/variables"
          ansible.become = "true"
          #ansible.tags = "update_hosts"
          #ansible.limit = "web"
          #ansible.config_file = "provisioning/ansible.cfg"
        end

      end
  end
end
