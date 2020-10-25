# -*- mode: ruby -*-
# vim: set ft=ruby :
home = ENV['HOME']

MACHINES = {
  :dcs01 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.61', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :dcs02 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.62', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :dcs03 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.63', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :pg01 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.64', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :pg02 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.65', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :pg03 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.66', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :zoo01 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.67', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :zoo02 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.68', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :zoo03 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.69', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :kafka01 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.70', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :kafka02 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.71', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :kafka03 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.72', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :schemaregistry=> {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.73', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :kafkaconnect => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.74', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :influxdb => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.75', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :grafana => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.76', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :telegraf => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.77', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  }, 
  :elasticnode1 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.78', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :elasticnode2 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.79', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :elasticnode3 => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.80', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },
  :kibana => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.81', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  }, 
  :logstash => {
    :box_name => "centos/7",
    :net => [
               {ip: '10.51.21.82', adapter: 2, netmask: "255.255.255.0", virtualbox__intnet: "mc-net"},
            ]
  },  
}

Vagrant.configure("2") do |config|
  config.vm.define "dcs01" do |c|
    c.vm.hostname = "dcs01"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2251, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 8500, host: 4003, host_ip: '127.0.0.1'
  end
  config.vm.define "dcs02" do |c|
    c.vm.hostname = "dcs02"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2252, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 8500, host: 4004, host_ip: '127.0.0.1'
  end
  config.vm.define "dcs03" do |c|
    c.vm.hostname = "dcs03"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2253, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 8500, host: 4005, host_ip: '127.0.0.1'
  end
  config.vm.define "pg01" do |c|
    c.vm.hostname = "pg01"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2254, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "pg02" do |c|
    c.vm.hostname = "pg02"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2255, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "pg03" do |c|
    c.vm.hostname = "pg03"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2256, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "zoo01" do |c|
    c.vm.hostname = "zoo01"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2257, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "zoo02" do |c|
    c.vm.hostname = "zoo02"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2258, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "zoo03" do |c|
    c.vm.hostname = "zoo03"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2259, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "kafka01" do |c|
    c.vm.hostname = "kafka01"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2260, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "kafka02" do |c|
    c.vm.hostname = "kafka02"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2261, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "kafka03" do |c|
    c.vm.hostname = "kafka03"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2262, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "schemaregistry" do |c|
    c.vm.hostname = "schemaregistry"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2263, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "kafkaconnect" do |c|
    c.vm.hostname = "kafkaconnect"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2264, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "influxdb" do |c|
    c.vm.hostname = "influxdb"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2265, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 8086, host: 8086, id: "influxdb", host_ip: '127.0.0.1'
  end
  config.vm.define "grafana" do |c|
    c.vm.hostname = "grafana"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2266, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 3000, host: 3000, id: "grafana", host_ip: '127.0.0.1'
  end
  config.vm.define "telegraf" do |c|
    c.vm.hostname = "telegraf"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2267, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "elasticnode1" do |c|
    c.vm.hostname = "elasticnode1"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2268, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 9200, host: 9200, id: "elasticsearch", host_ip: '127.0.0.1'
  end
  config.vm.define "elasticnode2" do |c|
    c.vm.hostname = "elasticnode2"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2269, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "elasticnode3" do |c|
    c.vm.hostname = "elasticnode3"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2270, id: "ssh", host_ip: '127.0.0.1'
  end
  config.vm.define "kibana" do |c|
    c.vm.hostname = "kibana"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2271, id: "ssh", host_ip: '127.0.0.1'
    c.vm.network "forwarded_port", adapter: 1, guest: 5601, host: 5601, id: "kibana", host_ip: '127.0.0.1'
  end
  config.vm.define "logstash" do |c|
    c.vm.hostname = "logstash"
    c.vm.network "forwarded_port", adapter: 1, guest: 22, host: 2272, id: "ssh", host_ip: '127.0.0.1'
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
          v.memory = "768"
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
          ansible.playbook = "00_all.yml"
          ansible.inventory_path = "mc_dev_hosts_vagrant"
          ansible.extra_vars = "variables"
          ansible.become = "true"
          #ansible.tags = "update_hosts"
          #ansible.limit = "web"
          #ansible.config_file = "provisioning/ansible.cfg"
        end

      end
  end
end

