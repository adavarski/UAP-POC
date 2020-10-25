<h1>Docker containers</h1>
<b>Ref:</b>

- [Mysql DockerHub](https://hub.docker.com/_/mysql)
- [Zabbix Server DockerHub](https://hub.docker.com/r/zabbix/zabbix-server-mysql)
- [Zabbix Frontend DockerHub](https://hub.docker.com/r/zabbix/zabbix-web-nginx-mysql)
- [Grafana](https://grafana.com/docs/grafana/latest/installation/docker/)

### Usage:
```
docker-compose up -d
Creating network "zabbix_zabbix" with driver "bridge"
Creating mysql ... done
Creating zabbix-frontend ... done
Creating zabbix-server   ... done
Creating grafana         ... done
Creating zabbix-agent    ... done

```

Zabbix Web UI http://{IP} —> (Credentials: User: Admin; Password: zabbix) : add agent 10.20.0.6 Groups ”Linux server” and assign “Template OS Linux by Zabbix agent”

Clean:

```
docker-compose down
```

