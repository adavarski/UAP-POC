### TIG stack (Telegraf/InfluxDB/Grafana)
[Telegraf](https://www.influxdata.com/time-series-platform/telegraf/) is a plugin-driven server agent for collecting and reporting metrics.  
[InfluxDB](https://www.influxdata.com/time-series-platform/influxdb/) handle massive amounts of time-stamped information.  
[Grafana](https://grafana.com/) is an open platform for beautiful analytics and monitoring.  

####  Environment (.env) 

`.env` file exposes environment variables:

* **TELEGRAF_HOST** - agent hostname
* **INFLUXDB_HOST** - database hostname
* **INFLUXDB_PORT** - database port
* **INFLUXDB_DATABASE** - database name
* **INFLUXDB_ADMIN_USER** - admin user
* **INFLUXDB_ADMIN_PASSWORD** - admin password
* **GRAFANA_PORT** - monitoring port
* **GRAFANA_USER** - monitoring user
* **GRAFANA_PASSWORD** - monitoring password
* **GRAFANA_PLUGINS_ENABLED** - enable monitoring plugins
* **GRAFANA_PLUGINS** - monitoring plugins list (fetch all available plugins if empty)

Modify it according to your needs.

Example .env: 

```
TELEGRAF_HOST=telegraf

INFLUXDB_HOST=influxdb
INFLUXDB_PORT=8086
INFLUXDB_DATABASE=metrics
INFLUXDB_ADMIN_USER=grafana
INFLUXDB_ADMIN_PASSWORD=grafana

GRAFANA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin
GRAFANA_PLUGINS_ENABLED=true
GRAFANA_PLUGINS=grafana-piechart-panel
```

#### Run TIG stack 

```bash
# rm -rf /var/lib/influxdb
# docker-compose up -d 
```
Note: rm -rf /var/lib/influxdb to clear on every stack new run.

#### Clean 
```bash
# docker-compose down
# docker rmi -f `docker images|grep grafana|awk '{print $3}'`
# docker rmi -f `docker images|grep influxdb|awk '{print $3}'`
# docker rmi -f `docker images|grep telegraf|awk '{print $3}'`

# rm -rf /var/lib/influxdb
```


Then access grafana at `http://{IP}:3000` if stack is running on remote server or `http://localhost:3000` if locally runned.


