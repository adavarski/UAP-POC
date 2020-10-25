Docker compose file for setting up a EFK service
================================================

Set up Elasticsearch, Fluentd, and Kibana.

Example
-------
```
docker-compose up -d
Creating network "docker_default" with the default driver
Creating docker_elasticsearch_1 ... done
Creating docker_fluentd_1       ... done
Creating docker_kibana_1        ... done

```
Then, go to your browser and access `http://localhost:5601` (kibana). Index is : `fluentd-*`. You should be able to see the logs in kibana's discovery tab. 

After you are done, just run:

    docker-compose down


