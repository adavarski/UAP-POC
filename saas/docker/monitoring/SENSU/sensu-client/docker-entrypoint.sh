#!/bin/sh

set -e

cat /tmp/sensu/conf.d/client.json \
  | sed "s/CLIENT_NAME/${CLIENT_NAME}/g" \
  | sed "s/CLIENT_IP/${CLIENT_IP}/g" > /etc/sensu/conf.d/client.json

cat /tmp/sensu/conf.d/rabbitmq.json \
  | sed "s/SENSU_HOST/${SENSU_HOST}/g" \
  | sed "s/SENSU_USER/${SENSU_USER}/g" \
  | sed "s/SENSU_PASSWORD/${SENSU_PASSWORD}/g" > /etc/sensu/conf.d/rabbitmq.json

/opt/sensu/bin/sensu-client &
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start sensu client: $status"
  exit $status
fi

while /bin/true; do
  sleep 60
done