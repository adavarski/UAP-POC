#!/bin/bash
set -e

# Count all running containers
running_containers=$(python /etc/sensu/plugins/docker_request.py "/containers/json?filters={\"status\": [\"running\"]}" \
    | grep \"Id\" \
    | wc -l)

# Count all containers
total_containers=$(python /etc/sensu/plugins/docker_request.py /containers/json?all=1 \
    | grep \"Id\" \
    | wc -l)

# Count all images
total_images=$(python /etc/sensu/plugins/docker_request.py /images/json \
    | grep \"Id\" \
    | wc -l)

exited_containers=$(python /etc/sensu/plugins/docker_request.py "/containers/json?filters={\"status\": [\"exited\"]}" \
    | grep \"Id\" \
    | wc -l)

restarting_containers=$(python /etc/sensu/plugins/docker_request.py "/containers/json?filters={\"status\": [\"restarting\"]}" \
    | grep \"Id\" \
    | wc -l)

echo "Running: ${running_containers}/${total_containers} \
- Exited: ${exited_containers} \
- Restarting: ${restarting_containers}"

# if [ ${running_containers} -lt 1 ]; then
#     exit 1;
# fi

if [ ${restarting_containers} -gt 0 ]; then
    exit 2;
fi

if [ ${exited_containers} -gt 0 ]; then
    exit 1;
fi