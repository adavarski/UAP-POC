#!/bin/bash

set -x
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
PROJECTDIR="$(cd $BASEDIR/.. > /dev/null && pwd -P)"


cd ${PROJECTDIR}/
docker-compose -f ${PROJECTDIR}/saas/docker/docker-compose.yml up -d
cd ${PROJECTDIR}/saas/docker/monitoring/ELK/ && docker-compose up -d
cd ${PROJECTDIR}/saas/docker/monitoring/TIG/ && docker-compose up -d
cd ${PROJECTDIR}/saas/docker/monitoring/SENSU/ && docker-compose up -d
cd ${PROJECTDIR}/saas/docker/messaging/ && docker-compose up -d
cd ${PROJECTDIR}/saas/docker/postgresqlha/ && docker-compose up -d
