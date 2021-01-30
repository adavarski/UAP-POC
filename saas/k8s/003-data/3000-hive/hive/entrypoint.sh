#!/bin/bash

# provide ample time for other
# services to come online and
sleep 10

# replace hive-site.xml values with environment
# variables
HIVE_CONF_TEMPLATE="/opt/hive/conf/hive-site-template.xml"
HIVE_CONF="/opt/hive/conf/hive-site.xml"

cp $HIVE_CONF_TEMPLATE $HIVE_CONF
echo "Setting MySQL endpoint: $MYSQL_ENDPOINT"
echo "Setting S3 endpoint: $S3A_ENDPOINT"

# template replacements. using //\//\\/ to escape slashes
sed -i'' "s/MYSQL_ENDPOINT/${MYSQL_ENDPOINT//\//\\/}/g" $HIVE_CONF
sed -i'' "s/MYSQL_USER/${MYSQL_USER//\//\\/}/g" $HIVE_CONF
sed -i'' "s/MYSQL_PASSWORD/${MYSQL_PASSWORD//\//\\/}/g" $HIVE_CONF
sed -i'' "s/S3A_ENDPOINT/${S3A_ENDPOINT//\//\\/}/g" $HIVE_CONF
sed -i'' "s/S3A_ACCESS_KEY/${S3A_ACCESS_KEY//\//\\/}/g" $HIVE_CONF
sed -i'' "s/S3A_SECRET_KEY/${S3A_SECRET_KEY//\//\\/}/g" $HIVE_CONF
sed -i'' "s/S3A_PATH_STYLE_ACCESS/${S3A_PATH_STYLE_ACCESS//\//\\/}/g" $HIVE_CONF

# add metastore schema to mysql
$HIVE_HOME/bin/schematool -dbType mysql -initSchema
$HIVE_HOME/bin/hiveserver2 start & # port 10000
$HIVE_HOME/bin/hiveserver2 --service metastore # port 9083
