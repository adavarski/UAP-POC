#!/bin/bash

HIVE_HOME=$(pwd)/src/apache-hive-3.1.2-bin
MIRROR=https://repo1.maven.org/maven2

curl $MIRROR/org/apache/hadoop/hadoop-aws/3.1.1/hadoop-aws-3.1.1.jar -o $HIVE_HOME/lib/hadoop-aws-3.1.1.jar
curl $MIRROR/com/amazonaws/aws-java-sdk/1.11.406/aws-java-sdk-1.11.307.jar -o $HIVE_HOME/lib/aws-java-sdk-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-core/1.11.307/aws-java-sdk-core-1.11.307.jar -o $HIVE_HOME/lib/aws-java-sdk-core-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-dynamodb/1.11.307/aws-java-sdk-dynamodb-1.11.307.jar -o $HIVE_HOME/lib/aws-java-sdk-dynamodb-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-kms/1.11.307/aws-java-sdk-kms-1.11.307.jar -o $HIVE_HOME/lib/aws-java-sdk-kms-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-s3/1.11.307/aws-java-sdk-s3-1.11.307.jar -o $HIVE_HOME/lib/aws-java-sdk-s3-1.11.307.jar
curl $MIRROR/org/apache/httpcomponents/httpclient/4.5.3/httpclient-4.5.3.jar -o $HIVE_HOME/lib/httpclient-4.5.3.jar
curl $MIRROR/joda-time/joda-time/2.9.9/joda-time-2.9.9.jar -o $HIVE_HOME/lib/joda-time-2.9.9.jar
curl $MIRROR/mysql/mysql-connector-java/5.1.48/mysql-connector-java-5.1.48.jar -o $HIVE_HOME/lib/mysql-connector-java-5.1.48.jar
