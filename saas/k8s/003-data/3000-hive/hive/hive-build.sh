#!/bin/bash
mkdir ./src
curl -L http://mirror.cc.columbia.edu/pub/software/apache/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz -o ./src/apache-hive-3.1.2-bin.tar.gz
curl -L http://archive.apache.org/dist/hadoop/common/hadoop-3.1.2/hadoop-3.1.2.tar.gz -o ./src/hadoop-3.1.2.tar.gz
tar -xzvf ./src/apache-hive-3.1.2-bin.tar.gz -C ./src
tar -xzvf ./src/hadoop-3.1.2.tar.gz -C ./src

# extend Apache Hive’s capabilities by adding JAR files containing
# the functionality needed for connecting to S3-compatible object storage
# and MySQL for schema and metadata management:

export HIVE_LIB=$(pwd)/src/apache-hive-3.1.2-bin/lib
export MIRROR=https://repo1.maven.org/maven2
curl $MIRROR/org/apache/hadoop/hadoop-aws/3.1.1/hadoop-aws-3.1.1.jar -o $HIVE_LIB/hadoop-aws-3.1.1.jar
curl $MIRROR/com/amazonaws/aws-java-sdk/1.11.406/aws-java-sdk-1.11.307.jar -o $HIVE_LIB/aws-java-sdk-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-core/1.11.307/aws-java-sdk-core-1.11.307.jar -o $HIVE_LIB/aws-java-sdk-core-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-dynamodb/1.11.307/aws-java-sdk-dynamodb-1.11.307.jar -o $HIVE_LIB/aws-java-sdk-dynamodb-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-kms/1.11.307/aws-java-sdk-kms-1.11.307.jar -o $HIVE_LIB/aws-java-sdk-kms-1.11.307.jar
curl $MIRROR/com/amazonaws/aws-java-sdk-s3/1.11.307/aws-java-sdk-s3-1.11.307.jar -o $HIVE_LIB/aws-java-sdk-s3-1.11.307.jar
curl $MIRROR/org/apache/httpcomponents/httpclient/4.5.3/httpclient-4.5.3.jar -o $HIVE_LIB/httpclient-4.5.3.jar
curl $MIRROR/joda-time/joda-time/2.9.9/joda-time-2.9.9.jar -o $HIVE_LIB/joda-time-2.9.9.jar
curl $MIRROR/mysql/mysql-connector-java/5.1.48/mysql-connector-java-5.1.48.jar -o $HIVE_LIB/mysql-connector-java-5.1.48.jar
