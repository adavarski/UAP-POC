# BigData: MinIO Data Lake Object Storage(S3) with Hive/Presto SQL-Engines

## Modern Data Warehouse

Data Lakes and Data Lake as Object Storage. Transactional databases, data warehouses, and data marts are all technologies that intend to store and retrieve data in known structures. Organizations often need to store new and varied types of data, often whose form is not known or suitable for structured data systems. The concept of managing this idea of unlimited data in any conceivable form is known as a Data Lake. Traditionally, filesystems and block storage solutions store most file-based data that an organization wishes to gather and maintain outside of its database management systems. Filesystems and block storage systems are challenging to scale, with varying degrees of fault tolerance, distribution, and limited support for metadata and analytics. HDFS (Hadoop Distributed File System) has been a popular choice for organizations needing the conceptual advantage of a Data Lake. HDFS is complicated to set up and maintain, typically requiring dedicated infrastructure and one or more experts to keep it operational and performant. This demo use a Data Lake with Object Storage, implemented with MinIO. MinIO provides a distributed, fault-tolerant object storage system compatible with Amazon S3. MinIO is horizontally scalable and supports objects up to five terabytes, with no limit to the number of objects it may store. These capabilities alone meet the basic conceptual requirements of a data lake. However, MinIO is extensible though its support for events and a powerful S3-compatible query system.

This demo on Data Warehouses extends a modern Data Lake with the distributed object storage system MinIO. Data Lakes store a wide variety of data forms, while Data Warehouses manage a wide variety of data sources. Data Warehouses provide access to data catalogs, metadata, indexes, key/value stores, message queues, event streams, document, and relational databases, including Data Lakes. The line between Data Lakes and Data Warehouses is not always clear; this demo distinguishes the concept of data warehousing as any managed collection of sources containing data that is either processed, organized, indexed, cataloged, or otherwise identified with a purpose. Open Source Data Lake management systems such as Delta Lake (https://delta.io/) bring ACID transactions, metadata, unified streaming, and batch data, while Kylo ( https://kylo.io/ ) offers a robust user interface for data ingestion, preparation, and discovery. These sophisticated Data Lake applications begin to blur the line between a vast, formless Data Lake and the well-organized, Warehouse. However, the results of these systems are likely candidates for higher-level Data Warehouse concepts. Data lakes are indiscriminate in their collection of data; when organizations acquire data of any kind, the need to store it may arise before the business case for its use. When the value and purpose for a set of data is understood, it may then be processed, schemas developed, attributes indexed, values normalized, and metadata catalogued for the awareness of interested services or human analysts. Data Warehouses expose access to real-time event and message data and collections of historical data, readied for decision support systems, business intelligence, analytics, machine learning, and inference.

This demo considers modern Data Warehouses and Data Lakes as an open (employing containerization), cloud-native platform, the cloud represented by Kubernetes (container orchestration), and the platform as an ever-growing collection of data management applications exposed through APIs and graphical user interfaces with the ability to deploy business logic within. Many organizations and applications require access to a variety of data sources, from common RDBMS databases to distributed document, object, and key stores—resulting from trends in Digital Transformation, IoT, and Data Science activities such as Machine Learning. Correlating data from various sources is a common practice; however, depending on the relationships between these sources, the process can be challenging. Migrating all data sources to a commercial Data Warehouse may be cost-prohibitive, impose unacceptable limitations, or result in vendor lock-in. Constructing a modern, cloud-native, vendor-neutral Data Warehouse on Kubernetes may open up new possibilities even alongside commercial applications and PaaS offerings. A tremendous amount of functionality and flexibility is achieved with little effort and capital, starting small with a near-limitless ability to scale.
This section adds Apache Hive to Kubernetes, applying new layers atop the data platform developed throughout this demo. Hive demonstrate the ability to represent  data sources such as MinIO (S3), creating a centralized data access point with distributed query execution.

## Hive Overview

Apache Hive is Data Warehouse software initially developed by Facebook and later given to the Apache Software Foundation. Organizations such as
Netflix and FINRA use Hive to query massive volumes of structured data across distributed storage systems, including Hadoop’s HDFS and Amazon
S3. Hive simplifies the complex MapReduce jobs typically required for querying Big Data by providing a standard SQL interface. While Hive is not
a database, it delivers the ability to project schema onto any structured data stored in HDFS or S3-compatible storage. Amazon’s AWS offers the
product Elastic MapReduce, including a version of Hive as a service. Apache Hive enables organizations to harness enormous quantities of
structured data not managed by formal database management systems, steady streams of IoT data, exports from legacy systems, and ad hoc data
ingestion. Apache Hive reduces the complexity and effort to perform Data Science activities, including business analytics, business intelligence, and
Machine Learning, by providing an SQL interface, metadata, and schema onto a vast Data Lake. 

## Create Hive docker image

This section creates a custom Apache Hive container configured to use MySQL for the storage of schema and metadata related to objects residing
in an S3-compatible distributed storage system, such as the MinIO cluster (configured before). Apache Hive, like many Big Data applications evolved outside the Cloud-Native and Kubernetes ecosystems, therefore requiring a bit more effort in onboarding it into the cluster. The following starts with building a custom container suitable for use with Kubernetes and local experimentation.


This repository is for the demonstration of Apache Hive utilizing a MySQL database for metadata storage, specifically for the projection of schema atop S3 object storage.

Custom Hive container build instructions (download and uncompress both Apache Hive and its main dependency Apache Hadoop and extend Apache Hive’s capabilities by adding JAR files containing the functionality needed for connecting to S3-compatible object storage and MySQL for schema and metadata management)

```shell script

cd ./hive ; mkdir ./src

# Download Apache Hive
curl -L http://mirror.cc.columbia.edu/pub/software/apache/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz -o ./src/apache-hive-3.1.2-bin.tar.gz

# Download Apache Hadoop
curl -L http://archive.apache.org/dist/hadoop/common/hadoop-3.1.2/hadoop-3.1.2.tar.gz -o ./src/hadoop-3.1.2.tar.gz

# Uncompress both
tar -xzvf ./src/apache-hive-3.1.2-bin.tar.gz -C ./src
tar -xzvf ./src/hadoop-3.1.2.tar.gz -C ./src

# Add Jars
./deps.sh

# create container
docker build -t davarski-hive-s3m:3.1.2 .
docker tag davarski-hive-s3m:3.1.2 davarski/hive-s3m:3.1.2-1.0.0
```
Note: Hive, like many Java-based applications, uses XML files for configuration, in this case, hive-site.xml. However, packaging configuration values containing
sensitive authentication tokens, passwords, and environment-specific services locations would be an anti-pattern causing security concerns and limiting
container reusability. Mounting a configuration file from a filesystem (or ConfigMaps in the case of Kubernetes) is a standard method of configuring
containers and provides considerable flexibility for admins or developers using the container; however, this method limits the ability to leverage values
from existing Secrets and ConfigMap values available in Kubernetes. The technique described in this section creates a configuration file template to be
populated by the container with environment variables at runtime (file named hive-site-template.xml). Shell script named entrypoint.sh as the container’s initial
process. The entry point script uses sed to replace values in the hive-site.xml configuration file with values from the environment variables passed in through the container runtime, defined in the previous section. After applying the configuration, the script runs the utility schematool to add any MySQL database and tables Hive requires to store schema and metadata. Finally, the entry point script starts both a Hive server and a Hive Metastore server.

## Local Hive Testing and publishing
This section tests the Hive container built in the previous section by creating a database and table schema mapped to the MinIO (S3) bucket test. Create the bucket test in the MinIO cluster. Later, Hive will be used to catalog object locations as data sources and project schema onto them. The following demonstrates the creation of a data source by creating a schema in Hive mapped to the empty bucket test:

### create MinIO bucket
```
$ mc mb minio-cluster/test1
```

### local test (remote S3:MinIO)
docker-compose up

Note: After starting Docker Compose, the new Apache Hive container connects to MySQL, creating a database and tables used to store schema
and metadata defined later. The Apache Hive container exposes three ports: HiveServer2 listens on port 10000, providing SQL access over thrift/JDBC; Hive Metastore listens on port 9083, allowing access to metadata and tables over the thrift protocol; and Hive provides a web interface on port 10002 for performance monitoring, debugging, and observation.

### connect to hive CLI
docker exec -it hive /opt/hive/bin/hive

### create table
```
CREATE DATABASE IF NOT EXISTS test;
CREATE EXTERNAL TABLE IF NOT EXISTS test.message (id int,message string) row format delimited fields terminated by ',' lines terminated by "\n" location 's3a://test/messages';
INSERT INTO test.message VALUES (1, "Test1");
SELECT * FROM test.message;
```

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Hive-local-workstation-testing.png" width="800">


Note: The previous test created a type of distributed database capable of cataloging and querying petabytes of data from a distributed, highly scalable MinIO object storage system. The preceding exercise is capable of modeling existing data, provided that all data in the specified bucket and prefix (/test/messages/) has the same structure. This powerful concept allows organizations to begin collecting structured data and apply a schema in the future, once the need to access it arises.

### Add container to registry:
```shell script
docker login
docker push davarski/hive-s3m:3.1.2-1.0.0
```

# Hive K8s deploy/test:

Brings the power of Apache Hive to the Kubernetes platform. Running Hive in Kubernetes brings all the advantages provided by container management, networking,
monitoring, and logical proximity to all the services within the data platform.

Deploys the custom Apache Hive container developed earlier. Hive supplies SQL-like capabilities atop Apache Hadoop, extending its use to a broader range of data analytics, analysis, and management applications. Hadoop’s Big Data capabilities are traditionally associated with the Hadoop Distributed File System (HDFS). However, the custom container developed earlier extends Hive with the ability to use S3-compatible object storage as a modern alternative to Hadoop’s HDFS. Apache Hive creates a Data Warehouse within a broader Data Lake, as shown:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Hive-warehousing-structures-and-semi-structures.png" width="800">

The following configuration defines a hive Kubernetes Service backed by a hive Deployment implementing the custom image davarski/hive-s3m:3.1.2-1.0.0. The new Hive container uses MySQL to store schema, defining structured and semi-structured objects stored in MinIO (S3).

```
kubectl apply -f ../003-data/3000-hive/10-mysql-metadata_backend.yml  
kubectl apply -f ../003-data/3000-hive/20-service.yml  
kubectl apply -f ../003-data/3000-hive/30-deployment.yml 
```

```
mc mb minio-cluster/test2
CREATE DATABASE IF NOT EXISTS test2;
CREATE EXTERNAL TABLE IF NOT EXISTS test2.message (id int,message string) row format delimited fields terminated by ',' lines terminated by "\n" location 's3a://test2/messages';
INSERT INTO test2.message VALUES (1, "Test1");
SELECT * FROM test2.message;
```

Example:

```
$ mc mb minio-cluster/test2
Bucket created successfully `minio-cluster/test2`.

$ kubectl get svc/hive -n data
NAME   TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)                        AGE
hive   ClusterIP   10.43.178.75   <none>        10000/TCP,9083/TCP,10002/TCP   36m

$ kubectl get pod -n data |grep hive
hive-dccc9f446-6wsg2                1/1     Running   0          36m

$ kubectl logs hive-dccc9f446-6wsg2 -n data
cp: cannot stat '/hive-site-template.xml': No such file or directory
Setting MySQL endpoint: mysql-service:3306
Setting S3 endpoint: http://minio-service:9000
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/opt/hive/lib/log4j-slf4j-impl-2.10.0.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/opt/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.25.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
Metastore connection URL:	 jdbc:mysql://mysql-service:3306/objectmetastore?createDatabaseIfNotExist=true&useSSL=false
Metastore Connection Driver :	 com.mysql.jdbc.Driver
Metastore connection User:	 root
Starting metastore schema initialization to 3.1.0
Initialization script hive-schema-3.1.0.mysql.sql
....
Initialization script completed
schemaTool completed
2020-12-11 08:55:12: Starting Hive Metastore Server
2020-12-11 08:55:12: Starting HiveServer2
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/opt/hive/lib/log4j-slf4j-impl-2.10.0.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/opt/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.25.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/opt/hive/lib/log4j-slf4j-impl-2.10.0.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/opt/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.25.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
Hive Session ID = 0f7fda0a-3345-4041-89f7-c2716b4615d5


$ kubectl exec -it hive-dccc9f446-6wsg2  /opt/hive/bin/hive -n data
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/opt/hive/lib/log4j-slf4j-impl-2.10.0.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/opt/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.25.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
Hive Session ID = 0824f4d1-f6c7-4db6-9939-dcd8475d4da5

Logging initialized using configuration in jar:file:/opt/hive/lib/hive-common-3.1.2.jar!/hive-log4j2.properties Async: true
Hive Session ID = 806c444e-b85c-4980-bb76-9713fb1e4393
Hive-on-MR is deprecated in Hive 2 and may not be available in the future versions. Consider using a different execution engine (i.e. spark, tez) or using Hive 1.X releases.
hive> CREATE DATABASE IF NOT EXISTS test2;
OK
Time taken: 0.76 seconds
hive> CREATE EXTERNAL TABLE IF NOT EXISTS test2.message (id int,message string) row format delimited fields terminated by ',' lines terminated by "\n" location 's3a://test2/messages';
OK
Time taken: 2.152 seconds
hive> INSERT INTO test2.message VALUES (1, "Test1");
Query ID = root_20201211092448_bc67f163-7b0f-4d93-b6a4-f5fad73c6383
Total jobs = 3
Launching Job 1 out of 3
Number of reduce tasks determined at compile time: 1
In order to change the average load for a reducer (in bytes):
  set hive.exec.reducers.bytes.per.reducer=<number>
In order to limit the maximum number of reducers:
  set hive.exec.reducers.max=<number>
In order to set a constant number of reducers:
  set mapreduce.job.reduces=<number>
Job running in-process (local Hadoop)
2020-12-11 09:24:53,986 Stage-1 map = 0%,  reduce = 0%
2020-12-11 09:24:54,997 Stage-1 map = 100%,  reduce = 100%
Ended Job = job_local137323622_0001
Stage-4 is selected by condition resolver.
Stage-3 is filtered out by condition resolver.
Stage-5 is filtered out by condition resolver.
Loading data to table test2.message
MapReduce Jobs Launched: 
Stage-Stage-1:  HDFS Read: 0 HDFS Write: 0 SUCCESS
Total MapReduce CPU Time Spent: 0 msec
OK
Time taken: 9.052 seconds
hive> INSERT INTO test2.message VALUES (2, "Test2");
Query ID = root_20201211092520_596cb1bf-1bdd-4381-b40f-53e6ef97128f
Total jobs = 3
Launching Job 1 out of 3
Number of reduce tasks determined at compile time: 1
In order to change the average load for a reducer (in bytes):
  set hive.exec.reducers.bytes.per.reducer=<number>
In order to limit the maximum number of reducers:
  set hive.exec.reducers.max=<number>
In order to set a constant number of reducers:
  set mapreduce.job.reduces=<number>
Job running in-process (local Hadoop)
2020-12-11 09:25:23,234 Stage-1 map = 0%,  reduce = 0%
2020-12-11 09:25:24,273 Stage-1 map = 100%,  reduce = 100%
Ended Job = job_local73448910_0002
Stage-4 is selected by condition resolver.
Stage-3 is filtered out by condition resolver.
Stage-5 is filtered out by condition resolver.
Loading data to table test2.message
MapReduce Jobs Launched: 
Stage-Stage-1:  HDFS Read: 0 HDFS Write: 0 SUCCESS
Total MapReduce CPU Time Spent: 0 msec
OK
Time taken: 6.661 seconds
hive> SELECT * FROM test2.message;
OK
1	Test1
2	Test2
Time taken: 0.312 seconds, Fetched: 2 row(s)
hive> exit;

$ mc ls minio-cluster/test2/messages
[2020-12-11 11:26:04 EET]     0B messages/
$ mc ls minio-cluster/test2/messages/
[2020-12-11 11:24:56 EET]     8B 000000_0
[2020-12-11 11:25:24 EET]     8B 000000_0_copy_1
$ mc cat minio-cluster/test2/messages/000000_0
1,Test1
$ mc cat minio-cluster/test2/messages/000000_0_copy_1
2,Test2


```

### Hive UI
```
$ kubectl port-forward hive-dccc9f446-6wsg2 10002:10002 -n data
Forwarding from 127.0.0.1:10002 -> 10002
Forwarding from [::1]:10002 -> 10002
Handling connection for 10002

```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Hive-UI.png" width="800">

### JupyterLab/Jupyter Notebook (Jupyter environment)

Pre: Create MinIO bucket: exports
```
$ mc mb minio-cluster/exports 
Bucket created successfully `minio-cluster/exports`.

```

Jupyter Notebooks are a browser-based (or web-based) IDE (integrated development environments)

Build custom JupyterLab docker image and pushing it into DockerHub container registry.
```
$ cd ./jupyterlab
$ docker build -t jupyterlab-eth .
$ docker tag jupyterlab-eth:latest davarski/jupyterlab-eth:latest
$ docker login 
$ docker push davarski/jupyterlab-eth:latest
```
Run Jupyter Notebook inside k8s as pod:

```
kubectl run -i -t jupyter-notebook --namespace=data --restart=Never --rm=true --env="JUPYTER_ENABLE_LAB=yes" --image=davarski/jupyterlab-eth:latest 

```
Example output:
```
davar@carbon:~$ export KUBECONFIG=~/.kube/k3s-config-jupyter 
davar@carbon:~$ kubectl run -i -t jupyter-notebook --namespace=data --restart=Never --rm=true --env="JUPYTER_ENABLE_LAB=yes" --image=davarski/jupyterlab-eth:latest
If you don't see a command prompt, try pressing enter.
[I 08:24:34.011 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 08:24:34.378 LabApp] Loading IPython parallel extension
[I 08:24:34.402 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 08:24:34.402 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[W 08:24:34.413 LabApp] JupyterLab server extension not enabled, manually loading...
[I 08:24:34.439 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 08:24:34.440 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 08:24:34.441 LabApp] Serving notebooks from local directory: /home/jovyan
[I 08:24:34.441 LabApp] The Jupyter Notebook is running at:
[I 08:24:34.441 LabApp] http://(jupyter-notebook or 127.0.0.1):8888/?token=5bebb78cc162e7050332ce46371ca3adc82306fac0bc082a
[I 08:24:34.441 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 08:24:34.451 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-7-open.html
    Or copy and paste one of these URLs:
        http://(jupyter-notebook or 127.0.0.1):8888/?token=5bebb78cc162e7050332ce46371ca3adc82306fac0bc082a
```

Once the Pod is running, copy the generated token from the output logs. Jupyter Notebooks listen on port 8888 by default. In testing and demonstrations such as this, it is common to port-forward Pod containers directly to a local workstation rather than configure Services and Ingress. Caution Jupyter Notebooks intend and purposefully allow remote code execution. Exposing Jupyter Notebooks to public interfaces requires proper security considerations.

Port-forward the test-notebook Pod with the following command: 
``
kubectl port-forward jupyter-notebook 8888:8888 -n data
``
Browse to http://localhost:8888//?token=5bebb78cc162e7050332ce46371ca3adc82306fac0bc082a

Preparing Test Data:

Note: Apache Hive provides the ability to project schema onto empty buckets, allowing for the creation of ad hoc yet well-structured data sets. While Hive is not itself a database, it can create massively scalable object-based databases atop distributed object storage, in this case the S3-compatible MinIO. Hive provides the ability to store schema supporting existing structured and semi-structured objects of a given type. The following exercise creates a new blood donor example data set, consisting of one million records distributed across one thousand CSV files. Each record contains the comma-separated values for email, name, blood type, birthday, and state of fictional donors.


```
!pip install Faker==2.0.3
!pip install minio==5.0.1

import os
import datetime
from faker import Faker
from minio import Minio
from minio.error import (ResponseError,
                         BucketAlreadyOwnedByYou,
                         BucketAlreadyExists)

fake = Faker()
def makeDonor():
    fp = fake.profile(fields=[
        "name",
        "birthdate",
        "blood_group"
    ])
    return (
        fake.ascii_safe_email(),
        fp["name"],
        fp["blood_group"],
        fp["birthdate"].strftime("%Y-%m-%d"),
        fake.state(),
    )

bucket = "exports"
mc = Minio('minio-service.data:9000',
            access_key='minio',
            secret_key='minio123',
            secure=False)
try:
    mc.make_bucket(bucket)
except BucketAlreadyOwnedByYou as err:
    pass
except BucketAlreadyExists as err:
    pass
except ResponseError as err:
    raise


for i in range(1,1001):
    now = datetime.datetime.now()
    dtstr = now.strftime("%Y%m%d%H%M%S")
    filename = f'donors/{dtstr}.csv'
    tmp_file = f'./{dtstr}.csv'
    with open(tmp_file,"w+") as tf:
        tf.write("email,name,type,birthday,state\n")
        for ii in range(1,1001):
            line = ",".join(makeDonor()) + "\n"
            tf.write(line)
        mc.fput_object(bucket, filename, tmp_file, content_type='application/csv')
    os.remove(tmp_file)
    print(f'{i:02}: {filename}')
```
Example Output:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Hive-MinIO-Juputer-1.png" width="800">

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Hive-MinIO-Jupyter-2.png" width="800">

Check MinIO bucket


```

$ mc ls minio-cluster/exports/donors/
[2020-12-11 13:58:20 EET]     5B 20201211115820.csv
[2020-12-11 13:58:21 EET]     5B 20201211115821.csv
[2020-12-11 14:06:42 EET]     5B 20201211120641.csv
[2020-12-11 14:06:42 EET]     5B 20201211120642.csv
[2020-12-11 14:10:41 EET]     5B 20201211121041.csv
$ mc cat minio-cluster/exports/donors/20201211115820.csv

```

Create Schema

```
CREATE DATABASE exports;
CREATE TABLE exports.donors (email string, name string, blood_type string, birthday date, state string) row format delimited fields terminated by ',' lines terminated by "\n" location 's3a://exports/donors';
SHOW DATABASES;
SHOW TABLES IN exports;
```
Example: 

```
$ kubectl exec -it hive-dccc9f446-6wsg2 /opt/hive/bin/hive -n data
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/opt/hive/lib/log4j-slf4j-impl-2.10.0.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/opt/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.25.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
Hive Session ID = 60fa7946-59aa-480a-83ac-acbd1c875544

Logging initialized using configuration in jar:file:/opt/hive/lib/hive-common-3.1.2.jar!/hive-log4j2.properties Async: true
Hive Session ID = 8f0e6b76-af4a-41ac-a556-6fdcb4fb8bb4
Hive-on-MR is deprecated in Hive 2 and may not be available in the future versions. Consider using a different execution engine (i.e. spark, tez) or using Hive 1.X releases.
hive> CREATE DATABASE exports;
OK
Time taken: 0.744 seconds
hive> CREATE TABLE exports.donors (email string, name string, blood_type string, birthday date, state string) row format delimited fields terminated by ',' lines terminated by "\n" location 's3a://exports/donors';
OK
Time taken: 2.187 seconds
hive> 

hive> select * from exports.donors;

```
Note: This demo uses a custom Apache Hive container to project schema onto the distributed object-store. While the single Hive container is capable of executing queries through ODBC/thrift exposed over the hive:1000 Kubernetes Service, a more extensive Hive cluster is necessary for executing production workloads directly against Hive. 

### Beeline CLI

Different ways how to connect
```
# from metastore (loopback) 
beeline -u jdbc:hive2://
    
# from hive-server (to metastore)
beeline -u "jdbc:hive2://localhost:10000/default;auth=noSasl" -n hive -p hive  

# exec script from file (example)
beeline -u jdbc:hive2:// -f /tmp/create-table.hql
```


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/diagram-beeline.png" width="600">

Note: Including Presto (see bellow) into picture

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/diagram-presto-hive.png" width="600">


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/beeline-hive-metastore.png" width="300">



Example: 

Create two csv files and cp to MinIO buckets: hive:

```
# Minio example
# mc config host add minio-cluster http://minio.data.davar.com:9000 minio minio123
mc mb minio-cluster/hive
mc cp ./dummy-data/iris.csv minio-cluster/hive/warehouse/iris/iris.csv
mc cp ./dummy-data/users.csv minio-cluster/hive/warehouse/users/users.csv
```

Example Output:

```
$ head -n5 dummy-data/users.csv 
1,Zik,Zhvu
2,John,Doe
3,Somename,Somelastname

$ head -n5 dummy-data/iris.csv 
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa

$ mc mb minio-cluster/hive
Bucket created successfully `minio-cluster/hive`.
$ mc cp ./dummy-data/iris.csv minio-cluster/hive/warehouse/iris/iris.csv
./dummy-data/iris.csv:                     3.63 KiB / 3.63 KiB ┃▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┃ 
$ mc cp ./dummy-data/users.csv minio-cluster/hive/warehouse/users/users.csv
./dummy-data/users.csv:                    46 B / 46 B ┃▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┃ 2.17 KiB/s 0s
```

Note: To create tables for test data, I use Option 1 in this example:

```
CREATE DATABASE warehouse;

CREATE EXTERNAL TABLE warehouse.users (id DECIMAL, name STRING, 
lastname STRING) 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
LOCATION 's3a://hive/warehouse/users/';

CREATE EXTERNAL TABLE warehouse.iris (sepal_length DECIMAL, sepal_width DECIMAL, 
petal_length DECIMAL, petal_width DECIMAL, species STRING) 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
LOCATION 's3a://hive/warehouse/iris/'
TBLPROPERTIES ("skip.header.line.count"="1");

```
Example Output:

```
$ kubectl exec -it hive-dccc9f446-6wsg2 bash -n data
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
root@hive-dccc9f446-6wsg2:/# /opt/hive/bin/beeline -u jdbc:hive2://
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/opt/hive/lib/log4j-slf4j-impl-2.10.0.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/opt/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.25.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
Connecting to jdbc:hive2://
Hive Session ID = 923ef7db-a6d5-471a-9a7f-79b10d30fdae
20/12/11 15:32:40 [main]: WARN session.SessionState: METASTORE_FILTER_HOOK will be ignored, since hive.security.authorization.manager is set to instance of HiveAuthorizerFactory.
20/12/11 15:32:41 [main]: WARN metastore.ObjectStore: datanucleus.autoStartMechanismMode is set to unsupported value null . Setting it to value: ignored
20/12/11 15:32:42 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:42 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:42 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:42 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:42 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:42 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:43 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:43 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:43 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:43 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:43 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
20/12/11 15:32:43 [main]: WARN DataNucleus.MetaData: Metadata has jdbc-type of null yet this is not valid. Ignored
Connected to: Apache Hive (version 3.1.2)
Driver: Hive JDBC (version 3.1.2)
Transaction isolation: TRANSACTION_REPEATABLE_READ
Beeline version 3.1.2 by Apache Hive
0: jdbc:hive2://> CREATE DATABASE warehouse;
20/12/11 15:32:52 [HiveServer2-Background-Pool: Thread-23]: WARN metastore.ObjectStore: Failed to get database hive.warehouse, returning NoSuchObjectException
OK
No rows affected (0.978 seconds)
0: jdbc:hive2://> CREATE EXTERNAL TABLE warehouse.users (id DECIMAL, name STRING, 
. . . . . . . . > lastname STRING) 
. . . . . . . . > ROW FORMAT DELIMITED 
. . . . . . . . > FIELDS TERMINATED BY ','
. . . . . . . . > LINES TERMINATED BY '\n'
. . . . . . . . > LOCATION 's3a://hive/warehouse/users/';
OK
No rows affected (1.589 seconds)
0: jdbc:hive2://> CREATE EXTERNAL TABLE warehouse.iris (sepal_length DECIMAL, sepal_width DECIMAL, 
. . . . . . . . > petal_length DECIMAL, petal_width DECIMAL, species STRING) 
. . . . . . . . > ROW FORMAT DELIMITED 
. . . . . . . . > FIELDS TERMINATED BY ','
. . . . . . . . > LINES TERMINATED BY '\n'
. . . . . . . . > LOCATION 's3a://hive/warehouse/iris/'
. . . . . . . . > TBLPROPERTIES ("skip.header.line.count"="1");
OK
No rows affected (0.158 seconds)
0: jdbc:hive2://> select * from warehouse.users;
OK
+-----------+-------------+-----------------+
| users.id  | users.name  | users.lastname  |
+-----------+-------------+-----------------+
| 1         | Zik         | Zhvu            |
| 2         | John        | Doe             |
| 3         | Somename    | Somelastname    |
+-----------+-------------+-----------------+
3 rows selected (1.527 seconds)
0: jdbc:hive2://> select * from warehouse.iris;
OK
+--------------------+-------------------+--------------------+-------------------+---------------+
| iris.sepal_length  | iris.sepal_width  | iris.petal_length  | iris.petal_width  | iris.species  |
+--------------------+-------------------+--------------------+-------------------+---------------+
| 5                  | 4                 | 1                  | 0                 | setosa        |
| 5                  | 3                 | 1                  | 0                 | setosa        |
| 5                  | 3                 | 1                  | 0                 | setosa        |
| 5                  | 3                 | 2                  | 0                 | setosa        |
| 5                  | 4                 | 1                  | 0                 | setosa        |
| 5                  | 4                 | 2                  | 0                 | setosa        |
| 5                  | 3                 | 1                  | 0                 | setosa        |
| 5                  | 3                 | 2                  | 0                 | setosa        |
| 4                  | 3                 | 1                  | 0                 | setosa        |
| 5                  | 3                 | 2                  | 0                 | setosa        |
| 5                  | 4                 | 2                  | 0                 | setosa        |
| 5                  | 3                 | 2                  | 0                 | setosa        |
| 5                  | 3                 | 1                  | 0                 | setosa        |
| 4                  | 3                 | 1                  | 0                 | setosa        |
| 6                  | 4                 | 1                  | 0                 | setosa        |
| 6                  | 4                 | 2                  | 0                 | setosa        |
...
...
...
+--------------------+-------------------+--------------------+-------------------+---------------+
150 rows selected (0.281 seconds)
0: jdbc:hive2://> show databases;
OK
+----------------+
| database_name  |
+----------------+
| default        |
| warehouse      |
+----------------+
2 rows selected (0.069 seconds)


```
## Presto

Constructing a modern Data Warehouse in Kubernetes provides an abstraction from low-level infrastructure, a unified control plane, standardized configuration management, holistic monitoring, role-based access control, network policies, and compatibility with the rapidly growing landscape of cloud-native technologies.
This Presto demo configures two new data sources: MySQL representing a common RDBMS database, Apache Hive capable of projecting a schema atop the S3-compatible object storage set up in the previous sections (TODO: add data source: Apache Cassandra as a wide-column distributed NoSQL database). Presto, a distributed SQL query engine for Big Data, ties these existing data sources together into a single catalog, providing schema and connectivity. Presto natively supports over 20 typical data applications, including Elasticsearch and Apache Kafka (deployed using this repo). It is not uncommon to write an application that natively connects and consumes data from more than one source. However, technologies such as Presto consolidate and abstract this capability, distribute queries across a cluster to workers, aggregate results, and monitor performance. Centralized access to a vast warehouse of data from Presto reduces technical debt across specialized systems by managing diverse connectivity requirements and schema management.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/7-DWH_Presto_distributed_SQL_joining_multiple_datasources.png" width="800">


Presto is the final component of the modern Data Warehouse in this demo. According to the official website prestodb.io, “Presto is an open source distributed SQL query engine for running interactive analytic queries against data sources of all sizes ranging from gigabytes to petabytes.” Although Hive is also a distributed SQL query engine cable of querying vast quantities of data, Presto connects to a broader range of data sources, including Apache Hive. Aside from Presto’s high-performance querying capabilities, it provides a central catalog of data sources.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/7-DWH_Presto_distributed_SQL_query_multiple_datasources.png" width="800">

Presto reduces the amount of application logic needed to retrieve data from multiple sources, both through a standard SQL abstraction and removing the need for the client-side joining of data (in some cases considered an anti-pattern). Presto provides SQL abstraction across all its supported data sources, performs distributed query execution, and includes monitoring and observability. Presto supports client libraries for Go, C, Java, Node.js, PHP, Ruby, R, and Python. A growing set of web-based GUI clients, visualization, and dashboard applications support Presto, including the new business intelligence application Apache
Superset, from the creators of Apache Airflow.

This demo installs a Presto cluster with one workers and a coordinator in Kubernetes using a stable open source Helm chart by When I Work Data. 
Create a file named values.yml with content.  In Presto, a catalog represents a top-level data source. Note the three data sources (known as connectors) defined in the catalog section of the Helm chart configuration values.yml. The first two, obj.properties and hive.properties, use the hive-hadoop2 connector. Presto uses Hive for access to data files (objects) contained in HDFS or S3 and the Hive Metastore service for metadata and schema representing the data files.
The hive.properties configuration demonstrates the use of the custom Apache Hive container (installed in the previous section) for its Metastore
service backed by MySQL. Additionally, the mysql.properties demonstrate connections to MySQL as configured in this demo. Create a new Presto cluster by applying the Helm chart clone, along with custom configuration from values.yml:
```
cd ../003-data/4000-presto/
git clone git@github.com:apk8s/presto-chart.git
helm upgrade --install presto-data --namespace data --values values.yml ./presto-chart/presto
```
Example Output:
```
$ helm upgrade --install presto-data --namespace data --values values.yml ./presto-chart/presto
Release "presto-data" does not exist. Installing it now.
coalesce.go:160: warning: skipped value for environmentVariables: Not a table.
NAME: presto-data
LAST DEPLOYED: Fri Dec 11 19:20:54 2020
NAMESPACE: data
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
Get the application URL by running these commands:
  export POD_NAME=$(kubectl get pods --namespace data -l "app=presto,release=presto-data,component=coordinator" -o jsonpath="{.items[0].metadata.name}")
  echo "Visit http://127.0.0.1:8080 to use your application"
  kubectl port-forward $POD_NAME 8080:8080
```

Once Helm completes the install process, the Kubernetes cluster contains one Presto worker nodes and one Presto coordinator. Finally, add a Kubernetes Ingress configuration backed by the new presto-data:80 service generated by the Helm chart. 

```
kubectl apply -f 50-ingress.yml
```

Presto simple tests:

```
SHOW CATALOGS [ LIKE pattern ]
SHOW SCHEMAS [ FROM catalog ] [ LIKE pattern ]
SHOW TABLES [ FROM schema ] [ LIKE pattern ]
...
...
```

```
$ kubectl exec -it pod/presto-data-coordinator-64f7ffbb99-65ggt bash -n data
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
$ kubectl exec -it pod/presto-data-coordinator-64f7ffbb99-65ggt bash -n data
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.

presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ echo "SHOW CATALOGS;" > query-example.sql
presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ presto -f ./query-example.sql
"hive"
"mysql"
"obj"
"system"

presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ echo "SELECT * FROM system.runtime.nodes;" > query-example.sql
presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ presto -f ./query-example.sql
"presto-data-coordinator-64f7ffbb99-65ggt","http://10.42.0.45:8080","0.217","true","active"
"presto-data-worker-678564cfc5-q4xnc","http://10.42.0.46:8080","0.217","false","active"

presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ echo "DESCRIBE hive.warehouse.iris;" > query-example.sql
presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ presto -f ./query-example.sql
"sepal_length","decimal(10,0)","",""
"sepal_width","decimal(10,0)","",""
"petal_length","decimal(10,0)","",""
"petal_width","decimal(10,0)","",""
"species","varchar","",""
presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ echo "DESCRIBE hive.warehouse.users;" > query-example.sql
presto@presto-data-coordinator-64f7ffbb99-65ggt:~$ presto -f ./query-example.sql
"id","decimal(10,0)","",""
"name","varchar","",""
"lastname","varchar","",""

presto@presto-data-coordinator-64f7ffbb99-5hlrb:~$ echo "SELECT * from hive.warehouse.users;" > query-example.sql
presto@presto-data-coordinator-64f7ffbb99-5hlrb:~$ presto -f ./query-example.sql
"1","Zik","Zhvu"
"2","John","Doe"
"3","Somename","Somelastname"

presto@presto-data-coordinator-64f7ffbb99-5hlrb:~$ echo "SELECT * from hive.warehouse.iris;" > query-example.sql
presto@presto-data-coordinator-64f7ffbb99-5hlrb:~$ presto -f ./query-example.sql
"5","4","1","0","setosa"
"5","3","1","0","setosa"
"5","3","1","0","setosa"
"5","3","2","0","setosa"
"5","4","1","0","setosa"
"5","4","2","0","setosa"
"5","3","1","0","setosa"
"5","3","2","0","setosa"
"4","3","1","0","setosa"
"5","3","2","0","setosa"
"5","4","2","0","setosa"
"5","3","2","0","setosa"
"5","3","1","0","setosa"
...
...

```
Jupyter Notebook:

```
!pip install presto-python-client==0.7.0
```
```
import prestodb
import os
import pandas as pd
```
```
conn=prestodb.dbapi.connect(
     host='presto-data.data',
     port=80,
     user=os.environ['HOSTNAME']
)
cur = conn.cursor()

```
```
cur.execute('SHOW CATALOGS')
rows = cur.fetchall()
rows
```
```
cur.execute('SELECT * FROM system.runtime.nodes')
rows = cur.fetchall()

df = pd.DataFrame(
    rows, 
    columns=[d[0] for d in cur.description]
)

df
```
```
cur.execute('DESCRIBE hive.warehouse.users')
rows = cur.fetchall()

df = pd.DataFrame(
    rows, 
    columns=[d[0] for d in cur.description]
)
df
```
```
cur.execute('SELECT * from hive.warehouse.users')
rows = cur.fetchall()

df = pd.DataFrame(
    rows, 
    columns=[d[0] for d in cur.description]
)
df
```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Presto-JupyterNotebook.png" width="800">



An Ingress configuration, exposes the Presto UI at https://presto.data.davar.com as depicted in: 


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Presto-UI-queries-jupyter.png" width="800">

Monitoring and observability are critical, for both Big Data and web-scale data operations. The Presto web user interface supports drill-downs into each query providing query details including resource utilization, timeline, error information, stages, and tasks related to the execution. Additionally, Presto provides a Live Plan, as shown bellow depicting the execution flow between stages in real time through a network diagram.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo5-BigData-MinIO-Hive-Presto/pictures/Presto-query-liveplan.png" width="800">

Presto is a comprehensive solution for building a modern Data Warehouse within Kubernetes; its support for a range of data sources fits the growing needs of IoT and Machine Learning, providing the ability to retrieve, coalesce, correlate, transform, and analyze limitless quantities and structures of data.

