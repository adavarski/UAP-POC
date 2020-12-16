# Apache Spark

This example shows how to deploy a stateless Apache Spark cluster with S3 support on Kubernetes. This is based on the "official" [kubernetes/spark](https://github.com/kubernetes/examples/tree/master/staging/spark) example, which also contains a few more details on the deployment steps.

## Deploying Spark on Kubernetes

Create a Docker Container and push to public repo

```
cd spark-s3
docker build -t davarski/spark-s3:2.3.0 .
docker login
docker push davarski/spark-s3:2.3.0
```

Deploy the Spark master Replication Controller and Service:

```
$ kubectl create -f 10-spark-master-controller.yaml
$ kubectl create -f 20-spark-master-service.yaml
$ kubectl create -f 50-ingress.yaml
```

Next, start your Spark workers:

```
$ kubectl create -f 60-spark-worker-controller.yaml
```

Let's wait until everything is up and running:

```
$ kubectl get all -n data|grep spark
pod/spark-master-controller-jrlv8       1/1     Running   0          106m
pod/spark-worker-controller-ql4lx       1/1     Running   0          88m
pod/spark-worker-controller-z8wv5       1/1     Running   0          88m
replicationcontroller/spark-master-controller   1         1         1       106m
replicationcontroller/spark-worker-controller   2         2         2       88m
service/spark-master-headless    ClusterIP   None            <none>        <none>                       88m
service/spark-master             ClusterIP   10.43.38.55     <none>        7077/TCP,8080/TCP            88m

$ kubectl get ing -n data|grep spark
Warning: extensions/v1beta1 Ingress is deprecated in v1.14+, unavailable in v1.22+; use networking.k8s.io/v1 Ingress
spark-ingress      <none>   spark.data.davar.com     192.168.0.100   80, 443   108s

```
Spark UI (ingress):


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-UI-ingress.png" width="800">


## Running queries against S3

Spark job against Demo5 Dataset: 
```
$ mc ls minio-cluster/iris
[2020-12-11 16:15:23 EET] 3.6KiB iris.csv
$ mc cat minio-cluster/iris/iris.csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3,1.4,0.2,setosa
...
```

Now, let's fire up a Spark shell and try out some commands:

```
kubectl exec -it spark-master-controller-jrlv8 -n data /bin/bash
export SPARK_DIST_CLASSPATH=$(hadoop classpath)
spark-shell
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://192.168.132.147:4040
Spark context available as 'sc' (master = spark://spark-master:7077, app id = app-20170405152342-0000).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.1.0
      /_/

Using Scala version 2.11.8 (OpenJDK 64-Bit Server VM, Java 1.8.0_111)
Type in expressions to have them evaluated.
Type :help for more information.

scala>
```

Excellent, now let's tell our Spark cluster the details of our S3 target, this will use https by default:

```
sc.hadoopConfiguration.set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")
sc.hadoopConfiguration.set("fs.s3a.access.key", "minio")
sc.hadoopConfiguration.set("fs.s3a.secret.key", "minio123")
sc.hadoopConfiguration.set("fs.s3a.path.style.access", "true")
```


```
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.IntParam
import org.apache.spark.sql.SQLContext
import org.apache.spark.graphx._
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

val conf = new SparkConf().setAppName("iris")
val sqlContext = new SQLContext(sc)

import sqlContext.implicits._
import sqlContext._

val speciesDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")

speciesDF.registerTempTable("species")

val fltCountsql = sqlContext.sql("select s.sepal_length,s.sepal_width from species s")
fltCountsql.show()
```
Example Output:

```
$ kubectl exec -it spark-master-controller-jrlv8 -n data /bin/bash
kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
root@spark-master-hostname:/# export SPARK_DIST_CLASSPATH=$(hadoop classpath)
root@spark-master-hostname:/# spark-shell
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://spark-master-hostname.spark-master-headless.data.svc.cluster.local:4040
Spark context available as 'sc' (master = spark://spark-master:7077, app id = app-20201215075314-0006).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.3.0
      /_/
         
Using Scala version 2.11.8 (OpenJDK 64-Bit Server VM, Java 1.8.0_111)
Type in expressions to have them evaluated.
Type :help for more information.

scala> sc.hadoopConfiguration.set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")

scala> sc.hadoopConfiguration.set("fs.s3a.access.key", "minio")

scala> sc.hadoopConfiguration.set("fs.s3a.secret.key", "minio123")

scala> sc.hadoopConfiguration.set("fs.s3a.path.style.access", "true")

scala> import org.apache.spark._
import org.apache.spark._

scala> import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD

scala> import org.apache.spark.util.IntParam
import org.apache.spark.util.IntParam

scala> import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SQLContext

scala> import org.apache.spark.graphx._
import org.apache.spark.graphx._

scala> import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.graphx.util.GraphGenerators

scala> import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint

scala> import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vectors

scala> import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.DecisionTree

scala> import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.model.DecisionTreeModel

scala> import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.util.MLUtils

scala> 

scala> val conf = new SparkConf().setAppName("iris")
conf: org.apache.spark.SparkConf = org.apache.spark.SparkConf@71687d8f

scala> val sqlContext = new SQLContext(sc)
warning: there was one deprecation warning; re-run with -deprecation for details
sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@1bbb48dd

scala> 

scala> import sqlContext.implicits._
import sqlContext.implicits._

scala> import sqlContext._
import sqlContext._

scala> 

scala> val speciesDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")
speciesDF: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 3 more fields]

scala> 

scala> speciesDF.registerTempTable("species")
warning: there was one deprecation warning; re-run with -deprecation for details

scala> 

scala> val fltCountsql = sqlContext.sql("select s.sepal_length,s.sepal_width from species s")
fltCountsql: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double]

scala> fltCountsql.show()
+------------+-----------+                                                      
|sepal_length|sepal_width|
+------------+-----------+
|         5.1|        3.5|
|         4.9|        3.0|
|         4.7|        3.2|
|         4.6|        3.1|
|         5.0|        3.6|
|         5.4|        3.9|
|         4.6|        3.4|
|         5.0|        3.4|
|         4.4|        2.9|
|         4.9|        3.1|
|         5.4|        3.7|
|         4.8|        3.4|
|         4.8|        3.0|
|         4.3|        3.0|
|         5.8|        4.0|
|         5.7|        4.4|
|         5.4|        3.9|
|         5.1|        3.5|
|         5.7|        3.8|
|         5.1|        3.8|
+------------+-----------+
only showing top 20 rows


scala> 


```

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-UI-app.png" width="800">

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-UI-worker.png" width="800">


## Further notes

This setup is a just an inital introduction on getting S3 working with Apache Spark on Kubernetes. Getting insights out of your data is the next step, but also optimizing performance is an important topic. For example, using Spark's `parallelize` call to parallelize object reads can yield massive performance improvements over using a simple `sc.textFiles(s3a://spark/*)` as used in this example.

## Executing Spark Applications
We use an interactive shell (spark-shell) currently. You can use an interactive shell (spark-shell or pyspark) or submit an application (spark-submit)
to execute Spark applications. Some prefer to use interactive web-based notebooks such as Apache Zeppelin and Jupyter to interact with Spark. Commercial vendors such as Databricks and Cloudera provide their own interactive notebook environment as well. I will use the spark-shell and Jupyter notebooks throughout the demo. 


