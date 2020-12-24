
AI, ML and Deep ML:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-ML-overview.png" width="800">

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


# Introduction to Spark and Spark MLlib

Spark is a unified big data processing framework for processing and analyzing large datasets. Spark provides high-level APIs in Scala, Python, Java, and R with powerful libraries including MLlib for machine learning, Spark SQL for SQL support, Spark Streaming for real-time streaming, and GraphX for graph processing. ii Spark was founded by Matei Zaharia at the University of California, Berkeley’s AMPLab and was later donated to the Apache Software Foundation, becoming a top-level project on February 24, 2014. iii The first version was released on May 30, 2017.


## Overview
Spark was developed to address the limitations of MapReduce, Hadoop’s original data processing framework. Matei Zaharia saw MapReduce’s limitations at UC Berkeley and Facebook (where he did his internship) and sought to create a faster and more generalized, multipurpose data processing framework that can handle iterative and interactive applications. v It provides a unified platform that supports multiple types of workloads such as streaming, interactive, graph processing, machine learning, and batch. Spark jobs can run multitude of times faster than equivalent MapReduce jobs due to its fast in-memory capabilities and advanced DAG (directed acyclic graph) execution engine. Spark was written in Scala and consequently it is the de facto programming interface for Spark. We will use Scala throughout this demo. We will use PySpark, the Python API for Spark, for distributed deep learning. 


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-ML-Apache_Spark_ecosystem.png" width="800">


-Spark Core
Spark Core is the most fundamental building block of Spark as shown in above figure. It is the backbone of Spark’s supreme functionality features. Spark Core enables the in-memory computations that drive the parallel and distributed processing of data. All the features of Spark are built on top of Spark Core. Spark Core is responsible for managing tasks, I/O operations, fault tolerance, and memory management, etc.

-Spark SQL
This component mainly deals with structured data processing. The key idea is to fetch more information about the structure of the data to perform additional optimization. It can be considered a distributed SQL query engine.

-Spark Streaming
This component deals with processing the real-time streaming data in a scalable and fault tolerant manner. It uses micro batching to read and process incoming streams of data. It creates micro batches of streaming data, executes batch processing, and passes it to some file storage or live dashboard. Spark Streaming can ingest the data from multiple sources like Kafka and Flume.

-Spark MLlib
This component is used for building Machine Learning Models on Big Data in a distributed manner. The traditional technique of building ML models using Python’s scikit learn library faces lot of challenges when data size is huge whereas MLlib is designed in a way that offers feature engineering and machine learning at scale. MLlib has most of the algorithms implemented for classification, regression, clustering, recommendation system, and natural language processing.

-Spark GraphX/Graphframe
This component excels in graph analytics and graph parallel execution. Graph frames can be used to understand the underlying relationships and visualize the insights from data.


## Cluster Managers
Cluster managers manage and allocate cluster resources. Spark supports the standalone cluster manager that comes with Spark (Standalone Scheduler), YARN, Mesos, and Kubernetes.


## Architecture
At a high level, Spark distributes the execution of Spark applications’ tasks across the cluster nodes (see below figure). Every Spark application has a SparkContext object within its driver program. The SparkContext represents a connection to your cluster manager, which provides computing resources to your Spark applications. After connecting to the cluster, Spark acquires executors on your worker nodes. Spark then sends your application code to the executors. An application will usually run one or more jobs in response to a Spark action. Each job is then divided by Spark into smaller directed acyclic graph (DAG) of stages or tasks. Each task is then distributed and sent to executors across the worker nodes for execution.

Each Spark application gets its own set of executors. Because tasks from different applications run in different JVMs, a Spark application cannot interfere with another Spark application. This also means that it’s difficult for Spark applications to share data without using an external data source such as HDFS or S3. Using an off-heap memory storage such as Tachyon (a.k.a. Alluxio) can make data sharing faster and easier. 

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-ML-Apache_Spark_architecture.png" width="800">


## Executing Spark Applications (basic)
You use an interactive shell (spark-shell or pyspark) or submit an application (spark-submit) to execute Spark applications. Some prefer to use interactive web-based notebooks such as Apache Zeppelin and Jupyter to interact with Spark. Commercial vendors such as Databricks and Cloudera provide their own interactive notebook environment as well. I will use the spark-shell throughout the demo. There are two deploy modes for launching Spark applications in an environment with a cluster manager such as YARN.

-Cluster Mode
In cluster mode, the driver program runs inside an application master managed by YARN. The client can exit without affecting the execution of the application. To launch applications or the spark-shell in cluster mode:
```
spark-shell --master yarn --deploy-mode cluster
spark-submit --class mypath.myClass --master yarn --deploy-mode cluster
```
-Client Mode
In client mode, the driver program runs in the client. The application master is only used for requesting resources from YARN. To launch applications or the spark-shell in client mode:
```
spark-shell --master yarn --deploy-mode client
spark-submit --class mypath.myClass --master yarn --deploy-mode client
```

## Introduction to the spark-shell
You typically use an interactive shell for ad hoc data analysis or exploration. It’s also a good tool to learn the Spark API. Spark’s interactive shell is available in Spark or Python. A SparkSession named “spark” is automatically created when you start spark-shell.

## Spark SQL, Dataset, and DataFrames API
Spark SQL was developed to make it easier to process and analyze structured data. Dataset is similar to an RDD in that it supports strong typing, but under the hood Dataset has a much more efficient engine. Starting in Spark 2.0, the Dataset API is now the primary programming interface. The DataFrame is just a Dataset with named columns, similar to relational table. Together, Spark SQL and DataFrames provide a powerful programming interface for processing and analyzing structured data. 

Note: The DataFrame and Dataset APIs have been unified in Spark 2.0. The DataFrame is now just a type alias for a Dataset of Row, where a Row is a
generic untyped object. In contrast, Dataset is a collection of strongly typed objects Dataset. Scala supports strongly typed and untyped API, while in Java,
Dataset is the main abstraction. DataFrames is the main programming interface for R and Python due to its lack of support for compile-time type safety.


## Spark Data Sources

-CSV, XML, JSON, Relational and MPP Databases, Parquet, Hbase, Amazon S3, Solr, Microsoft Excel, Secure FTP

Note: Analytical Massively Parallel Processing (MPP) Databases are databases that are optimized for analytical workloads: aggregating and processing large datasets (examples: Snowflake, Redshift, Impala, Presto, Azure DW, etc.)

## Spark Ecosystem of Connections

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/k8s-spark-ecosystem-of-connections.png" width="800">


## Introduction to Spark MLlib
Machine learning is one of Spark’s main applications. Spark MLlib includes popular machine learning algorithms for regression, classification, clustering, collaborative filtering, and frequent pattern mining. It also provides a wide set of features for building pipelines, model selection and tuning, and feature selection, extraction, and transformation.

### Spark MLlib Algorithms
Spark MLlib includes a plethora of machine learning algorithms for various tasks. We cover some of them in succeeding sections.
```
Classification
• Logistic Regression (Binomial and Multinomial)
• Decision Tree
• Random Forest
• Gradient-Boosted Tree
• Multilayer Perceptron
• Linear Support Vector Machine
• Naïve Bayes
• One-vs-Rest

Regression
• Linear Regression
• Decision Tree
• Random Forest
• Gradient-Boosted Tree
• Survival Regression
• Isotonic Regression

Clustering
• K-Means
• Bisecting K-Means
• Gaussian Mixture Model
• Latent Dirichlet Allocation (LDA)

Collaborative Filtering
•Alternating Least Square (ALS)

Frequent Pattern Mining
• FP-Growth
• PrefixSpan
```

--------------------------------------------------------------------------------------
# Supervised Learning:
Supervised learning is a machine learning task that makes prediction using a training dataset. Supervised learning can be categorized into either classification or regression. Regression is for predicting continuous values such as price, temperature, or distance, while classification is for predicting categories such as yes or no, spam or not spam, or malignant or benign.

## Classification
Classification is perhaps the most common supervised machine learning task. You most likely have already encountered applications that utilized classification without even realizing it. Popular use cases include medical diagnosis, targeted marketing, spam detection, credit risk prediction, and sentiment analysis, to mention a few. There are three types of classification tasks.

### Binary Classification
A task is binary or binomial classification if there are only two categories. For example, when using binary classification algorithm for spam detection, the output variable can have two categories: spam or not spam. For detecting cancer, the categories can be malignant or benign. For targeted marketing, predicting the likelihood of someone buying an item such as milk, the categories can simply be yes or no.

### Multiclass Classification
Multiclass or multinomial classification tasks have three or more categories. For example, to predict weather conditions you might have five categories: rainy, cloudy, sunny, snowy, and windy. To extend our targeted marketing example, multiclass classification can be used to predict if a customer is more likely to buy whole milk, reduced-fat milk, low-fat milk, or skim milk.

### Multilabel Classification
In multilabel classification, multiple categories can be assigned to each observation. In contrast, only one category can be assigned to an observation in multiclass classification. Using our targeted marketing example, multilabel classification is used not only to predict if a customer is more likely to buy milk, but other items as well such as cookies, butter, hotdogs, or bread.


### Spark MLlib Classification Algorithms

Spark MLlib includes several algorithms for classification. I will discuss the most popular algorithms and provide easy-to-follow code examples. Later in the demo, I will note more advanced, next-generation algorithms such as XGBoost and LightGBM.

- Logistic Regression

Logistic regression is a linear classifier that predicts probabilities. It uses a logistic (sigmoid) function to transform its output into a probability value that can be mapped to two (binary) classes. Multiclass classification is supported through multinomial logistic (softmax) regression. We will use logistic regression in one of our examples later in the demo.

- Support Vector Machine

Support vector machine is a popular algorithm that works by finding the optimal hyperplane that maximizes the margin between two classes, dividing the data points into separate classes by as wide a gap as possible. The data points closest to the classification boundary are known as support vectors.

- Naïve Bayes

Naïve Bayes is a simple multiclass linear classification algorithm based on Bayes' theorem. Naïve Bayes got its name because it naively assumes that the features in a dataset are independent, ignoring any possible correlations between features. We use naïve Bayes in our sentiment analysis.

- Multilayer Perceptron

Multilayer perceptron is a feedforward artificial network that consists of several fully connected layers of nodes. Nodes in the input layer correspond to the input dataset. Nodes in the intermediate layers utilize a logistic (sigmoid) function, while nodes in the final output layer use a softmax function to support multiclass classification. The number of nodes in the output layer must match the number of classes.

- Decision Trees

A decision tree predicts the value of an output variable by learning decision rules inferred from the input variables. Visually, a decision tree looks like a tree inverted with the root node at the top. Every internal node represents a test on an attribute. Leaf nodes represent a class label, while an individual branch represents the result of a test.

- Random Forest

Random Forest is an ensemble algorithm that uses a collection of decision trees for classification and regression. It uses a method called bagging (or bootstrap aggregation) to reduce variance while maintaining low bias. Bagging trains individual trees from subsets of the training data. In addition to bagging, Random Forest uses another method called feature bagging. In contrast to bagging (using subsets of observations), feature bagging uses a subset of features (columns). Feature bagging aims to reduce the correlation between the decision trees. Without feature bagging, the individual trees will be extremely similar especially in situations where there are only a few dominant features. For classification, a majority vote of the output, or the mode, of the individual trees becomes the final prediction of the model. For regression, the average of the output of the individual trees becomes the final output. Spark trains several trees in parallel since each tree is trained independently in Random Forest. 

- Gradient-Boosted Trees

Gradient-Boosted Tree (GBT) is another tree-based ensemble algorithm similar to Random Forest. GBTs use a technique known as boosting to create a strong learner
from weak learners (shallow trees). GBTs train an ensemble of decision trees sequentially with each succeeding tree decreasing the error of the previous tree.
This is done by using the residuals of the previous model to fit the next model. viii This residual-correction process is performed a set number of iterations with the number of iterations determined by cross-validation, until the residuals have been fully minimized.

### Third-Party Classification and Regression Algorithms
Countless open source contributors have devoted time and effort in developing third-party machine learning algorithms for Spark. Although they are not part of the core Spark MLlib library, companies such as Databricks (XGBoost) and Microsoft (LightGBM) have put their support behind these projects and are used extensively around the world. XGBoost and LightGBM are currently considered the next-generation machine learning algorithms for classification and regression. They are the go-to algorithms in situations where accuracy and speed are critical.

Examples:

----------- Spark MLlib: Multiclass Classification with Logistic Regression

Logistic regression is a linear classifier that predicts probabilities. It is popular for its ease of use and fast training speed and is frequently used for both binary classification and multiclass classification. A linear classifier such as logistic regression is suitable when your data has a clear decision boundary. In cases where the classes are not linearly separable, nonlinear classifiers such as tree-based ensembles should be considered.

Example1:

We will work on a multiclass classification problem for our first example using the popular Iris dataset. The dataset contains three classes of 50 instances
each, where each class refers to a variety of iris plant (Iris Setosa, Iris Versicolor, and Iris Virginica). As you can see from bellow figure, Iris Setosa is linearly separable from Iris Versicolor and Iris Virginica, but Iris Versicolor and Iris Virginica are not linearly separable from each other. Logistic regression should still do a decent job at classifying the dataset.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-ML-dataset.png" width="800">


Our goal is to predict the type of Iris plant given a set of features. The dataset contains four numeric features: sepal_length, sepal_width, petal_length, and petal_width (all in centimeters).

 Classification Using Logistic Regression ---> Code

```
davar@carbon:~$  kubectl exec -it spark-master-controller-4nljh -n data -- bash
root@spark-master-hostname:/# export SPARK_DIST_CLASSPATH=$(hadoop classpath)
root@spark-master-hostname:/# spark-shell
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://spark-master-hostname.spark-master-headless.data.svc.cluster.local:4040
Spark context available as 'sc' (master = spark://spark-master:7077, app id = app-20201217023528-0000).
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

scala> 

// Create a schema for our data.

scala> import org.apache.spark.sql.types._
import org.apache.spark.sql.types._

scala> var irisSchema = StructType(Array (
     |     StructField("sepal_length",   DoubleType, true),
     |     StructField("sepal_width",   DoubleType, true),
     |     StructField("petal_length",   DoubleType, true),
     |     StructField("petal_width",   DoubleType, true),
     |     StructField("class",  StringType, true)
     |     ))
irisSchema: org.apache.spark.sql.types.StructType = StructType(StructField(sepal_length,DoubleType,true), StructField(sepal_width,DoubleType,true), StructField(petal_length,DoubleType,true), StructField(petal_width,DoubleType,true), StructField(class,StringType,true))

// Read the CSV file from MinIO bucket. Use the schema that we just defined.

scala> sc.hadoopConfiguration.set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")

scala> sc.hadoopConfiguration.set("fs.s3a.access.key", "minio")

scala> sc.hadoopConfiguration.set("fs.s3a.secret.key", "minio123")

scala> sc.hadoopConfiguration.set("fs.s3a.path.style.access", "true")

scala> val dataDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")
dataDF: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 3 more fields]

// Check the schema.

scala> dataDF.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)

// Inspect the data to make sure they’re in the correct format.

scala> dataDF.show
+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
|         5.4|        3.9|         1.7|        0.4| setosa|
|         4.6|        3.4|         1.4|        0.3| setosa|
|         5.0|        3.4|         1.5|        0.2| setosa|
|         4.4|        2.9|         1.4|        0.2| setosa|
|         4.9|        3.1|         1.5|        0.1| setosa|
|         5.4|        3.7|         1.5|        0.2| setosa|
|         4.8|        3.4|         1.6|        0.2| setosa|
|         4.8|        3.0|         1.4|        0.1| setosa|
|         4.3|        3.0|         1.1|        0.1| setosa|
|         5.8|        4.0|         1.2|        0.2| setosa|
|         5.7|        4.4|         1.5|        0.4| setosa|
|         5.4|        3.9|         1.3|        0.4| setosa|
|         5.1|        3.5|         1.4|        0.3| setosa|
|         5.7|        3.8|         1.7|        0.3| setosa|
|         5.1|        3.8|         1.5|        0.3| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 20 rows

// Calculate summary statistics for our data. This can
// be helpful in understanding the distribution of your data.


scala> dataDF.describe().show(5,15)
+-------+---------------+---------------+---------------+---------------+---------+
|summary|   sepal_length|    sepal_width|   petal_length|    petal_width|  species|
+-------+---------------+---------------+---------------+---------------+---------+
|  count|            150|            150|            150|            150|      150|
|   mean|5.8433333333...|3.0540000000...|3.7586666666...|1.1986666666...|     null|
| stddev|0.8280661279...|0.4335943113...|1.7644204199...|0.7631607417...|     null|
|    min|            4.3|            2.0|            1.0|            0.1|   setosa|
|    max|            7.9|            4.4|            6.9|            2.5|virginica|
+-------+---------------+---------------+---------------+---------------+---------+

// The input column spices is currently a string. We'll use
// StringIndexer to encode it into a double. The new value
// will be stored in the new output column called label.

scala> import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexer

scala> val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
labelIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_236a70ccadd0

scala> val dataDF2 = labelIndexer.fit(dataDF).transform(dataDF)
dataDF2: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]


// Check the schema of the new DataFrame.

scala> dataDF2.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 |-- label: double (nullable = false)

// Inspect the new column added to the DataFrame.

scala> dataDF2.show
+------------+-----------+------------+-----------+-------+-----+
|sepal_length|sepal_width|petal_length|petal_width|species|label|
+------------+-----------+------------+-----------+-------+-----+
|         5.1|        3.5|         1.4|        0.2| setosa|  2.0|
|         4.9|        3.0|         1.4|        0.2| setosa|  2.0|
|         4.7|        3.2|         1.3|        0.2| setosa|  2.0|
|         4.6|        3.1|         1.5|        0.2| setosa|  2.0|
|         5.0|        3.6|         1.4|        0.2| setosa|  2.0|
|         5.4|        3.9|         1.7|        0.4| setosa|  2.0|
|         4.6|        3.4|         1.4|        0.3| setosa|  2.0|
|         5.0|        3.4|         1.5|        0.2| setosa|  2.0|
|         4.4|        2.9|         1.4|        0.2| setosa|  2.0|
|         4.9|        3.1|         1.5|        0.1| setosa|  2.0|
|         5.4|        3.7|         1.5|        0.2| setosa|  2.0|
|         4.8|        3.4|         1.6|        0.2| setosa|  2.0|
|         4.8|        3.0|         1.4|        0.1| setosa|  2.0|
|         4.3|        3.0|         1.1|        0.1| setosa|  2.0|
|         5.8|        4.0|         1.2|        0.2| setosa|  2.0|
|         5.7|        4.4|         1.5|        0.4| setosa|  2.0|
|         5.4|        3.9|         1.3|        0.4| setosa|  2.0|
|         5.1|        3.5|         1.4|        0.3| setosa|  2.0|
|         5.7|        3.8|         1.7|        0.3| setosa|  2.0|
|         5.1|        3.8|         1.5|        0.3| setosa|  2.0|
+------------+-----------+------------+-----------+-------+-----+
only showing top 20 rows

// Combine the features into a single vector
// column using the VectorAssembler transformer.

scala> import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler

scala> val features = Array("sepal_length","sepal_width","petal_length","petal_width")
features: Array[String] = Array(sepal_length, sepal_width, petal_length, petal_width)

scala> val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_33bc94c8c3fb


scala> val dataDF3 = assembler.transform(dataDF2)
dataDF3: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 5 more fields]

// Inspect the new column added to the DataFrame.

scala> dataDF3.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 |-- label: double (nullable = false)
 |-- features: vector (nullable = true)

// Inspect the new column added to the DataFrame.

scala> dataDF3.show
+------------+-----------+------------+-----------+-------+-----+-----------------+
|sepal_length|sepal_width|petal_length|petal_width|species|label|         features|
+------------+-----------+------------+-----------+-------+-----+-----------------+
|         5.1|        3.5|         1.4|        0.2| setosa|  2.0|[5.1,3.5,1.4,0.2]|
|         4.9|        3.0|         1.4|        0.2| setosa|  2.0|[4.9,3.0,1.4,0.2]|
|         4.7|        3.2|         1.3|        0.2| setosa|  2.0|[4.7,3.2,1.3,0.2]|
|         4.6|        3.1|         1.5|        0.2| setosa|  2.0|[4.6,3.1,1.5,0.2]|
|         5.0|        3.6|         1.4|        0.2| setosa|  2.0|[5.0,3.6,1.4,0.2]|
|         5.4|        3.9|         1.7|        0.4| setosa|  2.0|[5.4,3.9,1.7,0.4]|
|         4.6|        3.4|         1.4|        0.3| setosa|  2.0|[4.6,3.4,1.4,0.3]|
|         5.0|        3.4|         1.5|        0.2| setosa|  2.0|[5.0,3.4,1.5,0.2]|
|         4.4|        2.9|         1.4|        0.2| setosa|  2.0|[4.4,2.9,1.4,0.2]|
|         4.9|        3.1|         1.5|        0.1| setosa|  2.0|[4.9,3.1,1.5,0.1]|
|         5.4|        3.7|         1.5|        0.2| setosa|  2.0|[5.4,3.7,1.5,0.2]|
|         4.8|        3.4|         1.6|        0.2| setosa|  2.0|[4.8,3.4,1.6,0.2]|
|         4.8|        3.0|         1.4|        0.1| setosa|  2.0|[4.8,3.0,1.4,0.1]|
|         4.3|        3.0|         1.1|        0.1| setosa|  2.0|[4.3,3.0,1.1,0.1]|
|         5.8|        4.0|         1.2|        0.2| setosa|  2.0|[5.8,4.0,1.2,0.2]|
|         5.7|        4.4|         1.5|        0.4| setosa|  2.0|[5.7,4.4,1.5,0.4]|
|         5.4|        3.9|         1.3|        0.4| setosa|  2.0|[5.4,3.9,1.3,0.4]|
|         5.1|        3.5|         1.4|        0.3| setosa|  2.0|[5.1,3.5,1.4,0.3]|
|         5.7|        3.8|         1.7|        0.3| setosa|  2.0|[5.7,3.8,1.7,0.3]|
|         5.1|        3.8|         1.5|        0.3| setosa|  2.0|[5.1,3.8,1.5,0.3]|
+------------+-----------+------------+-----------+-------+-----+-----------------+
only showing top 20 rows

// Let's measure the statistical dependence between
// the features and the class using Pearson correlation.

scala> dataDF3.stat.corr("petal_length","label")
res11: Double = -0.6491005272522323

scala> dataDF3.stat.corr("petal_width","label")
res12: Double = -0.5807485170333053

scala> dataDF3.stat.corr("sepal_length","label")
res13: Double = -0.46003915650023736

scala> dataDF3.stat.corr("sepal_width","label")
res14: Double = 0.6121647247038234


// The petal_length and petal_width have extremely high class correlation,
// while sepal_length and sepal_width have low class correlation.
// Correlation evaluates how strong the linear relationship between two variables. 
// You can use correlation to select relevant features (feature-class correlation) 
// and identify redundant features (intra-feature correlation).

// Divide our dataset into training and test datasets.

scala> val seed = 1234
seed: Int = 1234

// We can now fit a model on the training dataset
// using logistic regression.

scala> val Array(trainingData, testData) = dataDF3.randomSplit(Array(0.8, 0.2), seed)
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [sepal_length: double, sepal_width: double ... 5 more fields]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [sepal_length: double, sepal_width: double ... 5 more fields]

scala> import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegression

scala> val lr = new LogisticRegression()
lr: org.apache.spark.ml.classification.LogisticRegression = logreg_c1cd04b61fcd

// Train a model using our training dataset.

scala> val model = lr.fit(trainingData)
2020-12-16 23:05:10 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
2020-12-16 23:05:10 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
model: org.apache.spark.ml.classification.LogisticRegressionModel = logreg_c1cd04b61fcd

// Predict on our test dataset.

scala> val predictions = model.transform(testData)
predictions: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 8 more fields]

// Note the new columns added to the DataFrame:
// rawPrediction, probability, prediction.


scala> predictions.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 |-- label: double (nullable = false)
 |-- features: vector (nullable = true)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)
 |-- prediction: double (nullable = false)


// Inspect the predictions.

scala> predictions.select("sepal_length","sepal_width","petal_length","petal_width","label","prediction").show
+------------+-----------+------------+-----------+-----+----------+
|sepal_length|sepal_width|petal_length|petal_width|label|prediction|
+------------+-----------+------------+-----------+-----+----------+
|         4.3|        3.0|         1.1|        0.1|  2.0|       2.0|
|         4.4|        2.9|         1.4|        0.2|  2.0|       2.0|
|         4.4|        3.0|         1.3|        0.2|  2.0|       2.0|
|         4.8|        3.1|         1.6|        0.2|  2.0|       2.0|
|         5.0|        3.3|         1.4|        0.2|  2.0|       2.0|
|         5.0|        3.4|         1.5|        0.2|  2.0|       2.0|
|         5.0|        3.6|         1.4|        0.2|  2.0|       2.0|
|         5.1|        3.4|         1.5|        0.2|  2.0|       2.0|
|         5.2|        2.7|         3.9|        1.4|  0.0|       0.0|
|         5.2|        4.1|         1.5|        0.1|  2.0|       2.0|
|         5.3|        3.7|         1.5|        0.2|  2.0|       2.0|
|         5.6|        2.9|         3.6|        1.3|  0.0|       0.0|
|         5.8|        2.8|         5.1|        2.4|  1.0|       1.0|
|         6.0|        2.2|         4.0|        1.0|  0.0|       0.0|
|         6.0|        2.9|         4.5|        1.5|  0.0|       0.0|
|         6.0|        3.4|         4.5|        1.6|  0.0|       0.0|
|         6.2|        2.8|         4.8|        1.8|  1.0|       1.0|
|         6.2|        2.9|         4.3|        1.3|  0.0|       0.0|
|         6.3|        2.8|         5.1|        1.5|  1.0|       0.0|
|         6.7|        3.1|         5.6|        2.4|  1.0|       1.0|
+------------+-----------+------------+-----------+-----+----------+
only showing top 20 rows

// Inspect the rawPrediction and probability columns.


scala> predictions.select("rawPrediction","probability","prediction").show(false)
+------------------------------------------------------------+---------------------------------+----------+
|rawPrediction                                               |probability                      |prediction|
+------------------------------------------------------------+---------------------------------+----------+
|[93.23317326685446,-17698.196916992016,17604.963743725166]  |[0.0,0.0,1.0]                    |2.0       |
|[1288.6293224909996,-15099.078099714581,13810.448777223583] |[0.0,0.0,1.0]                    |2.0       |
|[644.6618555051145,-16138.962302020398,15494.300446515286]  |[0.0,0.0,1.0]                    |2.0       |
|[1200.6450429890074,-15273.923054843319,14073.278011854314] |[0.0,0.0,1.0]                    |2.0       |
|[326.84100433275216,-17035.43728626484,16708.59628193209]   |[0.0,0.0,1.0]                    |2.0       |
|[-131.28203073375062,-17441.113937019323,17572.395967753073]|[0.0,0.0,1.0]                    |2.0       |
|[-1326.2947487458268,-19203.778565855297,20530.07331460112] |[0.0,0.0,1.0]                    |2.0       |
|[75.78341692400227,-17281.986850424266,17206.20343350026]   |[0.0,0.0,1.0]                    |2.0       |
|[8272.524436731177,4737.893562400536,-13010.417999131716]   |[1.0,0.0,0.0]                    |0.0       |
|[-3732.9866952627153,-22948.222726365493,26681.209421628202]|[0.0,0.0,1.0]                    |2.0       |
|[-1163.2214408390673,-19132.073956824606,20295.295397663678]|[0.0,0.0,1.0]                    |2.0       |
|[7561.410274769612,2211.6297515682463,-9773.040026337858]   |[1.0,0.0,0.0]                    |0.0       |
|[11664.126489779048,16434.120733000975,-28098.24722278002]  |[0.0,1.0,0.0]                    |1.0       |
|[12143.121278439361,6878.316256286532,-19021.43753472589]   |[1.0,0.0,0.0]                    |0.0       |
|[9543.009614360093,7233.872033579927,-16776.88164794002]    |[1.0,0.0,0.0]                    |0.0       |
|[6946.302161890255,4385.869877754393,-11332.17203964465]    |[1.0,0.0,0.0]                    |0.0       |
|[11262.508816564241,11523.917891102865,-22786.426707667106] |[2.9612931937190816E-114,1.0,0.0]|1.0       |
|[9454.258472433969,5386.1187015682635,-14840.377174002231]  |[1.0,0.0,0.0]                    |0.0       |
|[11272.784504117686,10336.656374549253,-21609.440878666937] |[1.0,0.0,0.0]                    |0.0       |
|[12339.190845418703,17283.442111644374,-29622.63295706308]  |[0.0,1.0,0.0]                    |1.0       |
+------------------------------------------------------------+---------------------------------+----------+
only showing top 20 rows


// Evaluate the model. Several evaluation metrics are available
// for multiclass classification: f1 (default), accuracy,
// weightedPrecision, and weightedRecall.



scala> import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

scala> val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_154a8f76ac68

scala> val f1 = evaluator.evaluate(predictions)
f1: Double = 0.958119658119658

scala> val wp = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
wp: Double = 0.9635416666666667

scala> val wr = evaluator.setMetricName("weightedRecall").evaluate(predictions)
wr: Double = 0.9583333333333334

scala> val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
accuracy: Double = 0.9583333333333334
```

Logistic regression is a popular classification algorithm often used as a first baseline algorithm due to its speed and simplicity. For production use, more advanced tree-based ensembles is generally preferred due to their superior accuracy and ability to capture complex nonlinear relationships in datasets.

Other examples:

----spark MLlib: Churn Prediction with Random Forest --> www.kaggle.com/becksddf/churn-in-telecoms-dataset 

----eXtreme Gradient Boosting with XGBoost4J-Spark 

----LightGBM: Fast Gradient Boosting from Microsoft

----Sentiment Analysis with Naïve Bayes 


## Regression

Regression is a supervised machine learning task for predicting continuous numeric values. Popular use cases include sales and demand forecasting, predicting stock, home or commodity prices, and weather forecasting, to mention a few.

### Simple Linear Regression
Linear regression is used for examining linear relationships between one or more independent variable(s) and a dependent variable. The analysis of the relationship between a single independent variable and a single continuous dependent variable is known as simple linear regression.


Example2.

For our example, we will use simple linear regression to show how home prices (the dependent variable) change based on the area’s average household income (the
independent variable). Code:
```
root@spark-master-hostname:/# export SPARK_DIST_CLASSPATH=$(hadoop classpath)
root@spark-master-hostname:/# spark-shell
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://spark-master-hostname.spark-master-headless.data.svc.cluster.local:4040
Spark context available as 'sc' (master = spark://spark-master:7077, app id = app-20201217023528-0000).
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

scala> import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegression

scala> import spark.implicits._
import spark.implicits._

scala> val dataDF = Seq(
     | (50000, 302200),
     | (75200, 550000),
     | (90000, 680000),
     | (32800, 225000),
     | (41000, 275000),
     | (54000, 300500),
     | (72000, 525000),
     | (105000, 700000),
     | (88500, 673100),
     | (92000, 695000),
     | (53000, 320900),
     | (85200, 652800),
     | (157000, 890000),
     | (128000, 735000),
     | (71500, 523000),
     | (114000, 720300),
     | (33400, 265900),
     | (143000, 846000),
     | (68700, 492000),
     | (46100, 285000)
     | ).toDF("avg_area_income","price")
dataDF: org.apache.spark.sql.DataFrame = [avg_area_income: int, price: int]

scala> dataDF.show
+---------------+------+
|avg_area_income| price|
+---------------+------+
|          50000|302200|
|          75200|550000|
|          90000|680000|
|          32800|225000|
|          41000|275000|
|          54000|300500|
|          72000|525000|
|         105000|700000|
|          88500|673100|
|          92000|695000|
|          53000|320900|
|          85200|652800|
|         157000|890000|
|         128000|735000|
|          71500|523000|
|         114000|720300|
|          33400|265900|
|         143000|846000|
|          68700|492000|
|          46100|285000|
+---------------+------+


scala> import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler

scala> val assembler = new VectorAssembler().setInputCols(Array("avg_area_income")).setOutputCol("feature")
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_178de1651880

scala> val dataDF2 = assembler.transform(dataDF)
dataDF2: org.apache.spark.sql.DataFrame = [avg_area_income: int, price: int ... 1 more field]

scala> dataDF2.show
+---------------+------+----------+
|avg_area_income| price|   feature|
+---------------+------+----------+
|          50000|302200| [50000.0]|
|          75200|550000| [75200.0]|
|          90000|680000| [90000.0]|
|          32800|225000| [32800.0]|
|          41000|275000| [41000.0]|
|          54000|300500| [54000.0]|
|          72000|525000| [72000.0]|
|         105000|700000|[105000.0]|
|          88500|673100| [88500.0]|
|          92000|695000| [92000.0]|
|          53000|320900| [53000.0]|
|          85200|652800| [85200.0]|
|         157000|890000|[157000.0]|
|         128000|735000|[128000.0]|
|          71500|523000| [71500.0]|
|         114000|720300|[114000.0]|
|          33400|265900| [33400.0]|
|         143000|846000|[143000.0]|
|          68700|492000| [68700.0]|
|          46100|285000| [46100.0]|
+---------------+------+----------+


scala> val lr = new LinearRegression().setMaxIter(10).setFeaturesCol("feature").setLabelCol("price")
lr: org.apache.spark.ml.regression.LinearRegression = linReg_8a217afa45b1

scala> val model = lr.fit(dataDF2)
2020-12-17 02:39:42 WARN  WeightedLeastSquares:66 - regParam is zero, which might cause numerical instability and overfitting.
[Stage 0:============================================>              (6 + 2) / 8]2020-12-17 02:39:48 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
2020-12-17 02:39:48 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
2020-12-17 02:39:48 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
2020-12-17 02:39:48 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
model: org.apache.spark.ml.regression.LinearRegressionModel = linReg_8a217afa45b1

scala> import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vectors

scala> val testData = spark.createDataFrame(Seq(Vectors.dense(75000)).map(Tuple1.apply)).toDF("feature")
testData: org.apache.spark.sql.DataFrame = [feature: vector]

scala> val predictions = model.transform(testData)
predictions: org.apache.spark.sql.DataFrame = [feature: vector, prediction: double]

scala> predictions.show
+---------+------------------+
|  feature|        prediction|
+---------+------------------+
|[75000.0]|504090.35842779215|
+---------+------------------+


```
### Multiple Regression with XGBoost4J-Spark
Multiple regression is used in more realistic scenarios where there are two or more independent variables and a single continuous dependent variable. It is common to have both linear and nonlinear features in real-world use cases. Tree-based ensemble algorithms like XGBoost have the ability to handle both linear and nonlinear features which makes it ideal for most production environments. In most situations using tree-based ensembles such as XGBoost for multiple regression should lead to significantly better prediction accuracy. Because XGBoost supports both classification and regression, using XGBoost for regression is very similar to classification. (spark-shell --packages ml.dmlc:xgboost4j-spark:0.81)


### Multiple Regression with LightGBM
LightGBM comes with the LightGBMRegressor class specifically for regression tasks. 

# Unsupervised Learning

Unsupervised learning is a machine learning task that finds hidden patterns and structure in the dataset without the aid of labeled responses. Unsupervised learning is ideal when you only have access to input data and training data is unavailable or hard to obtain. Common methods include clustering, topic modeling, anomaly detection, and principal component analysis.

## Clustering with K-Means
Clustering is an unsupervised machine learning task for grouping unlabeled observations that have some similarities. Popular clustering use cases include customer segmentation, fraud analysis, and anomaly detection. Clustering is also often used to generate training data for classifiers in cases where training data is scarce or unavailable. K-Means is one of the most popular unsupervised learning algorithms for clustering. Spark MLlib includes a more scalable implementation of K-means known as K-means. In the example K-means grouping the observations in the Iris dataset into two distinct clusters. K-Means requires the users to provide the number of clusters k to the algorithm. There are ways to find the optimal number of clusters for your dataset. We will discuss
the elbow and silhouette method later.

Example3: A Customer Segmentation Example Using K-Means
```

// Let's start with our example by creating some sample data.

scala> val custDF = Seq(
     | (100, 29000,"M","F","CA",25),
     | (101, 36000,"M","M","CA",46),
     | (102, 5000,"S","F","NY",18),
     | (103, 68000,"S","M","AZ",39),
     | (104, 2000,"S","F","CA",16),
     | (105, 75000,"S","F","CA",41),
     | (106, 90000,"M","M","MA",47),
     | (107, 87000,"S","M","NY",38)
     | ).toDF("customerid", "income","maritalstatus","gender","state","age")
custDF: org.apache.spark.sql.DataFrame = [customerid: int, income: int ... 4 more fields]

// Perform some preprocessing steps.

scala> import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexer


scala> val genderIndexer = new StringIndexer().setInputCol("gender").setOutputCol("gender_idx")
genderIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_b36b7df06362

scala> val stateIndexer = new StringIndexer().setInputCol("state").setOutputCol("state_idx")
stateIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_c171b935f5f3

scala> val mstatusIndexer = new StringIndexer().setInputCol("maritalstatus").setOutputCol("maritalstatus_idx")
mstatusIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_7fed494489e5

scala> import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.OneHotEncoderEstimator                     
                     
scala> val encoder = new OneHotEncoderEstimator().setInputCols(Array("gender_idx","state_idx","maritalstatus_idx")).setOutputCols(Array("gender_enc","state_enc","maritalstatus_enc"))
encoder: org.apache.spark.ml.feature.OneHotEncoderEstimator = oneHotEncoder_b09f3a2f825b

              
scala> val custDF2 = genderIndexer.fit(custDF).transform(custDF)
custDF2: org.apache.spark.sql.DataFrame = [customerid: int, income: int ... 5 more fields]

scala> val custDF3 = stateIndexer.fit(custDF2).transform(custDF2)
custDF3: org.apache.spark.sql.DataFrame = [customerid: int, income: int ... 6 more fields]

scala> val custDF4 = mstatusIndexer.fit(custDF3).transform(custDF3)
custDF4: org.apache.spark.sql.DataFrame = [customerid: int, income: int ... 7 more fields]

scala> custDF4.select("gender_idx","state_idx","maritalstatus_idx").show
+----------+---------+-----------------+
|gender_idx|state_idx|maritalstatus_idx|
+----------+---------+-----------------+
|       1.0|      0.0|              1.0|
|       0.0|      0.0|              1.0|
|       1.0|      1.0|              0.0|
|       0.0|      2.0|              0.0|
|       1.0|      0.0|              0.0|
|       1.0|      0.0|              0.0|
|       0.0|      3.0|              1.0|
|       0.0|      1.0|              0.0|
+----------+---------+-----------------+


scala> val custDF5 = encoder.fit(custDF4).transform(custDF4)
custDF5: org.apache.spark.sql.DataFrame = [customerid: int, income: int ... 10 more fields]

scala> custDF5.printSchema
root
 |-- customerid: integer (nullable = false)
 |-- income: integer (nullable = false)
 |-- maritalstatus: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- state: string (nullable = true)
 |-- age: integer (nullable = false)
 |-- gender_idx: double (nullable = false)
 |-- state_idx: double (nullable = false)
 |-- maritalstatus_idx: double (nullable = false)
 |-- gender_enc: vector (nullable = true)
 |-- state_enc: vector (nullable = true)
 |-- maritalstatus_enc: vector (nullable = true)

scala> custDF5.select("gender_enc","state_enc","maritalstatus_enc").show
+-------------+-------------+-----------------+
|   gender_enc|    state_enc|maritalstatus_enc|
+-------------+-------------+-----------------+
|    (1,[],[])|(3,[0],[1.0])|        (1,[],[])|
|(1,[0],[1.0])|(3,[0],[1.0])|        (1,[],[])|
|    (1,[],[])|(3,[1],[1.0])|    (1,[0],[1.0])|
|(1,[0],[1.0])|(3,[2],[1.0])|    (1,[0],[1.0])|
|    (1,[],[])|(3,[0],[1.0])|    (1,[0],[1.0])|
|    (1,[],[])|(3,[0],[1.0])|    (1,[0],[1.0])|
|(1,[0],[1.0])|    (3,[],[])|        (1,[],[])|
|(1,[0],[1.0])|(3,[1],[1.0])|    (1,[0],[1.0])|
+-------------+-------------+-----------------+


scala> import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler

scala> val assembler = new VectorAssembler().setInputCols(Array("income","gender_enc", "state_enc","maritalstatus_enc", "age")).setOutputCol("features")
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_1f56fd5ee258

scala> val custDF6 = assembler.transform(custDF5)
custDF6: org.apache.spark.sql.DataFrame = [customerid: int, income: int ... 11 more fields]

scala> custDF6.printSchema
root
 |-- customerid: integer (nullable = false)
 |-- income: integer (nullable = false)
 |-- maritalstatus: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- state: string (nullable = true)
 |-- age: integer (nullable = false)
 |-- gender_idx: double (nullable = false)
 |-- state_idx: double (nullable = false)
 |-- maritalstatus_idx: double (nullable = false)
 |-- gender_enc: vector (nullable = true)
 |-- state_enc: vector (nullable = true)
 |-- maritalstatus_enc: vector (nullable = true)
 |-- features: vector (nullable = true)

scala> custDF6.select("features").show(false)
+----------------------------------+
|features                          |
+----------------------------------+
|(7,[0,2,6],[29000.0,1.0,25.0])    |
|[36000.0,1.0,1.0,0.0,0.0,0.0,46.0]|
|[5000.0,0.0,0.0,1.0,0.0,1.0,18.0] |
|[68000.0,1.0,0.0,0.0,1.0,1.0,39.0]|
|[2000.0,0.0,1.0,0.0,0.0,1.0,16.0] |
|[75000.0,0.0,1.0,0.0,0.0,1.0,41.0]|
|(7,[0,1,6],[90000.0,1.0,47.0])    |
|[87000.0,1.0,0.0,1.0,0.0,1.0,38.0]|
+----------------------------------+

                
scala> import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StandardScaler

scala> val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
scaler: org.apache.spark.ml.feature.StandardScaler = stdScal_8271e368caf7

scala> val custDF7 = scaler.fit(custDF6).transform(custDF6)
custDF7: org.apache.spark.sql.DataFrame = [customerid: int, income: int ... 12 more fields]

scala> custDF7.printSchema
root
 |-- customerid: integer (nullable = false)
 |-- income: integer (nullable = false)
 |-- maritalstatus: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- state: string (nullable = true)
 |-- age: integer (nullable = false)
 |-- gender_idx: double (nullable = false)
 |-- state_idx: double (nullable = false)
 |-- maritalstatus_idx: double (nullable = false)
 |-- gender_enc: vector (nullable = true)
 |-- state_enc: vector (nullable = true)
 |-- maritalstatus_enc: vector (nullable = true)
 |-- features: vector (nullable = true)
 |-- scaledFeatures: vector (nullable = true)

scala> custDF7.select("scaledFeatures").show(8,65)
+-----------------------------------------------------------------+
|                                                   scaledFeatures|
+-----------------------------------------------------------------+
|(7,[0,2,6],[0.8144011366375091,1.8708286933869707,2.0282443813...|
|[1.0109807213431148,1.8708286933869707,1.8708286933869707,0.0,...|
|[0.1404139890754326,0.0,0.0,2.160246899469287,0.0,1.9321835661...|
|[1.9096302514258834,1.8708286933869707,0.0,0.0,2.8284271247461...|
|[0.05616559563017304,0.0,1.8708286933869707,0.0,0.0,1.93218356...|
|[2.106209836131489,0.0,1.8708286933869707,0.0,0.0,1.9321835661...|
|(7,[0,1,6],[2.5274518033577866,1.8708286933869707,3.8130994368...|
|[2.443203409912527,1.8708286933869707,0.0,2.160246899469287,0....|
+-----------------------------------------------------------------+

// We’ll create two clusters.

scala> import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeans

scala> val kmeans = new KMeans().setFeaturesCol("scaledFeatures").setPredictionCol("prediction").setK(2)
kmeans: org.apache.spark.ml.clustering.KMeans = kmeans_af695d4329ba

             
scala> import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline

scala> val pipeline = new Pipeline().setStages(Array(genderIndexer, stateIndexer,mstatusIndexer, encoder, assembler, scaler, kmeans))
pipeline: org.apache.spark.ml.Pipeline = pipeline_23ace5be30ad

scala> val model = pipeline.fit(custDF)

scala> val clusters = model.transform(custDF)
               
                           
scala> clusters.select("customerid","income","maritalstatus","gender","state","age","prediction").show
                
                
+----------+------+-------------+------+-----+---+----------+
|customerid|income|maritalstatus|gender|state|age|prediction|
+----------+------+-------------+------+-----+---+----------+
|       100| 29000|            M|     F|   CA| 25|        1|
|       101| 36000|            M|     M|   CA| 46|        0|
|       102|  5000|            S|     F|   NY| 18|         1|
|       103| 68000|            S|     M|   AZ| 39|        0|
|       104|  2000|            S|     F|   CA| 16|         1|
|       105| 75000|            S|     F|   CA| 41|        0|
|       106| 90000|            M|     M|   MA| 47|        0|
|       107| 87000|            S|     M|   NY| 38|        0|
+----------+------+-------------+------+-----+---+----------+

scala> import org.apache.spark.ml.clustering.KMeansModel

scala> val model = pipeline.stages.last.asInstanceOf[KMeansModel]
scala> model.clusterCenters.foreach(println)

[1.9994952044341603,0.37416573867739417,0.7483314773547883,0.4320493798938574,
0.565685424949238,1.159310139695155,3.4236765156588613]
[0.3369935737810382,1.8708286933869707,1.247219128924647,0.7200822998230956,
0.0,1.288122377439061,1.5955522466340666]                         
             
             
We evaluate our cluster by computing Within Set Sum of Squared Errors (WSSSE).
Examining the WSSSE using the “elbow method” is often used to assist in determining
the optimal number of clusters. The elbow method works by fitting the model with a
range of values for k and plotting it against the WSSSE. Visually examine the line chart,
and if it resembles a flexed arm, the point where it bends on the curve (the “elbow”)
indicates the most optimal value for k.

scala> val wssse = model.computeCost(custDF)
scala> wssse: Double = 32.09801038868844            
                
 
Another way to evaluate cluster quality is by computing the silhouette coefficient
score. The silhouette score provides a metric of how close each point in one cluster is
to points in the other clusters. The larger the silhouette score, the better the quality of
the cluster. A score closer to 1 indicates that the points are closer to the centroid of the
cluster. A score closer to 0 indicates that the points are closer to other clusters, and a
negative value indicates the points may have been designated to the wrong cluster.

scala> import org.apache.spark.ml.evaluation.ClusteringEvaluator
scala> val evaluator = new ClusteringEvaluator()
scala> val silhouette = evaluator.evaluate(clusters)
silhouette: Double = 0.6722088068201866
```

## Topic Modeling with Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) was developed in 2003 by David M. Blei, Andrew Ng, and Michael Jordan, although a similar algorithm used in population genetics was also proposed by Jonathan K. Pritchard, Matthew Stephens, and Peter Donnelly in 2000. LDA, as applied to machine learning, is based on a graphical model and is the first algorithm included in Spark MLlib built on GraphX. Latent Dirichlet Allocation is widely used for topic modeling. Topic models automatically derive the themes (or topics) in a group of documents. These topics can be used for content-based recommendations, document classification, dimensionality reduction, and featurization.

## Anomaly Detection with Isolation Forest
Anomaly or outlier detection identifies rare observations that deviate significantly and stand out from majority of the dataset. It is frequently used in discovering fraudulent financial transactions, identifying cybersecurity threats, or performing predictive maintenance, to mention a few use cases. Anomaly detection is a popular research area in the field of machine learning. Several anomaly detection techniques have been invented throughout the years with varying degrees of effectiveness. For this demo, we will cover one of the most effective anomaly detection techniques called Isolation Forest. Isolation Forest is a tree-based ensemble algorithm for anomaly detection that was developed by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou.


##  Dimensionality Reduction with Principal Component Analysis 
Principal component analysis (PCA) is an unsupervised machine learning technique used for reducing the dimensionality of the feature space. It detects correlations between features and generates a reduced number of linearly uncorrelated features while retaining most of the variance in the original dataset. These more compact, linearly uncorrelated features are called principal components. The principal components are sorted in descending order of their explained variance. Dimensionality reduction is essential when there are a high number of features in your dataset. Machine learning use cases in the fields of genomics and industrial analytics, for instance, usually involve thousands or even millions of features. High dimensionality makes models more complex, increasing the chances of overfitting. Adding more features at a certain point will actually decrease the performance of the model. Moreover, training on high-dimensional data requires significant computing resources. These are collectively known as the curse of dimensionality. Dimensionality reduction techniques aim to overcome the
curse of dimensionality. Note that the principal components generated by PCA will not be interpretable. This is a deal-breaker in situations where you need to understand why the prediction was made. Furthermore, it is essential to standardize your dataset before applying PCA to prevent features that are on the largest scale to be considered more important than other features.

Principal component analysis (PCA) is a dimensionality reduction technique that combines correlated features into a smaller set of linearly uncorrelated features known as principal components. PCA has applications in multiple fields such as image recognition and anomaly detection.



Example3:
For our example, we will use PCA on the Iris dataset to project four-dimensional feature vectors into two-dimensional principal components.

```


root@spark-master-hostname:/# spark-shell
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://spark-master-hostname.spark-master-headless.data.svc.cluster.local:4040
Spark context available as 'sc' (master = spark://spark-master:7077, app id = app-20201216215914-0002).
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


scala> import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.ml.feature.{PCA, VectorAssembler}

scala> import org.apache.spark.sql.types._
import org.apache.spark.sql.types._

scala> var irisSchema = StructType(Array (
     |     StructField("sepal_length",   DoubleType, true),
     |     StructField("sepal_width",   DoubleType, true),
     |     StructField("petal_length",   DoubleType, true),
     |     StructField("petal_width",   DoubleType, true),
     |     StructField("class",  StringType, true)
     |     ))
irisSchema: org.apache.spark.sql.types.StructType = StructType(StructField(sepal_length,DoubleType,true), StructField(sepal_width,DoubleType,true), StructField(petal_length,DoubleType,true), StructField(petal_width,DoubleType,true), StructField(class,StringType,true))

scala> sc.hadoopConfiguration.set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")

scala> sc.hadoopConfiguration.set("fs.s3a.access.key", "minio")

scala> sc.hadoopConfiguration.set("fs.s3a.secret.key", "minio123")

scala> sc.hadoopConfiguration.set("fs.s3a.path.style.access", "true")

scala> val dataDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")
dataDF: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 3 more fields]

scala> dataDF.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)


scala> dataDF.show
+------------+-----------+------------+-----------+-------+                     
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
|         5.4|        3.9|         1.7|        0.4| setosa|
|         4.6|        3.4|         1.4|        0.3| setosa|
|         5.0|        3.4|         1.5|        0.2| setosa|
|         4.4|        2.9|         1.4|        0.2| setosa|
|         4.9|        3.1|         1.5|        0.1| setosa|
|         5.4|        3.7|         1.5|        0.2| setosa|
|         4.8|        3.4|         1.6|        0.2| setosa|
|         4.8|        3.0|         1.4|        0.1| setosa|
|         4.3|        3.0|         1.1|        0.1| setosa|
|         5.8|        4.0|         1.2|        0.2| setosa|
|         5.7|        4.4|         1.5|        0.4| setosa|
|         5.4|        3.9|         1.3|        0.4| setosa|
|         5.1|        3.5|         1.4|        0.3| setosa|
|         5.7|        3.8|         1.7|        0.3| setosa|
|         5.1|        3.8|         1.5|        0.3| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 20 rows


scala> dataDF.describe().show(5,15)
+-------+---------------+---------------+---------------+---------------+---------+
|summary|   sepal_length|    sepal_width|   petal_length|    petal_width|  species|
+-------+---------------+---------------+---------------+---------------+---------+
|  count|            150|            150|            150|            150|      150|
|   mean|5.8433333333...|3.0540000000...|3.7586666666...|1.1986666666...|     null|
| stddev|0.8280661279...|0.4335943113...|1.7644204199...|0.7631607417...|     null|
|    min|            4.3|            2.0|            1.0|            0.1|   setosa|
|    max|            7.9|            4.4|            6.9|            2.5|virginica|
+-------+---------------+---------------+---------------+---------------+---------+

scala> import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexer


scala> val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
labelIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_ab98e216b593

scala> val dataDF2 = labelIndexer.fit(dataDF).transform(dataDF)
dataDF2: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]


scala> dataDF2.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 |-- label: double (nullable = false)

scala> dataDF2.show
+------------+-----------+------------+-----------+-------+-----+
|sepal_length|sepal_width|petal_length|petal_width|species|label|
+------------+-----------+------------+-----------+-------+-----+
|         5.1|        3.5|         1.4|        0.2| setosa|  2.0|
|         4.9|        3.0|         1.4|        0.2| setosa|  2.0|
|         4.7|        3.2|         1.3|        0.2| setosa|  2.0|
|         4.6|        3.1|         1.5|        0.2| setosa|  2.0|
|         5.0|        3.6|         1.4|        0.2| setosa|  2.0|
|         5.4|        3.9|         1.7|        0.4| setosa|  2.0|
|         4.6|        3.4|         1.4|        0.3| setosa|  2.0|
|         5.0|        3.4|         1.5|        0.2| setosa|  2.0|
|         4.4|        2.9|         1.4|        0.2| setosa|  2.0|
|         4.9|        3.1|         1.5|        0.1| setosa|  2.0|
|         5.4|        3.7|         1.5|        0.2| setosa|  2.0|
|         4.8|        3.4|         1.6|        0.2| setosa|  2.0|
|         4.8|        3.0|         1.4|        0.1| setosa|  2.0|
|         4.3|        3.0|         1.1|        0.1| setosa|  2.0|
|         5.8|        4.0|         1.2|        0.2| setosa|  2.0|
|         5.7|        4.4|         1.5|        0.4| setosa|  2.0|
|         5.4|        3.9|         1.3|        0.4| setosa|  2.0|
|         5.1|        3.5|         1.4|        0.3| setosa|  2.0|
|         5.7|        3.8|         1.7|        0.3| setosa|  2.0|
|         5.1|        3.8|         1.5|        0.3| setosa|  2.0|
+------------+-----------+------------+-----------+-------+-----+
only showing top 20 rows

scala> import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler


scala> val features = Array("sepal_length","sepal_width","petal_length","petal_width")
features: Array[String] = Array(sepal_length, sepal_width, petal_length, petal_width)

scala> val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_59743e233092

scala> val dataDF3 = assembler.transform(dataDF2)
dataDF3: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 5 more fields]

scala> dataDF3.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 |-- label: double (nullable = false)
 |-- features: vector (nullable = true)
               
scala> dataDF3.show
+------------+-----------+------------+-----------+-------+-----+-----------------+
|sepal_length|sepal_width|petal_length|petal_width|species|label|         features|
+------------+-----------+------------+-----------+-------+-----+-----------------+
|         5.1|        3.5|         1.4|        0.2| setosa|  2.0|[5.1,3.5,1.4,0.2]|
|         4.9|        3.0|         1.4|        0.2| setosa|  2.0|[4.9,3.0,1.4,0.2]|
|         4.7|        3.2|         1.3|        0.2| setosa|  2.0|[4.7,3.2,1.3,0.2]|
|         4.6|        3.1|         1.5|        0.2| setosa|  2.0|[4.6,3.1,1.5,0.2]|
|         5.0|        3.6|         1.4|        0.2| setosa|  2.0|[5.0,3.6,1.4,0.2]|
|         5.4|        3.9|         1.7|        0.4| setosa|  2.0|[5.4,3.9,1.7,0.4]|
|         4.6|        3.4|         1.4|        0.3| setosa|  2.0|[4.6,3.4,1.4,0.3]|
|         5.0|        3.4|         1.5|        0.2| setosa|  2.0|[5.0,3.4,1.5,0.2]|
|         4.4|        2.9|         1.4|        0.2| setosa|  2.0|[4.4,2.9,1.4,0.2]|
|         4.9|        3.1|         1.5|        0.1| setosa|  2.0|[4.9,3.1,1.5,0.1]|
|         5.4|        3.7|         1.5|        0.2| setosa|  2.0|[5.4,3.7,1.5,0.2]|
|         4.8|        3.4|         1.6|        0.2| setosa|  2.0|[4.8,3.4,1.6,0.2]|
|         4.8|        3.0|         1.4|        0.1| setosa|  2.0|[4.8,3.0,1.4,0.1]|
|         4.3|        3.0|         1.1|        0.1| setosa|  2.0|[4.3,3.0,1.1,0.1]|
|         5.8|        4.0|         1.2|        0.2| setosa|  2.0|[5.8,4.0,1.2,0.2]|
|         5.7|        4.4|         1.5|        0.4| setosa|  2.0|[5.7,4.4,1.5,0.4]|
|         5.4|        3.9|         1.3|        0.4| setosa|  2.0|[5.4,3.9,1.3,0.4]|
|         5.1|        3.5|         1.4|        0.3| setosa|  2.0|[5.1,3.5,1.4,0.3]|
|         5.7|        3.8|         1.7|        0.3| setosa|  2.0|[5.7,3.8,1.7,0.3]|
|         5.1|        3.8|         1.5|        0.3| setosa|  2.0|[5.1,3.8,1.5,0.3]|
+------------+-----------+------------+-----------+-------+-----+-----------------+
only showing top 20 rows
 
// We will standardize the four attributes (sepal_length, sepal_width,
// petal_length, and petal_width) using StandardScaler even though they all
// have the same scale and measure the same quantity. As discussed earlier,
// standardization is considered the best practice and is a requirement for
// many algorithms such as PCA to execute optimally.               

scala> import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.StandardScaler

scala> val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
scaler: org.apache.spark.ml.feature.StandardScaler = stdScal_1b9e24b76c0a


scala> val dataDF4 = scaler.fit(dataDF3).transform(dataDF3)
dataDF4: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 6 more fields]

scala> dataDF4.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 |-- label: double (nullable = false)
 |-- features: vector (nullable = true)
 |-- scaledFeatures: vector (nullable = true)

// Generate two principal components.

scala> val pca = new PCA().setInputCol("scaledFeatures").setOutputCol("pcaFeatures").setK(2).fit(dataDF4)
2020-12-16 22:24:39 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
2020-12-16 22:24:39 WARN  LAPACK:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
pca: org.apache.spark.ml.feature.PCAModel = pca_cfddbb7708bb

scala> val dataDF5 = pca.transform(dataDF4)
dataDF5: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 7 more fields]

scala> dataDF5.printSchema
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 |-- label: double (nullable = false)
 |-- features: vector (nullable = true)
 |-- scaledFeatures: vector (nullable = true)
 |-- pcaFeatures: vector (nullable = true)

scala> dataDF5.select("scaledFeatures","pcaFeatures").show(false)
+---------------------------------------------------------------------------+-----------------------------------------+
|scaledFeatures                                                             |pcaFeatures                              |
+---------------------------------------------------------------------------+-----------------------------------------+
|[6.158928408838787,8.072061621390857,0.7934616853039358,0.26206798787142]  |[-1.7008636408214346,-9.798112476165109] |
|[5.9174018045706,6.9189099611921625,0.7934616853039358,0.26206798787142]   |[-1.8783851549940478,-8.640880678324866] |
|[5.675875200302412,7.38017062527164,0.7367858506393691,0.26206798787142]   |[-1.597800192305247,-8.976683127367169]  |
|[5.555111898168318,7.149540293231902,0.8501375199685027,0.26206798787142]  |[-1.6613406138855684,-8.720650458966217] |
|[6.038165106704694,8.302691953430596,0.7934616853039358,0.26206798787142]  |[-1.5770426874367196,-9.96661148272853]  |
|[6.52121831524107,8.99458294954981,0.9634891892976364,0.52413597574284]    |[-1.8942207975522354,-10.80757533867312] |
|[5.555111898168318,7.841431289351117,0.7934616853039358,0.39310198180713]  |[-1.5202989381570455,-9.368410789070643] |
|[6.038165106704694,7.841431289351117,0.8501375199685027,0.26206798787142]  |[-1.7314610064823877,-9.540884243679617] |
|[5.313585293900131,6.688279629152423,0.7934616853039358,0.26206798787142]  |[-1.6237061774493644,-8.202607301741613] |
|[5.9174018045706,7.149540293231902,0.8501375199685027,0.13103399393571]    |[-1.7764763044699745,-8.846965954487347] |
|[6.52121831524107,8.533322285470334,0.8501375199685027,0.26206798787142]   |[-1.8015813990792064,-10.361118028393015]|
|[5.7966385024365055,7.841431289351117,0.9068133546330697,0.26206798787142] |[-1.6382374187586244,-9.452155017757546] |
|[5.7966385024365055,6.9189099611921625,0.7934616853039358,0.13103399393571]|[-1.741187558292187,-8.587346593832775]  |
|[5.192821991766037,6.9189099611921625,0.6234341813102354,0.13103399393571] |[-1.3269417814262463,-8.358947926562632] |
|[7.004271523777445,9.22521328158955,0.6801100159748021,0.26206798787142]   |[-1.7728726239179156,-11.177765120852797]|
|[6.883508221643351,10.147734609748506,0.8501375199685027,0.52413597574284] |[-1.7138964933624494,-12.00737840334759] |
|[6.52121831524107,8.99458294954981,0.7367858506393691,0.52413597574284]    |[-1.7624485738747564,-10.80279308233496] |
|[6.158928408838787,8.072061621390857,0.7934616853039358,0.39310198180713]  |[-1.7749779157017282,-9.806684165653895] |
|[6.883508221643351,8.763952617510071,0.9634891892976364,0.39310198180713]  |[-2.0700941196997897,-10.720429432627522]|
|[6.158928408838787,8.763952617510071,0.8501375199685027,0.39310198180713]  |[-1.6257080769316516,-10.44826393443861] |
+---------------------------------------------------------------------------+-----------------------------------------+
only showing top 20 rows

//The explainedVariance method returns a vector containing the proportions of
//variance explained by each principal component. Our goal is to retain as much variance
//as possible in our new principal components.

scala> pca.explainedVariance
res15: org.apache.spark.ml.linalg.DenseVector = [0.7277045209380264,0.23030523267679512]

Based on the output of the method, the first principal component explains 72.77%
of the variance, while 23.03% of the variance is explained by the second principal
component. Cumulatively, the two principal components explain 95.8% of the
variance. As you can see, we lost some information when we reduced our dimensions.
This is generally an acceptable trade-off if there is substantial training performance
improvement while maintaining good model accuracy.
```

# Recomendations

# Graph Analysis

# Deep Learning

## Deep Learning Frameworks
This section provides a quick overview of some of the most popular deep learning frameworks currently available. This is by no means an exhaustive list. Most of these frameworks have more or less the same features. They differ in the level of community and industry adoption, popularity, and size of ecosystem.

- TensorFlow
TensorFlow is currently the most popular deep learning framework. It was developed at Google as a replacement for Theano. Some of the original developers of Theano went to Google and developed TensorFlow. TensorFlow provides a Python API that runs on top of an engine developed in C/C++.

- Theano
Theano was one of the first open source deep learning frameworks. It was originally released in 2007 by Montreal Institute for Learning Algorithms (MILA) at the University of Montreal. In September 2017 Yoshua Bengio officially announced that development on Theano would end. 

-PyTorch
PyTorch is an open source deep learning library developed by Facebook Artificial Intelligence Research (FAIR) Group. PyTorch user adoption has been recently surging and is the third most popular framework, behind TensorFlow and Keras.

-DeepLearning4J
DeepLearning4J is a deep learning library written in Java. It is compatible with JVM-based languages such as Scala and integrates with Spark and Hadoop. It uses ND4J, its own open source scientific computing library instead of Breeze. DeepLearning4J is a good option for developers who prefer Java or Scala over Python for deep learning (although DeepLearning4J has a Python API that uses Keras).

-CNTK
CNTK, also known as Microsoft Cognitive Toolkit, is a deep learning library developed by Microsoft Research and open sourced in April 2015. It uses a sequence of computational steps using a directed graph to describe a neural network. CNTK supports distributed deep learning across several GPUs and servers.

-Keras
Keras is a high-level deep learning framework developed at Google by Francois Chollet. It offers a simple and modular API that can run on top of TensorFlow, Theano, and CNTK. It enjoys widespread industry and community adoption and a vibrant ecosystem. Keras models can be deployed on iOS and Android devices, in the browser via Keras. js and WebDNN, on Google Cloud, on the JVM through a DL4J import feature from.


## Distributed Deep Learning with Spark

Training complex models like multi-object detectors can take hours, days, or even weeks. In most cases, a single multi-GPU machine is enough to train large models in a reasonable amount of time. For more demanding workloads, spreading computation across multiple machines can dramatically reduce training time, enabling rapid iterative experimentation and accelerating deep learning deployments. Spark’s parallel computing and big data capabilities make it the ideal platform for distributed deep learning. Using Spark for distributed deep learning has additional benefits particularly if you already have an existing Spark cluster. It’s convenient to analyze large amount of data stored on the same cluster where the data are stored such as HDFS, Hive, Impala, or HBase. You might also want to share results with other type of workloads running in the same cluster such as business intelligence, machine learning, ETL, and feature engineering.

Model Parallelism vs. Data Parallelism
There are two main approaches to distributed training of neural networks: model parallelism and data parallelism. In data parallelism, each server in a distributed environment gets a complete replica of the model but only a part of the data. Training is performed locally in each server by the replica of the model on the slice of the full dataset. In model parallelism, the model is split across different servers. Each server is allocated and is responsible for processing a different part of a single neural network, such as a layer. xxx Data parallelism is generally more popular due to its simplicity and ease of implementation. However, model parallelism is preferred for training models that are too big to fit in a single machine. DistBelief, Google’s framework for large-scale distributed deep learning, supports both model and data parallelism. Horovod, a distributed training framework from Uber, also supports both model and data parallelism.

### Distributed Deep Learning Frameworks for Spark
Thanks to third-party contributors, even though Spark’s deep learning support is still under development, there are several external distributed deep learning frameworks that run on top of Spark. We’ll describe the most popular ones.

-Deep Learning Pipelines

Deep Learning Pipelines is a third-party package from Databricks (the company founded by the same people who created Spark) that provides deep learning
functionality that integrates into the Spark ML Pipelines API. The Deep Learning Pipelines API uses TensorFlow and Keras with TensorFlow as a back end. It includes an ImageSchema that can be used to load images into a Spark DataFrame. It supports transfer learning, distributed hyperparameter tuning, and deploying models as SQL functions. Deep Learning Pipelines is still under active development at the time of this demo.

-BigDL

BigDL is a distributed deep learning library for Apache Spark from Intel. It is different from most deep learning frameworks in that it only supports CPU. It uses multithreading and Intel’s Math Kernel Library for Deep Neural Networks (Intel MKL-DNN), an open source library for accelerating the performance of deep learning frameworks on Intel architecture. The performance is said to be comparable with conventional GPUs.

-CaffeOnSpark

CaffeOnSpark is a deep learning framework developed at Yahoo. It is a distributed extension of Caffe designed to run on top of Spark clusters. CaffeOnSpark is extensively used within Yahoo for content classification and image search.

-TensorFlowOnSpark

TensorFlowOnSpark is another deep learning framework developed at Yahoo. It supports distributed TensorFlow inferencing and training using Spark. It integrates with Spark ML Pipelines and supports model and data parallelism and asynchronous and synchronous training.

-TensorFrames

TensorFrames is an experimental library that allows TensorFlow to easily work with Spark DataFrames. It supports Scala and Python and provides an efficient way to pass data from Spark to TensorFlow and vice versa.

-Elephas

Elephas is a Python library that extends Keras to enable highly scalable distributed deep learning with Spark. Developed by Max Pumperla, Elephas implements distributed deep learning using data parallelism and is known for its ease of use and simplicity. It also supports distributed hyperparameter optimization and distributed training of ensemble models.

-Distributed Keras

Distributed Keras (Dist-Keras) is another distributed deep learning framework that run on top of Keras and Spark. It was developed by Joeri Hermans at CERN. It
supports several distributed optimization algorithms such as ADAG, Dynamic SGD, Asynchronous Elastic Averaging SGD (AEASGD), Asynchronous Elastic Averaging
Momentum SGD (AEAMSGD), and Downpour SGD.


Examples: 

1.Elephas: Distributed Deep Learning with Keras and Spark

Example: Handwritten Digit Recognition with MNIST Using Elephas
with Keras and Spark

2.Distributed Keras (Dist-Keras)

Example: Handwritten Digit Recognition with MNIST Using Dist-Keras with Keras and Spark

Example: Dogs and Cats Image Classification

# Appendix1 : Spark on k8s (Production)

When it was released, Apache Spark 2.3 introduced native support for running on top of Kubernetes. Spark [2.4](https://spark.apache.org/docs/2.4.0/index.html) extended this and brought better integration with the Spark shell. In this Appendix1, we'll look at how to get up and running with Spark on top of a Kubernetes cluster.


## Prerequisites

To utilize Spark with Kubernetes, you will need:

- A Kubernetes cluster that has role-based access controls (RBAC) and DNS services enabled
Sufficient cluster resources to be able to run a Spark session (at a practical level, this means at least three nodes with two CPUs and eight gigabytes of free memory)
- A properly configured kubectl that can be used to interface with the Kubernetes API
- Authority as a cluster administrator
- Access to a public Docker repository or your cluster configured so that it is able to pull images from a private repository
- Basic understanding of Apache Spark and its architecture

In this Appendix1, we are going to focus on directly connecting Spark to Kubernetes without making use of the [Spark Kubernetes operator](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator). The Kubernetes operator simplifies several of the manual steps and allows the use of custom resource definitions to manage Spark deployments.

## Overview

In this Appendix1, we will:
- Create a Docker container containing a Spark application that can be deployed on top of Kubernetes
- Integrate Spark with `kubectl` so that is able to start and monitor the status of running jobs
- Demonstrate how to launch Spark applications using `spark-submit`
- Start the Spark Shell and demonstrate how interactive sessions interact with the Kubernetes cluster

## Spark Essentials

Spark is a general cluster technology designed for distributed computation. While primarily used for analytic and data processing purposes, its model is flexible enough to handle distributed operations in a fault tolerant manner. It is a framework that can be used to build powerful data applications.

Every Spark application consists of three building blocks:
- The `Driver` boots and controls all processes. The driver serves as the master node in a Spark application or interactive session. It manages the job of splitting data operations into tasks and then scheduling them to run on executors (which themselves run on nodes of the cluster).
- The `Cluster Manager` helps the driver schedule work across nodes in the cluster using executors. Spark supports several different types of executors. The most common is Hadoop, but Mesos and Kubernetes are both available as options.
- The `Workers` run executors. Executors are distributed across the cluster and do the heavy lifting of a Spark program -data aggregation, machine learning training, and other miscellaneous number crunching. Except when running in "local" mode, executors run on some kind of a cluster to leverage a distributed environment with plenty of resources. They typically are created when a Spark application begins and often run for the entire lifetime of the Spark application. This pattern is called static allocation, and it is also possible to have dynamic allocation of executors which means that they will be initialized when data actually needs to be processed.

### Apache Spark’s Distributed Execution (more detailed)

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-components-and-architecture.png" width="800">

- Spark driver

As the part of the Spark application responsible for instantiating a SparkSession , the Spark driver has multiple roles: it communicates with the cluster manager; it requests resources (CPU, memory, etc.) from the cluster manager for Spark’s executors (JVMs); and it transforms all the Spark operations into DAG computations, schedules them, and distributes their execution as tasks across the Spark executors. Once the resources are allocated, it communicates directly with the executors.

- SparkSession

In Spark 2.0, the SparkSession became a unified conduit to all Spark operations and data. Not only did it subsume previous entry points to Spark like the SparkContext , SQLContext , HiveContext , SparkConf , and StreamingContext , but it also made working with Spark simpler and easier.

Note: Although in Spark 2.x the SparkSession subsumes all other con‐ texts, you can still access the individual contexts and their respective methods. In this way, the community maintained backward compatibility. That is, your old 1.x code with SparkContext or SQLContext will still work.

Through this one conduit, you can create JVM runtime parameters, define Data‐ Frames and Datasets, read from data sources, access catalog metadata, and issue
Spark SQL queries. SparkSession provides a single unified entry point to all of Spark’s functionality. In a standalone Spark application, you can create a SparkSession using one of the high-level APIs in the programming language of your choice. In the Spark shell the SparkSession is created for you, and you can
access it via a global variable called spark or sc .

Whereas in Spark 1.x you would have had to create individual contexts (for streaming, SQL, etc.), introducing extra boilerplate code, in a Spark 2.x application you cancreate a SparkSession per JVM and use it to perform a number of Spark operations.

- Cluster manager
The cluster manager is responsible for managing and allocating resources for the cluster of nodes on which your Spark application runs. Currently, Spark supports
four cluster managers: the built-in standalone cluster manager, Apache Hadoop YARN, Apache Mesos, and Kubernetes.

- Spark executor
A Spark executor runs on each worker node in the cluster. The executors communicate with the driver program and are responsible for executing tasks on the workers. In most deployments modes, only a single executor runs per node.



### Deployment modes

An attractive feature of Spark is its support for myriad deployment modes, enabling Spark to run in different configurations and environments. Because the cluster manager is agnostic to where it runs (as long as it can manage Spark’s executors and fulfill resource requests), Spark can be deployed in some of the most popular environments, such as Apache Hadoop YARN and Kubernetes, and can operate in different modes. 

Summarizes the available deployment modes.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/k8s-spark-cheatsheet-spark-deployment-models.png" width="800">

In a traditional Spark application, a driver can either run inside or outside of a cluster. Depending on where it executes, it will be described as running in "client mode" or "cluster mode."

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/spark-architecture.png" width="800">

Spark is composed of three components: A driver that tracks the general logic of a Spark program, a cluster manager which helps to find workers, and workers which execute data operations and report results.


### Networking Considerations: Executor-Driver Communication in Kubernetes

When Spark deploys an application inside of a Kubernetes cluster, Kubernetes doesn't handle the job of scheduling executor workload. Rather, its job is to spawn a small army of executors (as instructed by the cluster manager) so that workers are available to handle tasks. The driver then coordinates what tasks should be executed and which executor should take it on. Once work is assigned, executors execute the task and report the results of the operation back to the driver.

This last piece is important. Since a cluster can conceivably have hundreds or even thousands of executors running, the driver doesn't actively track them and request a status. Instead, the executors themselves establish a direct network connection and report back the results of their work. In complex environments, firewalls and other network management layers can block these connections from the executor back to the master. If this happens, the job fails.

This means that we need to take a degree of care when deploying applications. Kubernetes pods are often not able to actively connect to the launch environment (where the driver is running). If the job was started from within Kubernetes or is running in "cluster" mode, it's usually not a problem. All networking connections are from within the cluster, and the pods can directly see one another.

In client mode (which is how most Spark shells run), this is a problem. The executor instances usually cannot see the driver which started them, and thus they are not able to communicate back their results and status. This means interactive operations will fail.

Based on these requirements, the easiest way to ensure that your applications will work as expected is to package your driver or program as a pod and run that from within the cluster. In this post, we'll show how you can do that. First, we'll look at how to package Spark driver components in a pod and use that to submit work into the cluster using the "cluster mode." Then we'll show how a similar approach can be used to submit client mode applications, and the additional configuration required to make them work.

The ability to launch client mode applications is important because that is how most interactive Spark applications run, such as the PySpark shell.

## Proof of Concept

Any relatively complex technical project usually starts with a proof of concept to show that the goals are feasible. Spark on top of Kubernetes has a lot of moving parts, so it's best to start small and get more complicated after we have ensured that lower-level pieces work. To that end, in this post we will use a minimalist set of containers with the basic Spark runtime and toolset to ensure that we can get all of the parts and pieces configured in our cluster. Specifically, we will:

- Build the containers for the driver and executors using a multi-stage Dockerfile. We use a multi-stage Docker container to show how the entire build process can be automated. The Dockerfile can be modified later to inject additional components specific to the types of analysis, or other tools you might need.
- Create a service account and configure the authentication parameters required by Spark to connect to the Kubernetes control plane and launch workers.
- Start the containers and submit a sample job (calculating Pi) to test the setup.

### Building Containers

Pods are container runtimes which are instantiated from container images, and will provide the environment in which all of the Spark workloads run. While there are several container runtimes, the most popular is Docker. In this section, we'll create a set of container images that provide the fundamental tools and libraries needed by our environment.

In Docker, container images are built from a set of instructions collectively called a Dockerfile. Each line of a Dockerfile has an instruction and a value. Instructions are things like "run a command", "add an environment variable", "expose a port", and so-forth.

- Base Image

The code listing shows a [multi-stage Dockerfile](https://docs.docker.com/develop/develop-images/multistage-build/) which will build our base Spark environment. This will be used for running executors and as the foundation for the driver. In the first stage of the build we download the Apache Spark runtime to a temporary directory, extract it, and then copy the runtime components for Spark to a new container image. Using a multi-stage process allows us to automate the entire container build using the packages from the [Apache Spark downloads page](https://spark.apache.org/downloads.html).

In the second step, we configure the Spark container, set environment variables, patch a set of dependencies to avoid errors, and specify a non-root user which will be used to run Spark when the container starts.

Using the Docker image, we can build and tag the image. When it finishes, we need to push it to an external repository for it to be available for our Kubernetes cluster. The command in the listing shows how this might be done.

We use a DockerHub  public Docker registry. The image needs to be hosted somewhere accessible in order for Kubernetes to be able to use it. While it is possible to pull from a private registry, this involves additional steps and is not covered in this Appendix1.


- Driver Image

For the driver, we need a small set of additional resources that are not required by the executor/base image, including a copy of Kube Control that will be used by Spark to manage workers. The container is the same as the executor image in most other ways and because of that we use the executor image as the base.


As with the executor image, we need to build and tag the image, and then push to the registry.

```
cd ./jupyter/docker
docker login
# Build and tag the base/executor image
docker build -f ./Dockerfile.k8s-minio.executor -t davarski/spark301-k8s-minio-base .
# Push the contaimer image to a public registry
docker push davarski/spark301-k8s-minio-base

# Build and tag the driver image
docker build -f ./Dockerfile.k8s-minio.driver -t davarski/spark301-k8s-minio-driver .
# Push the contaimer image to a public registry
docker push davarski/spark301-k8s-minio-driver

# Appendix2 (Build/tag/push the jupyter image)
docker build -f ./Dockerfile.k8s-minio.jupyter -t davarski/spark301-k8s-minio-jupyter .
docker push davarski/spark301-k8s-minio-jupyter
```
Pull images into k8s(k3s):
```
export KUBECONFIG=~/.kube/k3s-config-jupyter 
sudo k3s crictl pull davarski/spark301-k8s-minio-base
sudo k3s crictl pull davarski/spark301-k8s-minio-driver

# Appendix2
sudo k3s crictl pull davarski/spark301-k8s-minio-jupyter
```


### Service Accounts and Authentication

For the driver pod to be able to connect to and manage the cluster, it needs two important pieces of data for authentication and authorization:

- The CA certificate, which is used to connect to the kubelet control daemon
- The auth (or bearer) token, which identifies a user and the scope of its permissions

There are a variety of strategies which might be used to make this information available to the pod, such as creating a secret with the values and mounting the secret as a read-only volume. A Kubernetes secret lets you store and manage sensitive information such as passwords. An easier approach, however, is to use a service account that has been authorized to work as a cluster admin. One of the cool things that Kubernetes does when running a pod under a service account is to create a volumeSource (basically a read-only mount) with details about the user context in which a pod is running.

Inside of the mount will be two files that provide the authentication details needed by kubectl:

    /var/run/secrets/kubernetes.io/serviceaccount/ca.crt: CA certificate
    /var/run/secrets/kubernetes.io/serviceaccount/token: Kubernetes authentication token

- Driver Service Account

The set of commands below will create a special service account (spark-driver) that can be used by the driver pods. It is configured to provide full administrative access to the namespace.
```
# Create spark-driver service account
kubectl create serviceaccount spark-driver

# Create a cluster and namespace "role-binding" to grant the account administrative privileges
kubectl create rolebinding spark-driver-rb --clusterrole=cluster-admin --serviceaccount=default:spark-driver
```

- Executor Service Account

While it is possible to have the executor reuse the spark-driver account, it's better to use a separate user account for workers. This allows for finer-grained tuning of the permissions. The worker account uses the "edit" permission, which allows for read/write access to most resources in a namespace but prevents it from modifying important details of the namespace itself.
```
# Create Spark executor account
kubectl create serviceaccount spark-minion

# Create rolebinding to offer "edit" privileges
kubectl create rolebinding spark-minion-rb --clusterrole=edit --serviceaccount=default:spark-minion
```

### Running a Test Job

With the images created and service accounts configured, we can run a test of the cluster using an instance of the spark301-k8s-minio-driver image. The command below will create a pod instance from which we can launch Spark jobs.

Creating a pod to deploy cluster and client mode Spark applications is sometimes referred to as deploying a "jump", "edge" , or "bastian" pod. It's variant of deploying a Bastion Host, where high-value or sensitive resources run in one environment and the bastion serves as a proxy.

```
# Create a jump pod using the Spark driver container and service account
kubectl run spark-test-pod --generator=run-pod/v1 -it --rm=true \
  --image=davarski/spark301-k8s-minio-driver \
  --serviceaccount=spark-driver \
  --command -- /bin/bash
```

The kubectl command creates a deployment and driver pod, and will drop into a BASH shell when the pod becomes available. The remainder of the commands in this section will use this shell.

Apache's Spark distribution contains an example program that can be used to calculate Pi. Since it works without any input, it is useful for running tests. We can check that everything is configured correctly by submitting this application to the cluster. Spark commands are submitted using spark-submit. In the container images created above, spark-submit can be found in the /opt/spark/bin folder.

spark-submit commands can become quite complicated. For that reason, let's configure a set of environment variables with important runtime parameters. While we define these manually here, in applications they can be injected from a ConfigMap or as part of the pod/deployment manifest.

```
# Define environment variables with accounts and auth parameters
export SPARK_NAMESPACE=default
export SA=spark-minion
export K8S_CACERT=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
export K8S_TOKEN=/var/run/secrets/kubernetes.io/serviceaccount/token

# Docker runtime image
export DOCKER_IMAGE=davarski/spark301-k8s-minio-base
export SPARK_DRIVER_NAME=spark-test1-pi
```
The command below submits the job to the cluster. It will deploy in "cluster" mode and references the `spark-examples` JAR from the container image. We tell Spark which program within the JAR to execute by defining a --class option. In this case, we wish to run `org.apache.spark.examples.SparkPi`

```
/opt/spark/bin/spark-submit --name sparkpi-test1 \
   --master k8s://https://kubernetes.default:443 \
  --deploy-mode cluster  \
  --class org.apache.spark.examples.SparkPi  \
  --conf spark.kubernetes.driver.pod.name=$SPARK_DRIVER_NAME  \
  --conf spark.kubernetes.authenticate.subdmission.caCertFile=$K8S_CACERT  \
  --conf spark.kubernetes.authenticate.submission.oauthTokenFile=$K8S_TOKEN  \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=$SA  \
  --conf spark.kubernetes.namespace=$SPARK_NAMESPACE  \
  --conf spark.executor.instances=2  \
  --conf spark.kubernetes.container.image=$DOCKER_IMAGE  \
  --conf spark.kubernetes.container.image.pullPolicy=Always \
  local:///opt/spark/examples/jars/spark-examples_2.12-3.0.1.jar 1000
```



The Kubernetes control API is available within the cluster within the `default` namespace and should be used as the Spark master. If Kubernetes DNS is available, it can be accessed using a namespace URL (`https://kubernetes.default:443` in the example above). Note the `k8s://https://` form of the URL. as this is not a typo. The `k8s://` prefix is how Spark knows the provider type.

The `local://` path of the `jar` above references the file in the executor Docker image, not on jump pod that we used to submit the job. Both the driver and executors rely on the path in order to find the program logic and start the task.

If you watch the pod list while the job is running using `kubectl get pods`, you will see a "driver" pod be initialized with the name provided in the `SPARK_DRIVER_NAME` variable. This will in turn launch executor pods where the work will actually be performed. When the program has finished running, the driver pod will remain with a "Completed" status. You can retrieve the results from the pod logs using:

```
# Retrieve the results of the program from the cluster
kubectl logs $SPARK_DRIVER_NAME

Toward the end of the application log you should see a result line similar to the one below:

20/12/19 10:56:11 INFO DAGScheduler: Job 0 finished: reduce at SparkPi.scala:38, took 16.059215 s
Pi is roughly 3.1416641114166413
20/12/19 10:56:11 INFO SparkUI: Stopped Spark web UI at http://spark-test-pod.default:4040
20/12/19 10:56:11 INFO KubernetesClusterSchedulerBackend: Shutting down all executors
20/12/19 10:56:11 INFO KubernetesClusterSchedulerBackend$KubernetesDriverEndpoint: Asking each executor to shut down
20/12/19 10:56:11 WARN ExecutorPodsWatchSnapshotSource: Kubernetes client has been closed (this is expected if the application is shutting down.)
20/12/19 10:56:11 INFO MapOutputTrackerMasterEndpoint: MapOutputTrackerMasterEndpoint stopped!
20/12/19 10:56:12 INFO MemoryStore: MemoryStore cleared
20/12/19 10:56:12 INFO BlockManager: BlockManager stopped
20/12/19 10:56:12 INFO BlockManagerMaster: BlockManagerMaster stopped
20/12/19 10:56:12 INFO OutputCommitCoordinator$OutputCommitCoordinatorEndpoint: OutputCommitCoordinator stopped!
20/12/19 10:56:12 INFO SparkContext: Successfully stopped SparkContext
20/12/19 10:56:12 INFO ShutdownHookManager: Shutdown hook called
20/12/19 10:56:12 INFO ShutdownHookManager: Deleting directory /tmp/spark-4a81c68d-6ffd-4b9c-831b-4587fccc4d12
20/12/19 10:56:12 INFO ShutdownHookManager: Deleting directory /tmp/spark-fb05ba62-8eed-41ea-bd6d-f6aea49021b5

```

### Client Mode Applications

When we switch from cluster to client mode, instead of running in a separate pod, the driver will run within the jump pod instance. This requires an additional degree of preparation, specifically:

- Because executors need to be able to connect to the driver application, we need to ensure that it is possible to route traffic to the pod and that we have published a port which the executors can use to communicate. To make the pod instance (easily) routable, we will create a headless service.
- Since the driver will be running from the jump pod, we need to modify the `SPARK_DRIVER_NAME` environment variable to reference that rather than an external (to be launched) pod.
- We need to provide additional configuration options to reference the driver host and port. These should then be passed to `spark-submit` via the `spark.driver.host` and `spark.driver.port` options, respectively.

#### Running Client Mode Applications Using `spark-submit`

To test client mode on the cluster, let's make the changes outlined above and then submit SparkPi a second time.

To start, because the driver will be running from the jump pod, let's modify `SPARK_DRIVER_NAME` environment variable and specify which port the executors should use for communicating their status.
```
# Modify the name of the spark driver 
export SPARK_DRIVER_NAME=spark-test-pod
export SPARK_DRIVER_PORT=20020
```
Next, to route traffic to the pod, we need to either have a domain or IP address. In Kubernetes, the most convenient way to get a stable network identifier is to create a service object. The command below will create a "headless" service that will allow other pods to look up the jump pod using its name and namespace.
```
# Expose the jump pod using a headless service
kubectl expose pod $SPARK_DRIVER_NAME --port=$SPARK_DRIVER_PORT \
  --type=ClusterIP --cluster-ip=None
```
Taking into account the changes above, the new `spark-submit` command will be similar to the one below:
```
/opt/spark/bin/spark-submit --name sparkpi-test1 \
   --master k8s://https://kubernetes.default:443 \
  --deploy-mode client  \
  --class org.apache.spark.examples.SparkPi  \
  --conf spark.kubernetes.driver.pod.name=$SPARK_DRIVER_NAME  \
  --conf spark.kubernetes.authenticate.subdmission.caCertFile=$K8S_CACERT  \
  --conf spark.kubernetes.authenticate.submission.oauthTokenFile=$K8S_TOKEN  \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=$SA  \
  --conf spark.kubernetes.namespace=$SPARK_NAMESPACE  \
  --conf spark.executor.instances=2  \
  --conf spark.kubernetes.container.image=$DOCKER_IMAGE  \
  --conf spark.kubernetes.container.image.pullPolicy=Always \
  --conf spark.driver.host=$HOSTNAME.$SPARK_NAMESPACE \
  --conf spark.driver.port=$SPARK_DRIVER_PORT \
  local:///opt/spark/examples/jars/spark-examples_2.12-3.0.1.jar 1000
```
Upon submitting the job, the driver will start and launch executors that report their progress. For this reason, we will see the results reported directly to `stdout` of the jump pod, rather than requiring we fetch the logs of a secondary pod instance.

As in the previous example, you should be able to find a line reporting the calculated value of Pi.

#### Starting the `pyspark` Shell

At this point, we've assembled all the pieces to show how an interactive Spark program (like the `pyspark` shell) might be launched. Similar to the client mode application, the shell will directly connect with executor pods which allows for calculations and other logic to be distributed, aggregated, and reported back without needing a secondary pod to manage the application execution.

The command below shows the options and arguments required to start the shell. It is similar to the spark-submit commands we've seen previously (with many of the same options), but there are some distinctions. The most consequential differences are:

- The shell is started using the `pyspark` script rather than `spark-submit` (`pyspark` is located in the same `/opt/spark/bin` directory as `spark-submit`)
- There is no main class or `jar` file referenced

```
# Define environment variables with accounts and auth parameters
export SPARK_NAMESPACE=default
export SA=spark-minion
export K8S_CACERT=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
export K8S_TOKEN=/var/run/secrets/kubernetes.io/serviceaccount/token

# Docker runtime image
export DOCKER_IMAGE=davarski/spark301-k8s-minio-base

# Modify the name of the spark driver 
export SPARK_DRIVER_NAME=spark-test-pod
export SPARK_DRIVER_PORT=20020

/opt/spark/bin/pyspark --name pyspark-test1 \
   --master k8s://https://kubernetes.default:443 \
  --deploy-mode client  \
  --conf spark.kubernetes.driver.pod.name=$SPARK_DRIVER_NAME  \
  --conf spark.kubernetes.authenticate.subdmission.caCertFile=$K8S_CACERT  \
  --conf spark.kubernetes.authenticate.submission.oauthTokenFile=$K8S_TOKEN  \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=$SA  \
  --conf spark.kubernetes.namespace=$SPARK_NAMESPACE  \
  --conf spark.executor.instances=2  \
  --conf spark.kubernetes.container.image=$DOCKER_IMAGE  \
  --conf spark.kubernetes.container.image.pullPolicy=Always \
  --conf spark.driver.host=$HOSTNAME.$SPARK_NAMESPACE \
  --conf spark.driver.port=$SPARK_DRIVER_PORT
```

After launch, it will take a few seconds or minutes for Spark to pull the executor container images and configure pods. When ready, the shell prompt will load. At that point, we can run a distributed Spark calculation to test the configuration:
```
# Create a distributed data set to test the session.
t = sc.parallelize(range(10))

# Calculate the approximate sum of values in the dataset
r = t.sumApprox(3)
print('Approximate sum: %s' % r)
```
If everything works as expected, you should see something similar to the output below:
```
Approximate sum: 45
```

The PySpark shell runs as a client application in Kubernetes

You can exit the shell by typing exit() or by pressing Ctrl+D. The spark-test-pod instance will delete itself automatically because the --rm=true option was used when it was created. You will need to manually remove the service created using kubectl expose. If you followed the earlier instructions, kubectl delete svc spark-test-pod should remove the object.

### Next Steps

Running Spark on the same Kubernetes infrastructure that you use for application deployment allows you to consolidate Big Data workloads inside the same infrastructure you use for everything else. In this Apendix1, we've seen how you can use jump pods and custom images to run Spark applications in both cluster and client mode.

While useful by itself, this foundation opens the door to deploying Spark alongside more complex analytic environments such as Jupyter or JupyterHub. In Apendix2 of this Demo, we will show how to extend the driver container with additional Python components and access our cluster resources from a Jupyter Kernel.

# Appendix2: Spark on k8s: Jupyter

After we've got Spark running on Kubernetes, we can integrate the runtime with applications like Jupyter? 

In many organizations, Apache Spark is the computational engine that powers big data. Spark, a general-purpose unified analytics engine built to transform, aggregate, and analyze large amounts of information, has become the de-facto brain behind large scale data processing, machine learning, and graph analysis.

When it was released, Apache Spark 2.3 introduced native support for running on top of Kubernetes. Spark 2.4 further extended the support and brought integration with the Spark shell. In a previous Apendix1, we showed the preparations and setup required to get Spark up and running on top of a Kubernetes cluster.

In this Apendix2, we'll take the next logical step and show how to run more complex analytic environments such as Jupyter so that it is also able to take advantage of the cluster for data exploration, visualization, or interactive prototyping.

This Apendix2 is how to use containers and Kubernetes for Data Science. Please check out Apendix1 which shows how to run Spark applications inside of Kubernetes.

## Jupyter

[Jupyter](https://jupyter.org/) allows you to work interactively work with a live running server and iteratively execute logic which remains persistent as long as the kernel is running. It is used to combine live-running code alongside images, data visualization, and other interactive elements such as maps. It has become a de-facto standard for exploratory data analysis and technical communication.



### Overview

In this Appendix2, we will:

- Extend the "driver" container in the previous Appendix1 to include Jupyter and integrate the traditional Python shell with PySpark so that it can run large analytic workloads on the cluster.
- Configure S3Contents, a drop-in replacement for the standard filesystem-backed storage system in Jupyter. Using S3Contents is desirable because containers are fragile. They will often crash or disappear, and when that happens the content of their filesystems is lost. When running in Kubernetes, it is therefore important to provide an external storage that will remain available if the container disappears.
- Create an "ingress" that allows for the Jupyter instance to be accessed from outside of the cluster.


### Packaging Jupyter

As a Python application, Jupyter can be installed with either pip or conda. We will be using `pip`.

The container images we created previously (`spark301-k8s-minio-base` and `spark301-k8s-minio-driver`) both have `pip` installed. For that reason, we can extend them directly to include Jupyter and other Python libraries.

The Dockerfiles used below shows we install Jupyter, S3Contents, and a small set of other common data science libraries including:

- [NumPy](https://numpy.org/): A library which implements efficient N-dimensional arrays, tools for manipulating data inside of NumPy arrays, interfaces for integrating C/C++ and Fortran code with Python, and a library of linear algebra, Fourier transform, and random number capabilities.
- [Matplotlib](https://matplotlib.org/): A popular data visualization library designed to create charts and graphs.
- [Seaborn](https://seaborn.pydata.org/): A set of additional tools based on matplotlib which extends the basic interfaces for creating statistical charts intended for data exploration.
- [scikit-learn](https://scikit-learn.org/stable/): A machine learning library for Python that provides simple tools for data mining and analysis; preprocessing and model selection, as well as implementations of classification, regression, clustering, and dimensionality reduction models.


In addition to the Data Science libraries, the Dockerfile also configures a user for Jupyter and a working directory. This is done because it is (generally) a bad idea to run a Dockerized application as root. This may seem an arbitrary concern as the container will be running as a privileged user inside of the Kubernetes cluster and will have the ability to spawn other containers within its namespace. One of the things that Jupyter provides is a shell interface. By running as a non-privileged user, there is some degree of isolation in case the notebook server becomes compromised.

The dependency installation is split over multiple lines in order to decrease the size of the layers. Large Docker image layers may experience timeouts or other transport issues. This makes container design something of an art. It's a good idea to keep container images as small as possible with as few layers as possible, but you still need to provide the tools to ensure that the container is useful.



### Build/tag/testing the image.


When the container finishes building, we will want to test it locally to ensure that the application starts.


```
cd ./jupyter/docker

docker login

# Build and tag the jupyter
docker build -f ./Dockerfile.k8s-minio.jupyter -t davarski/spark301-k8s-minio-jupyter .

# Push the Jupyter container image to a remote registry
docker push davarski/spark301-k8s-minio-jupyter
```
Pull images into k8s(k3s):
```
export KUBECONFIG=~/.kube/k3s-config-jupyter 

sudo k3s crictl pull davarski/spark301-k8s-minio-jupyter
```

Testing the Image Locally

```
# Test the container image locally to ensure that it starts as expected.
# Jupyter Lab/Notebook is started using the command jupyter lab.
# We provide the --ip 0.0.0.0 so that it will bind to all interfaces.
docker run -it --rm -p 8888:8888 \
    davarski/spark301-k8s-minio-jupyter \
    jupyter lab --ip 0.0.0.0
```
Example output:

```
$ docker run -it --rm -p 8888:8888 davarski/spark301-k8s-minio-jupyter jupyter lab --ip 0.0.0.0
++ id -u
+ myuid=1000
++ id -g
+ mygid=777
+ set +e
++ getent passwd 1000
+ uidentry=jovyan:x:1000:777::/home/jovyan:/bin/bash
+ set -e
+ '[' -z jovyan:x:1000:777::/home/jovyan:/bin/bash ']'
+ SPARK_CLASSPATH=':/opt/spark/jars/*'
+ env
+ grep SPARK_JAVA_OPT_
+ sort -t_ -k4 -n
+ sed 's/[^=]*=\(.*\)/\1/g'
+ readarray -t SPARK_EXECUTOR_JAVA_OPTS
+ '[' -n '' ']'
+ '[' '' == 2 ']'
+ '[' '' == 3 ']'
+ '[' -n '' ']'
+ '[' -z ']'
+ case "$1" in
+ echo 'Non-spark-on-k8s command provided, proceeding in pass-through mode...'
Non-spark-on-k8s command provided, proceeding in pass-through mode...
+ CMD=("$@")
+ exec /usr/bin/tini -s -- jupyter lab --ip 0.0.0.0
[I 11:21:27.529 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 11:21:27.844 LabApp] JupyterLab extension loaded from /usr/local/lib/python3.8/dist-packages/jupyterlab
[I 11:21:27.845 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
[I 11:21:27.847 LabApp] Serving notebooks from local directory: /home/jovyan/work
[I 11:21:27.847 LabApp] Jupyter Notebook 6.1.5 is running at:
[I 11:21:27.847 LabApp] http://b6456cfe1ee2:8888/?token=c8abb0f836f8e379800088dd42840a82f07a0dd085eb0ff0
[I 11:21:27.847 LabApp]  or http://127.0.0.1:8888/?token=c8abb0f836f8e379800088dd42840a82f07a0dd085eb0ff0
[I 11:21:27.847 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 11:21:27.851 LabApp] No web browser found: could not locate runnable browser.
[C 11:21:27.851 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-13-open.html
    Or copy and paste one of these URLs:
        http://b6456cfe1ee2:8888/?token=c8abb0f836f8e379800088dd42840a82f07a0dd085eb0ff0
     or http://127.0.0.1:8888/?token=c8abb0f836f8e379800088dd42840a82f07a0dd085eb0ff0

```
Once the program starts, you will see an entry in the logs which says, "To access the notebook ... copy and paste one of these URLs ...". Included at the end of the URL is a "token" value that is required to authenticate to the server. Copy this to your system clipboard.

Included in the Jupyter startup logs will be an access URL that includes a "token". This value is required for authentication to the server.

Upon visiting the URL you will be prompted for the token value (unless you copied the entire access URL and pasted that into the navigation bar of the browser). Paste the token into the authentication box and click "Log In." You will be taken to the Jupyter Dashboard/Launcher.


Testing the Container in Kubernetes

Once you've verified that the container image works as expected in our local environment, we need to validate that it also runs in Kubernetes. This involves three steps:

- Pushing the container image to a public repository so that it can be deployed onto the cluster
- Launching an instance inside of Kubernetes using kubectl run
- Connecting to the container instance by mapping a port from the pod to the local environment using kubectl port-forward

To perform these steps, you will need two terminals. In the first, you will run the following two commands:

```
# Push the Jupyter container image to a remote registry
docker push davarski/spark301-k8s-minio-jupyter

# Start an instance of the container in Kubernetes
kubectl run jupyter-test-pod --generator=run-pod/v1 -it --rm=true \
  --image=davarski/spark301-k8s-minio-jupyter \
  --serviceaccount=spark-driver \
  --command -- jupyter lab --ip 0.0.0.0
```
Example output:

```
$ kubectl run jupyter-test-pod --generator=run-pod/v1 -it --rm=true --image=davarski/spark301-k8s-minio-jupyter --serviceaccount=spark-driver  --command -- jupyter lab --ip 0.0.0.0
Flag --generator has been deprecated, has no effect and will be removed in the future.
If you don't see a command prompt, try pressing enter.
[I 11:26:46.920 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 11:26:47.235 LabApp] JupyterLab extension loaded from /usr/local/lib/python3.8/dist-packages/jupyterlab
[I 11:26:47.235 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
[I 11:26:47.237 LabApp] Serving notebooks from local directory: /home/jovyan/work
[I 11:26:47.237 LabApp] Jupyter Notebook 6.1.5 is running at:
[I 11:26:47.237 LabApp] http://jupyter-test-pod:8888/?token=38b6ea6bc249317c97e85319218d912daa4a7730404dafca
[I 11:26:47.237 LabApp]  or http://127.0.0.1:8888/?token=38b6ea6bc249317c97e85319218d912daa4a7730404dafca
[I 11:26:47.237 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 11:26:47.241 LabApp] No web browser found: could not locate runnable browser.
[C 11:26:47.241 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://jupyter-test-pod:8888/?token=38b6ea6bc249317c97e85319218d912daa4a7730404dafca
     or http://127.0.0.1:8888/?token=38b6ea6bc249317c97e85319218d912daa4a7730404dafca

```


When the server is active, note the access URL and token value. Then, in the second terminal, run kubectl port-forward to map a local port to the container.

```
# Forward a port in the local environment to the pod to test the runtime
kubectl port-forward pod/jupyter-test-pod 8888:8888
```
With the port-forward running, open a browser and navigate to the locally mapped port (8088 in the example command above). Provide the token value and click "Log In." Like the local test, you should be routed to the Jupyter dashboard. Seeing the dashboard gives some confidence that the container image works as expected, but that doesn't test the Spark integration.

To test Spark, we need to do two things:

- Create a service so that executor pods are able to connect to the driver. Without a service, the executors will be unable to report their task progress to the driver and tasks will fail.
- Open a Jupyter Notebook, and initialize a `SparkContext`.

First, let's create the service.
```
kubectl expose pod jupyter-test-pod --type=ClusterIP --cluster-ip=None
```
With the service in place, let's initialize the SparkContext. From the launcher, click on the "Python 3" link under "Notebook." This will start a new Python 3 kernel and open the Notebook interface.

To test the Spark connection, we need to intialize a `SparkContext`.

Copy the code from the listing below into the notebook and execute the cell. The code defines the parameters needed by Spark to connect to the cluster and launch worker instances. It defines the URL to the Spark master, the container image that should be used for launching workers, the location of the authentication certificate and token, the service account which should be used by the driver instance, and the driver host and port. Specific values may need to be modified for your environment. For details on the parameters, refer to Appendix1 of this Demo.

```
import pyspark

conf = pyspark.SparkConf()

# Kubernetes is a Spark master in our setup. 
# It creates pods with Spark workers, orchestrates those 
# workers and returns final results to the Spark driver 
# (“k8s://https://” is NOT a typo, this is how Spark knows the “provider” type). 
conf.setMaster("k8s://https://kubernetes.default:443") 

# Worker pods are created from the base Spark docker image.
# If you use another image, specify its name instead.
conf.set(
    "spark.kubernetes.container.image", 
    "davarski/spark301-k8s-minio-base") 

# Authentication certificate and token (required to create worker pods):
conf.set(
    "spark.kubernetes.authenticate.caCertFile", 
    "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
conf.set(
    "spark.kubernetes.authenticate.oauthTokenFile", 
    "/var/run/secrets/kubernetes.io/serviceaccount/token")

# Service account which should be used for the driver
conf.set(
    "spark.kubernetes.authenticate.driver.serviceAccountName", 
    "spark-driver") 

# 2 pods/workers will be created. Can be expanded for larger workloads.
conf.set("spark.executor.instances", "2") 

# The DNS alias for the Spark driver. Required by executors to report status.
conf.set(
    "spark.driver.host", "jupyter-test-pod") 

# Port which the Spark shell should bind to and to which executors will report progress
conf.set("spark.driver.port", "29413") 

# Initialize spark context, create executors
sc = pyspark.SparkContext(conf=conf)
```
When the cell finishes executing, add the following code to a second cell and execute that. If successful, it will verify that Jupyter, Spark, Kubernetes, and the container images are all configured correctly.
```
# Create a distributed data set to test to the session
t = sc.parallelize(range(10))

# Calculate the approximate sum of values in the dataset
r = t.sumApprox(3)
print('Approximate sum: %s' % r)
```
Output:
```
Approximate sum: 45.0

```

Check k8s pods:
```
kubectl get po 
jupyter-test-pod                        1/1     Running     0          37m
busybox                                 1/1     Running     312        28d
dnsutils                                1/1     Running     308        28d
pyspark-shell-395a80767ae1e917-exec-1   1/1     Running     0          41s
pyspark-shell-395a80767ae1e917-exec-2   1/1     Running     0          41s
```


### Cloud Native Applications (simple)

Simple setup :

```
$ kubectl create -f ./k8s/jupyter-notebook.pod.yaml
$ kubectl create -f ./k8s/jupyter-notebook.svc.yaml
$ kubectl create -f ./k8s/jupyter-notebook.ingress.yaml
$ kubectl logs spark-jupyter
[I 06:31:54.383 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 06:32:07.846 LabApp] JupyterLab extension loaded from /usr/local/lib/python3.8/dist-packages/jupyterlab
[I 06:32:07.846 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
[I 06:32:07.860 LabApp] Serving notebooks from local directory: /home/jovyan/work
[I 06:32:07.860 LabApp] Jupyter Notebook 6.1.5 is running at:
[I 06:32:07.860 LabApp] http://spark-jupyter:8888/?token=74f5a0c7e5b281e9a2d3762f59b228970db65ae3e22bbd86
[I 06:32:07.861 LabApp]  or http://127.0.0.1:8888/?token=74f5a0c7e5b281e9a2d3762f59b228970db65ae3e22bbd86
[I 06:32:07.861 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 06:32:07.873 LabApp] No web browser found: could not locate runnable browser.
[C 06:32:07.873 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://spark-jupyter:8888/?token=74f5a0c7e5b281e9a2d3762f59b228970db65ae3e22bbd86
     or http://127.0.0.1:8888/?token=74f5a0c7e5b281e9a2d3762f59b228970db65ae3e22bbd86


```
Login to Juputer (UI)  https://jupyter.data.davar.com. using above token: 74f5a0c7e5b281e9a2d3762f59b228970db65ae3e22bbd86

With the service in place, let's initialize the `SparkContext`. From the launcher, click on the "Python 3" link under "Notebook." This will start a new Python 3 kernel and open the Notebook interface.

To test the Spark connection, we need to intialize a SparkContext.

Copy the code from the listing below into the notebook and execute the cell. The code defines the parameters needed by Spark to connect to the cluster and launch worker instances. It defines the URL to the Spark master, the container image that should be used for launching workers, the location of the authentication certificate and token, the service account which should be used by the driver instance, and the driver host and port. Specific values may need to be modified for your environment. For details on the parameters, refer to Appendix1 of this Demo.

```
import pyspark

conf = pyspark.SparkConf()

# Kubernetes is a Spark master in our setup. 
# It creates pods with Spark workers, orchestrates those 
# workers and returns final results to the Spark driver 
# (“k8s://https://” is NOT a typo, this is how Spark knows the “provider” type). 
conf.setMaster("k8s://https://kubernetes.default:443") 

# Worker pods are created from the base Spark docker image.
# If you use another image, specify its name instead.
conf.set(
    "spark.kubernetes.container.image", 
    "davarski/spark301-k8s-minio-base") 

# Authentication certificate and token (required to create worker pods):
conf.set(
    "spark.kubernetes.authenticate.caCertFile", 
    "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
conf.set(
    "spark.kubernetes.authenticate.oauthTokenFile", 
    "/var/run/secrets/kubernetes.io/serviceaccount/token")

# Service account which should be used for the driver
conf.set(
    "spark.kubernetes.authenticate.driver.serviceAccountName", 
    "spark-driver") 

# 2 pods/workers will be created. Can be expanded for larger workloads.
conf.set("spark.executor.instances", "2") 

# The DNS alias for the Spark driver. Required by executors to report status.
conf.set(
    "spark.driver.host", "spark-jupyter") 

# Port which the Spark shell should bind to and to which executors will report progress
conf.set("spark.driver.port", "29413") 

# Initialize spark context, create executors
sc = pyspark.SparkContext(conf=conf)
```
When the cell finishes executing, add the following code to a second cell and execute that. If successful, it will verify that Jupyter, Spark, Kubernetes, and the container images are all configured correctly.
```
# Create a distributed data set to test to the session
t = sc.parallelize(range(10))

# Calculate the approximate sum of values in the dataset
r = t.sumApprox(3)
print('Approximate sum: %s' % r)
```
Output:
```
Approximate sum: 45.0

```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/k3s-spark-jupyter.png" width="800">

Check k8s pods:
```
kubectl get po 
spark-jupyter                           1/1     Running     0          10m
busybox                                 1/1     Running     312        28d
dnsutils                                1/1     Running     308        28d
pyspark-shell-88a623767bd18995-exec-2   1/1     Running     0          19s
pyspark-shell-88a623767bd18995-exec-1   1/1     Running     0          20s
```
Note: exit() into cell will termintate pyspark-shell-88a623767bd18995-exec-2 & pyspark-shell-88a623767bd18995-exec-1 pods

### Cloud Native Applications (with MinIO:S3)

While the tests at the end of the previous section give us confidence that the Jupyter container image works as expected, we still don't have a robust "cloud native" application that we would want to deploy on a permanent basis.

- The first major problem is that the container storage will be transient. If the container instance restarts or gets migrated to a new host, any saved notebooks will be lost.
- The second major problem also arises in the context of a container restart. At the time it starts, the container looks for a token or password and generates a new random one if it is absent. This means that if the container gets migrated, the previous token will no longer work and the user will need to access the pod logs to learn what the new value is. That would require giving all users access to the cluster
- The third problem is that there is no convenient way to access the instance from outside of the cluster. Using kubectl to forward the application ports works great for testing, but there should be a more proper way to access the resource for users who lack administrative Kubernetes access.

The first two problems can be mediated by configuring resources for the container and injecting them into the pod as part of its deployment. The third problem can be solved by creating an ingress,.

S3Contents: Cloud Storage for Jupyter

Of the three problems, the most complex to solve is the first: Dealing with the transient problem. There are a number of approaches we might take:

- Creating a pod volume that mounts when the container starts that is backed by some type of PersistentVolume
- Deploying the application as part of resource that can be tied to physical storage on one of the hosts
- Using an external storage provider such as object storage

Of these three options, using an object storage is the most robust. Object storage servers such as Amazon S3 and MinIO have become the de-facto hard drives for storing data in cloud native applications, machine learning, and many other areas of development. They are an ideal place to store binary and blog information when using containers because they are redundant, have high IO throughput, and can be accessed by many containers simultaneously (which facilitates high availability).

As mentioned earlier in the Apendix1, there is a file plugin called S3Contents that can be used to save Jupyter files to object storage providers which implement the Amazon S3 API. We installed the plugin as part of building the container image.

To have Jupyter use an object store, we need to inject a set of configuration parameters into the container at the time it starts. This is usually done through a file called jupyter_notebook_config.py saved in the user's Jupyter folder (./juputer/k8s/jupyter). The code listing below shows an example what the resulting configuration of S3Contents might look like for MinIO.

```
    from s3contents import S3ContentsManager
    from IPython.lib import passwd
    c = get_config()
    
    # Tell Jupyter to use S3ContentsManager for all storage.
    # Startup auth Token
    c.NotebookApp.password = passwd("jupyter")
    # S3 Object Storage Configuration
    c.NotebookApp.contents_manager_class = S3ContentsManager
    c.S3ContentsManager.access_key_id = "minio"
    c.S3ContentsManager.secret_access_key = "minio123"
    c.S3ContentsManager.endpoint_url = "http://minio-service.data.:9000"
    c.S3ContentsManager.bucket = "spark-jupyter"
    c.S3ContentsManager.prefix = "notebooks"
```

Injecting Configuration Values Using ConfigMaps

Kubernetes ConfigMaps can be used to store configuration information about a program in a central location. When a pod starts, this data can then be injected as environment variables or mounted as a file. This provides a convenient way of ensuring that configuration values - such as those we'll need to get the external storage in Jupyter working or the authentication token/password - are the same for every pod instance that starts.

ConfigMaps are independent objects in Kubernetes. They are created outside of pods, deployments, or stateful sets, and their data is associated by reference. After a ConfigMap is defined, it is straightforward to include the needed metadata in the pod manifest.

We will use a single ConfigMap to solve the first two problems we described above. The code listing below shows a ConfigMap which both configures an S3 contents manager for Jupyter and provides a known password to the application server at startup.

The setup of an object storage such as Amazon S3, MinIO, OpenStack Swift is beyond the scope of this article. For information about which parameters are needed by specific services for S3Contents, refer to the [README](https://github.com/danielfrg/s3contents/blob/master/README.md) file available in the project's [GitHub repository](https://github.com/danielfrg/s3contents).

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: jupyter-notebook-config
data:
  storage_path.py: |
    from s3contents import S3ContentsManager
    from IPython.lib import passwd
    c = get_config()
    # Startup auth Token
    c.NotebookApp.password = passwd("jupyter")
    # S3 Object Storage Configuration
    c.NotebookApp.contents_manager_class = S3ContentsManager
    c.S3ContentsManager.access_key_id = "minio"
    c.S3ContentsManager.secret_access_key = "minio123"
    c.S3ContentsManager.endpoint_url = "http://minio-service.data.:9000"
    c.S3ContentsManager.bucket = "spark-jupyter"
    c.S3ContentsManager.prefix = "notebooks"
```


The YAML below shows how to reference the ConfigMap as a volume for a pod. The manifest in the listing roughly recreates the `kubectl run` command used earlier with the additional configuration required to access the ConfigMap. From this point forward, the configuration of the Jupyter application has become complex enough that we will use manifests to show its structure.

The ConfigMap data will be mounted at` ~/.jupyter/jupyter_notebook_config.py`, the path required by Jupyter in order to leverage the contents manager. The `fsGroup` option is used under the `securityContext` so that it can be read by a member of the analytics group `(gid=777)`.

```
apiVersion: v1
kind: Pod
metadata:
  name: spark-jupyter
  labels:
    app: spark-jupyter

spec:
  serviceAccountName: spark-driver
  
  securityContext:
    fsGroup: 777
  
  containers:
  - name: jupyter-noteboob-pod
    image: davarski/spark301-k8s-minio-jupyter
    imagePullPolicy: Always
    command: ["jupyter", "lab", "--ip", "0.0.0.0"]
    volumeMounts:
    - name: storage-config-volume
      mountPath: /home/jovyan/.jupyter/jupyter_notebook_config.py
      subPath: storage_path.py
  
  volumes:
  - name: storage-config-volume
    configMap:
      name: jupyter-notebook-config
  
  restartPolicy: Always

```
To work correctly with Spark, the pod needs to be paired with a service in order for executors to spawn and communicate with the driver successfully. The code listing below shows what this service manifest would look like.
```
apiVersion: v1
kind: Service
metadata:
  name: spark-jupyter

spec:
  clusterIP: None
  selector:
    app: spark-jupyter
  ports:
  - protocol: TCP
    port: 8888
    targetPort: 8888
```



With the ConfigMap in place, you can launch a new pod/service and repeat the connection test from above. Upon stopping and re-starting the pod instance, you should notice that any notebooks or other files you add to the instance survive rather than disappear. Likewise, on authenticating to the pod, you will be prompted for a password rather than needing to supply a random token.

Enabling External Access

The final piece needed for our Spark-enabled Jupyter instance is external access. Again, while there are several options on how this might be configured such as a load-balanced service, perhaps the most robust is via a Kubernetes Ingress. Ingress allows for HTTP and HTTPS routes from outside the cluster to be forwarded to services inside the cluster.

It provides a host of benefits including:

- Externally reachable URLs
- Load-balanced traffic
- SSL/TLS termination
- Named based virtual hosting

While the specific configuration of these options is outside the scope of this article, providing Ingress using the NGINX/Traefik/etc. controller offers a far more robust way to access the Jupyter instance than `kubectl port-forward`.

The code listing below shows an example of a TLS-terminated Ingress controller that will forward to the pod/service created earlier. The TLS certificate is provisioned using cert-manager. For details on cert-manager, see the project's homepage.

```
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: jupyter-ingress-ingress
  annotations:
    cert-manager.io/cluster-issuer: selfsigned-issuer
spec:
  rules:
    - host: jupyter.data.davar.com
      http:
        paths:
          - backend:
              serviceName: spark-jupyter
              servicePort: 8888
            path: /
  tls:
    - hosts:
        - jupyter.data.davar.com
      secretName: jupyter-production-tls

```

Once it has been created and the certificates issued, the Jupyter instance should now be available outside the cluster at `https://jupyter.data.davar.com`.

To Infinity and Beyond

At this point, we have configured a Jupyter instance with a full complement of Data Science libraries able to launch Spark applications on top of Kubernetes. It is configured to read and write its data to an object storage, and integrate with a host of powerful visualization frameworks. We've tried to make it as "Cloud Native" as possible, and could be run on our server instance in a highly available configuration if desired.

That is a powerful set of tools begging to be used and ready to go!

# Apendix3: Spark ML with PySpark and Jupyter

TODO: We'll start to put these tools into action to understand how best to work with large structured data sets, train machine learning models, work with graph databases, and analyze streaming datasets.

Add MinIO(S3) jars:
```
cd ./jupyter-1.0.0/docker

# Add lines 

$ grep curl Dockerfile.k8s-minio.executor 
RUN curl https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.0.1/hadoop-aws-3.0.1.jar -o /opt/spark/jars/hadoop-aws-3.0.1.jar
RUN curl https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-core/1.11.923/aws-java-sdk-core-1.11.923.jar -o /opt/spark/jars/aws-java-sdk-core-1.11.923.jar
RUN curl https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.11.923/aws-java-sdk-1.11.923.jar -o /opt/spark/jars/java-sdk-1.11.923.jar
RUN curl https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-kms/1.11.923/aws-java-sdk-kms-1.11.923.jar -o /opt/spark/jars/aws-java-sdk-kms-1.11.923.jar
RUN curl https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-s3/1.11.923/aws-java-sdk-s3-1.11.923.jar -o /opt/spark/jars/aws-java-sdk-s3-1.11.923.jar

$ grep 1.0.0 *
Dockerfile.cluster-dask:FROM davarski/spark301-k8s-minio-polyglot:1.0.0
Dockerfile.cv:FROM davarski/spark301-minio-dask:1.0.0
Dockerfile.hub-jupyter:FROM davarski/spark301-k8s-minio-jupyter:1.0.0
Dockerfile.hub-polyglot:FROM davarski/spark301-k8s-minio-dl:latest:1.0.0
Dockerfile.itk:FROM davarski/spark301-minio-dask:1.0.0
Dockerfile.k8s-minio.deep-learning:FROM davarski/spark301-k8s-minio-kafka:1.0.0
Dockerfile.k8s-minio.driver:FROM davarski/spark301-k8s-minio-base:1.0.0
Dockerfile.k8s-minio.jupyter:FROM davarski/spark301-k8s-minio-driver:1.0.0
Dockerfile.k8s-minio.ml-executor:FROM davarski/spark301-k8s-minio-base:1.0.0

# Rebuild images with MinIO(S3)
docker login
# Build and tag the base/executor image
docker build -f ./Dockerfile.k8s-minio.executor -t davarski/spark301-k8s-minio-base:1.0.0 .
# Push the contaimer image to a public registry
docker push davarski/spark301-k8s-minio-base:1.0.0

# Build and tag the driver image
docker build -f ./Dockerfile.k8s-minio.driver -t davarski/spark301-k8s-minio-driver:1.0.0 .
# Push the contaimer image to a public registry
docker push davarski/spark301-k8s-minio-driver:1.0.0

# Build/tag/push the jupyter image
docker build -f ./Dockerfile.k8s-minio.jupyter -t davarski/spark301-k8s-minio-jupyter:1.0.0 .
docker push davarski/spark301-k8s-minio-jupyter:1.0.0
```
Pull images into k8s(k3s):
```
export KUBECONFIG=~/.kube/k3s-config-jupyter 
sudo k3s crictl pull davarski/spark301-k8s-minio-base:1.0.0
sudo k3s crictl pull davarski/spark301-k8s-minio-driver:1.0.0
sudo k3s crictl pull davarski/spark301-k8s-minio-jupyter:1.0.0

```
Delete old deploy:
```
kubectl delete -f jupyter-notebook.svc.yaml -f jupyter-notebook.ingress.yaml -f jupyter-notebook.pod.yaml

```

Fix yamls:
```
$ cd ./jupyter-1.0.0/k8s
$ grep 1.0.0 *.yaml
jupyter-notebook.pod.yaml:    image: davarski/spark301-k8s-minio-jupyter:1.0.0
jupyter-notebook.pod.yaml.DEMO:    image: davarski/spark301-k8s-minio-jupyter:1.0.0
jupyter-notebook.pod.yaml.MINIO-BUCKET:    image: davarski/spark301-k8s-minio-jupyter:1.0.0
```

Apply new yamls:

```
kubectl apply -f jupyter-notebook.pod.yaml -f jupyter-notebook.svc.yaml -f jupyter-notebook.ingress.yaml
```

Check libs:

```
$ kubectl exec -it spark-jupyter -- bash -c "ls /opt/spark/jars/*aws*"
/opt/spark/jars/aws-java-sdk-core-1.11.923.jar	/opt/spark/jars/aws-java-sdk-kms-1.11.923.jar  /opt/spark/jars/aws-java-sdk-s3-1.11.923.jar  /opt/spark/jars/hadoop-aws-3.0.1.jar
```

Check Spark MinIO integration:
```
sc.hadoopConfiguration.set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")
sc.hadoopConfiguration.set("fs.s3a.access.key", "minio")
sc.hadoopConfiguration.set("fs.s3a.secret.key", "minio123")
sc.hadoopConfiguration.set("fs.s3a.path.style.access", "true")
val speciesDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")
speciesDF.show(2)

```
Example Output:

```
$ export KUBECONFIG=~/.kube/k3s-config-jupyter
$ kubectl exec -ti spark-jupyter -- bash

jovyan@spark-jupyter:~/work$ spark-shell
WARNING: An illegal reflective access operation has occurred
WARNING: Illegal reflective access by org.apache.spark.unsafe.Platform (file:/opt/spark/jars/spark-unsafe_2.12-3.0.1.jar) to constructor java.nio.DirectByteBuffer(long,int)
WARNING: Please consider reporting this to the maintainers of org.apache.spark.unsafe.Platform
WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
WARNING: All illegal access operations will be denied in a future release
20/12/22 01:36:20 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://spark-jupyter:4040
Spark context available as 'sc' (master = local[*], app id = local-1608600986357).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 3.0.1
      /_/
         
Using Scala version 2.12.10 (OpenJDK 64-Bit Server VM, Java 11.0.9.1)
Type in expressions to have them evaluated.
Type :help for more information.

scala> sc.hadoopConfiguration.set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")

scala> sc.hadoopConfiguration.set("fs.s3a.access.key", "minio")

scala> sc.hadoopConfiguration.set("fs.s3a.secret.key", "minio123")

scala> sc.hadoopConfiguration.set("fs.s3a.path.style.access", "true")

scala> val speciesDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")
20/12/22 01:36:59 WARN ApacheUtils: NoSuchMethodException was thrown when disabling normalizeUri. This indicates you are using an old version (< 4.5.8) of Apache http client. It is recommended to use http client version >= 4.5.9 to avoid the breaking change introduced in apache client 4.5.7 and the latency in exception handling. See https://github.com/aws/aws-sdk-java/issues/1919 for more information
speciesDF: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 3 more fields]

scala> speciesDF.show(2)
+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 2 rows

```

Test Spark k8s integration:
```
import pyspark

conf = pyspark.SparkConf()

# Kubernetes is a Spark master in our setup. 
# It creates pods with Spark workers, orchestrates those 
# workers and returns final results to the Spark driver 
# (“k8s://https://” is NOT a typo, this is how Spark knows the “provider” type). 
conf.setMaster("k8s://https://kubernetes.default:443") 

# Worker pods are created from the base Spark docker image.
# If you use another image, specify its name instead.
conf.set(
    "spark.kubernetes.container.image", 
    "davarski/spark301-k8s-minio-base:1.0.0") 

# Authentication certificate and token (required to create worker pods):
conf.set(
    "spark.kubernetes.authenticate.caCertFile", 
    "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
conf.set(
    "spark.kubernetes.authenticate.oauthTokenFile", 
    "/var/run/secrets/kubernetes.io/serviceaccount/token")

# Service account which should be used for the driver
conf.set(
    "spark.kubernetes.authenticate.driver.serviceAccountName", 
    "spark-driver") 

# 2 pods/workers will be created. Can be expanded for larger workloads.
conf.set("spark.executor.instances", "2") 

# The DNS alias for the Spark driver. Required by executors to report status.
conf.set(
    "spark.driver.host", "spark-jupyter") 

# Port which the Spark shell should bind to and to which executors will report progress
conf.set("spark.driver.port", "29413") 

# Initialize spark context, create executors
sc = pyspark.SparkContext(conf=conf)

sc._conf.getAll()

!kubectl get pod

# Create a distributed data set to test to the session
t = sc.parallelize(range(10))

# Calculate the approximate sum of values in the dataset
r = t.sumApprox(3)
print('Approximate sum: %s' % r)
```


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-Jupyter-k8s.png" width="800">


Test MinIO(S3) integration inside Jupyter Notebook (Login and cell):

```
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('minio_test').getOrCreate()
spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "minio")
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "minio123")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")
spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")
spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
speciesDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")
speciesDF.show(2)
```
Example Output:
```
from pyspark.sql import SparkSession

spark=SparkSession.builder.appName('minio_test').getOrCreate()

spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "minio")

spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "minio123")

spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://minio-service.data.svc.cluster.local:9000")

spark._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")

spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")

spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

speciesDF = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("s3a://iris/iris.csv")

speciesDF.show(2)

+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 2 rows


```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-Jupyter-k8s-minio.png" width="800">



## Linear Regression

The dataset that we are going to use for this example is a dummy
dataset and contains a total of 1,232 rows and 6 columns. We have to
use 5 input variables to predict the target variable using the Linear
Regression model. 

Upload `./jupyter/dataset/Linear_regression_dataset.csv` into Jupyter env:

1: Create the SparkSession Object
We start the Jupyter Notebook and import SparkSession and create a new
SparkSession object to use Spark:
```
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('lin_reg').getOrCreate()
```

2: Read the Dataset
We then load and read the dataset within Spark using Dataframe. We have
to make sure we have opened the PySpark from the same directory folder
where the dataset is available or else we have to mention the directory path
of the data folder:
```
df=spark.read.csv('Linear_regression_dataset.csv', inferSchema=True,header=True)
```
3: Exploratory Data Analysis
In this section, we drill deeper into the dataset by viewing the dataset,
validating the shape of the dataset, various statistical measures, and
correlations among input and output variables. We start with checking the
shape of the dataset.

```
print((df.count(), len(df.columns)))
```
The above output confirms the size of our dataset, and we can validate the
datatypes of the input values to check if we need to do change/cast any columns
datatypes. In this example, all columns contain Integer or double values.

```
df.printSchema()
```
There is a total of six columns out of which five are input columns
( var_1 to var_5) and target column (output). We can now use describe
function to go over statistical measures of the dataset.

```
df.describe().show(3,False)
```
This allows us to get a sense of distribution, measure of center, and
spread for our dataset columns. We then take a sneak peek into the dataset
using the head function and pass the number of rows that we want to view.

```
df.head(3)
```
We can check the correlation between input variables and output
variables using the corr function:
```
from pyspark.sql.functions import corr
df.select(corr('var_1','output')).show()
```
4: Feature Engineering
This is the part where we create a single vector combining all input features
by using Spark’s VectorAssembler. It creates only a single feature that
captures the input values for that row. So, instead of five input columns, it
essentially merges all input columns into a single feature vector column.

```
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
```
One can select the number of columns that would be used as input
features and can pass only those columns through the VectorAssembler. In
our case, we will pass all the five input columns to create a single feature
vector column.
```
df.columns
vec_assmebler=VectorAssembler(inputCols=['var_1','var_2', 'var_3', 'var_4', 'var_5'],outputCol='features')
features_df=vec_assmebler.transform(df)
features_df.printSchema()
```
As, we can see, we have an additional column (‘features’) that contains
the single dense vector for all of the inputs.
```
features_df.select('features').show(5,False)
```
We take the subset of the dataframe and select only the features
column and the output column to build the Linear Regression model.
```

model_df=features_df.select('features','output')
model_df.show(5,False)
print((model_df.count(), len(model_df.columns)))

```

5: Splitting the Dataset
We have to split the dataset into a training and test dataset in order to train
and evaluate the performance of the Linear Regression model built. We
split it into a 70/30 ratio and train our model on 70% of the dataset. We can
print the shape of train and test data to validate the size.

```
train_df,test_df=model_df.randomSplit([0.7,0.3])
print((train_df.count(), len(train_df.columns)))
print((test_df.count(), len(test_df.columns)))
```
6: Build and Train Linear Regression Model
In this part, we build and train the Linear Regression model using features
of the input and output columns. We can fetch the coefficients (B1, B2,
B3, B4, B5) and intercept (B0) values of the model as well. We can also
evaluate the performance of model on training data as well using r2. This
model gives a very good accuracy (86%) on training datasets.

```
from pyspark.ml.regression import LinearRegression
lin_Reg=LinearRegression(labelCol='output')
lr_model=lin_Reg.fit(train_df)
print(lr_model.coefficients)
print(lr_model.intercept)
training_predictions=lr_model.evaluate(train_df)
print(training_predictions.r2)
```
7: Evaluate Linear Regression Model on Test Data
The final part of this entire exercise is to check the performance of the model
on unseen or test data. We use the evaluate function to make predictions for
the test data and can use r2 to check the accuracy of the model on test data.
The performance seems to be almost similar to that of training.
```
test_predictions=lr_model.evaluate(test_df)
print(test_predictions.r2)
print(test_predictions.meanSquaredError)
```
Example Output:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-ML-jupyter-linear-regression.png" width="800">

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo6-Spark-ML/jupyter/ipynb/Linear_Regression.ipynb

## Logistic Regression

The dataset that we are going to use for this example is a dummy dataset
and contains a total of 20,000 rows and 6 columns. We have to use 5 input
variables to predict the target class using the logistic regression model.
This dataset contains information regarding online users of a retail sports
merchandise website. The data captures the country of user, platform
used, age, repeat visitor or first-time visitor, and the number of web
pages viewed on the website. It also has the information if the customer
ultimately bought the product or not (conversion status).

Upload `./jupyter/dataset/Log_Reg_dataset.csv` into Jupyter env:

1: Create the Spark Session Object
We start the Jupyter Notebook and import SparkSession and create a new
SparkSession object to use Spark.

```
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('log_reg').getOrCreate()
```
2: Read the Dataset
We then load and read the dataset within Spark using Dataframe. We have
to make sure we have opened the PySpark from the same directory folder
where the dataset is available or else we have to mention the directory path
of the data folder.
```
df=spark.read.csv('Log_Reg_dataset.csv',inferSchema=True,header=True)
```
3: Exploratory Data Analysis
In this section, we drill deeper into the dataset by viewing the dataset
and validating the shape of the it and various statistical measures of the
variables. We start with checking the shape of the dataset:
```
print((df.count(), len(df.columns)))
```

So, the above output confirms the size of our dataset and we can then
validate the datatypes of the input values to check if we need to change/
cast any columns datatypes.
```
df.printSchema()
```
As we can see, there are two such columns (Country, Search_Engine),
which are categorical in nature and hence need to be converted into
numerical form. Let’s have a look at the dataset using the show function in
Spark.
```
df.show(5)
```
We can now use the describe function to go over statistical measures of
the dataset.
```
df.describe().show()
```

We can observe that the average age of visitors is close to 28 years, and
they view around 9 web pages during the website visit.
Let us explore individual columns to understand the data in deeper
details. The groupBy function used along with counts returns the
frequency of each of the categories in the data.
```
df.groupBy('Country').count().show()
```

So, the maximum number of visitors are from Indonesia, followed by
India:
```
df.groupBy('Platform').count().show()
```
The Yahoo search engine users are the highest in numbers.
```
 df.groupBy('Status').count().show()
```

We have an equal number of users who are converted and non-­
converted.
Let’s use the groupBy function along with the mean to know more
about the dataset.
```
df.groupBy('Country').mean().show()
```
We have the highest conversion rate from Malaysia, followed by India.
The average number of web page visits is highest in Malaysia and lowest in
Brazil.
```
df.groupBy('Platform').mean().show()
```
We have the highest conversion rate from user visitors use the Google
search engine.
```
df.groupBy('Status').mean().show()
```
We can clearly see there is a strong connection between the conversion
status and the number of pages viewed along with repeat visits.

4: Feature Engineering
This is the part where we convert the categorical variable into numerical
form and create a single vector combining all the input features by using
Spark’s VectorAssembler.
```
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
```
Since we are dealing with two categorical columns, we will have to
convert the country and search engine columns into numerical form. The
machine learning model cannot understand categorical values.
The first step is to label the column using StringIndexer into
numerical form. It allocates unique values to each of the categories of the
column. So, in the below example, all of the three values of search engine
(Yahoo, Google, Bing) are assigned values (0.0,1.0,2.0). This is visible in the
column named search_engine_num.
```
search_engine_indexer = StringIndexer(inputCol="Platform", outputCol="Platform_Num").fit(df)
df = search_engine_indexer.transform(df)
df.show(3,False)
df.groupBy('Platform').count().orderBy('count', ascending=False).show(5,False)
df.groupBy('Platform_Num').count().orderBy('count', ascending=False).show(5,False)
```
The next step is to represent each of these values into the form of a
one hot encoded vector. However, this vector is a little different in terms of
representation as it captures the values and position of the values in the vector.
```
from pyspark.ml.feature import OneHotEncoder
search_engine_encoder=OneHotEncoder(inputCol="Platform_Num", outputCol="Platform_Vector")
df = search_engine_encoder.transform(df)
df.show(3,False)
df.groupBy('Platform_Vector').count().orderBy('count', ascending=False).show(5,False)

```

The final feature that we would be using for building Logistic
Regression is Search_Engine_Vector. Let’s understand what these column
values represent.
```
(2,[0],[1.0]) represents a vector of length 2 , with 1 value :
Size of Vector – 2
Value contained in vector – 1.0
Position of 1.0 value in vector – 0 th place
```
This kind of representation allows the saving of computational space
and hence a faster time to compute. The length of the vector is equal to
one less than the total number of elements since each value can be easily
represented with just the help of two columns. For example, if we need to
represent Search Engine using one hot encoding, conventionally, we can
do it as represented below.
```
Platform Google Yahoo Bing
----------------------------------
Google 1 0 0
Yahoo 0 1 0
Bing 0 0 1
-----------------------------------
```
Another way of representing the above information in an optimized
way is just using two columns instead of three as shown below.
```
Platform Google Yahoo
-----------------------------------
Google 1 0
Yahoo 0 1
Bing 0 0
-----------------------------------
```
Let’s repeat the same procedure for the other categorical column
(Country).
```
country_indexer = StringIndexer(inputCol="Country", outputCol="Country_Num").fit(df)
df = country_indexer.transform(df)
df.groupBy('Country').count().orderBy('count',ascending=False).show(5,False)
df.groupBy('Country_Num').count().orderBy('count', ascending=False).show(5,False)
country_encoder = OneHotEncoder(inputCol="Country_Num", outputCol="Country_Vector")
df = country_encoder.transform(df)
df.select(['Country','Country_Num','Country_Vector']).show(3,False)
df.groupBy('Country_Vector').count().orderBy('count', ascending=False).show(5,False)
```
Now that we have converted both the categorical columns into
numerical forms, we need to assemble all of the input columns into a
single vector that would act as the input feature for the model.
So, we select the input columns that we need to use to create the single
feature vector and name the output vector as features.
```
df_assembler = VectorAssembler(inputCols=['Platform_Vector','Country_Vector','Age', 'Repeat_Visitor','Web_pages_viewed'], outputCol="features")
df = df_assembler.transform(df)
df.printSchema()
```

As we can see, now we have one extra column named features, which
is nothing but a combination of all the input features represented as a
single dense vector.

```
df.select(['features','Status']).show(10,False)
```
Let us select only features column as input and the Status column as
output for training the logistic regression model.
```
model_df=df.select(['features','Status'])
```
5: Splitting the Dataset
We have to split the dataset into a training and test dataset in order to train
and evaluate the performance of the logistic regression model. We split it
in a 75/25 ratio and train our model on 75% of the dataset. Another use of
splitting the data is that we can use 75% of the data to apply cross-­validation
in order to come up with the best Hyperparameters. Cross-­validation can be
of a different type where one part of the training data is kept for training and
the remaining part is used for validation purposes. K-fold cross-validation is
primarily used to train the model with the best Hyperparameters.

We can print the shape of train and test data to validate the size.
```
training_df,test_df=model_df.randomSplit([0.75,0.25])
print(training_df.count())
training_df.groupBy('Status').count().show()
```
This ensures we have a balance set of the target class (Status) into the
training and test set.
```
print(test_df.count())
test_df.groupBy('Status').count().show()
```
6: Build and Train Logistic Regression Model
In this part, we build and train the logistic regression model using features
as the input column and status as the output column.
```
from pyspark.ml.classification import LogisticRegression
log_reg=LogisticRegression(labelCol='Status').fit(training_df)
```
Training Results
We can access the predictions made by the model using the evaluate
function in Spark that executes all the steps in an optimized way. That
gives another Dataframe that contains four columns in total, including
prediction and probability. The prediction column signifies the class
label that the model has predicted for the given row and probability
column contains two probabilities (probability for negative class at 0th
index and probability for positive class at 1st index).
```
train_results=log_reg.evaluate(training_df).predictions
train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).select(['Status','prediction','probability']).show(10,False)
```

So, in the above results, probability at the 0th index is for Status = 0
and probability as 1st index is for Status =1.

7: Evaluate Linear Regression Model on Test Data
The final part of the entire exercise is to check the performance of the
model on unseen or test data. We again make use of the evaluate function
to make predictions on the test.
We assign the predictions DataFrame to results and results DataFrame
now contains five columns.
```
results=log_reg.evaluate(test_df).predictions
results.printSchema()

```
We can filter the columns that we want to see using the select keyword.
```
results.select(['Status','prediction']).show(10,False)
```
Since this is a classification problem, we will use a confusion matrix to
gauge the performance of the model.

Confusion Matrix
We will manually create the variables for true positives, true negatives,
false positives, and false negatives to understand them better rather than
using the direct inbuilt function.
```
tp = results[(results.Status == 1)].count()
tn = results[(results.Status == 0)].count()
fp = results[(results.Status == 1)].count()
fn = results[(results.Status == 0)].count()
```
Accuracy
Accuracy is the most basic metric for evaluating any classifier; however, this is not the right indicator of
the performance of the model due to dependency on the target class
balance.
```
accuracy=float((true_postives+true_negatives) /(results.count()))
print(accuracy)
```
The accuracy of the model that we have built is around 94%.

Recall
Recall rate shows how much of the positive class cases we are able to
predict correctly out of the total positive class observations.
```
recall = float(true_postives)/(true_postives + false_negatives)
print(recall)
````
The recall rate of the model is around 0.94.

Precision

Precision rate talks about the number of true positives predicted
correctly out of all the predicted positives observations:
```
precision = float(true_postives) / (true_postives + false_positives)
print(precision)
```
So, the recall rate and precision rate are also in the same range, which
is due to the fact that our target class was well balanced.

## Random Forests

The dataset that we are going to use for this example is an open source
data set with a few thousand rows and six columns. We have to use five
input variables to predict the target variable using the random forest
model.

Upload `./jupyter/dataset/affairs.csv` into Jupyter env:


1: Create the Spark Session Object
We start the Jupyter Notebook and import SparkSession and create a new
SparkSession object to use Spark.

```
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('random_forest').getOrCreate()
```
2: Read the Dataset
We then load and read the dataset within Spark using Dataframe. We have
to make sure we have opened the PySpark from the same directory folder
where the dataset is available or else we have to mention the directory path
of the data folder.
```
df=spark.read.csv('affairs.csv',inferSchema=True,header=True)
```

3: Exploratory Data Analysis
In this section, we drill deeper into the dataset by viewing the dataset and
validating the shape of the dataset and various statistical measures of the
variables. We start with checking the shape of the dataset.
```
print((df.count(), len(df.columns)))
```
So, the above output confirms the size of our dataset and we can then
validate the data types of the input values to check if we need to change/
cast any columns data types.
```
df.printSchema()
```
As we can see there are no categorical columns which need to be
converted into numerical form. Let’s have a look at the dataset using show
function in Spark:

```
df.show(5)
```
We can now use the describe function to go over statistical measures of
the dataset.
```
df.describe().select('summary','rate_marriage','age','yrs_married','children','religious').show()
```
We can observe that the average age of people is close to 29 years, and
they have been married for 9 years.
Let us explore individual columns to understand the data in deeper
detail. The groupBy function used along with counts returns us the
frequency of each of the categories in the data.

```
df.groupBy('affairs').count().show()
```

So, we have more than 33% of the people who are involved in some
sort of extramarital affair out of a total number of people.
```
df.groupBy('rate_marriage').count().show()
```
The majority of the people rate their marriage very high (4 or 5), and
the rest rate it on the lower side. Let’s drill down a little bit further to
understand if the marriage rating is related to the affair variable or not.
```
df.groupBy('rate_marriage','affairs').count().orderBy('rate_marriage','affairs','count',ascending=True).show()
```
Clearly, the figures indicate a high percentage of people having affairs
when rating their marriages low. This might prove to be a useful feature for
the prediction. We will explore other variables as well in a similar manner.
```
df.groupBy('religious','affairs').count().orderBy('religious','affairs','count',ascending=True).show()
```
We have a similar story from ratings on religious perspective as well
as the number of people who have rated lower on religious features and a
higher percentage of affair involvement.
```
df.groupBy('children','affairs').count().orderBy('children','affairs','count',ascending=True).show()
```

The above table does not clearly indicate any of the trends regarding
the relation between the number of children and chances of being
involved in an affair. Let us use the groupBy function along with the mean
to know more about the dataset.
```
df.groupBy('affairs').mean().show()
```
So, the people who have affairs rate their marriages low and a little on
the higher side from an age standpoint. They have also been married for a
higher number of years and are less religious.

4: Feature Engineering
This is the part where we create a single vector combining all input
features by using Spark’s VectorAssembler.
```
from pyspark.ml.feature import VectorAssembler
```
We need to assemble all of the input columns into a single vector
that would act as the input feature for the model. So,we select the input
columns that we need to use to create the single feature vector and name
the output vector as features.
```
df_assembler = VectorAssembler(inputCols=['rate_marriage', 'age', 'yrs_married', 'children','religious'], outputCol="features")
df = df_assembler.transform(df)
df.printSchema()
```
As we can see, now we have one extra column named features, which
is nothing but a combination of all the input features represented as a
single dense vector.
```
df.select(['features','affairs']).show(10,False)
```
Let us select only the features column as input and the affairs column
as output for training the random forest model.
```
model_df=df.select(['features','affairs'])
```
5: Splitting the Dataset
We have to split the dataset into training and test datasets in order to train
and evaluate the performance of the random forest model. We split it into
a 75/25 ratio and train our model on 75% of the dataset. We can print the
shape of the train and test data to validate the size.
```
train_df,test_df=model_df.randomSplit([0.75,0.25])
print(train_df.count())
train_df.groupBy('affairs').count().show()
```
This ensures we have balanced set values for the target class (‘affairs’)
into the training and test sets.
```
test_df.groupBy('affairs').count().show()
```

6: Build and Train Random Forest Model
In this part, we build and train the random forest model using features
such as input and Status as the output colum.
```
from pyspark.ml.classification import RandomForestClassifier
rf_classifier=RandomForestClassifier(labelCol='affairs',numTrees=50).fit(train_df)
```
There are many hyperparameters that can be set to tweak the
performance of the model, but we are chosing the deafault ones here
except for one that is the number of decision trees that we want to build.

7: Evaluation on Test Data
Once we have trained our model on the training dataset, we can evaluate
its performance on the test set.
```
rf_predictions=rf_classifier.transform(test_df)
rf_predictions.show()
```
The first column in the predictions table is that of input features of the
test data. The second column is the actual label or output of the test data.
The third column (rawPrediction) represents the measure of confidence
for both possible outputs. The fourth column is that of conditional
probability of each class label, and the final column is the prediction by the
random forest classifier.We can apply a groupBy function on the prediction
column to find out the number of predictions made for the positive and
negative classes.

```
rf_predictions.groupBy('prediction').count().show()
```

To evaluate these preditions, we will import the
classificationEvaluators.
```
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```
Accuracy
```
rf_accuracy=MulticlassClassificationEvaluator(labelCol='affairs',metricName='accuracy').evaluate(rf_predictions)
print('The accuracy of RF on test data is {0:.0%}'.format(rf_accuracy))
```
The accuracy of RF on test data is 73%

Precision
```
rf_precision=MulticlassClassificationEvaluator(labelCol='affairs',metricName='weightedPrecision').evaluate(rf_predictions)
print('The precision rate on test data is {0:.0%}'.format(rf_precision))
```
The precision rate on test data is 71%

AUC
```
rf_auc=BinaryClassificationEvaluator(labelCol='affairs').evaluate(rf_predictions)
print( rf_auc)

```
[Out]: 0.738

As mentioned in the earlier part, RF gives the importance of each
feature in terms of predictive power, and it is very useful to figure out the
critical variables that contribute the most to predictions.
```
rf_classifier.featureImportances
```

We used five features and the importance can be found out using the
feature importance function. To know which input feature is mapped to
which index values, we can use metadata information.
```
df.schema["features"].metadata["ml_attr"]["attrs"]
```
```
[Out]:
  {'idx': 0, 'name': 'rate_marriage'},
  {'idx': 1, 'name': 'age'},
  {'idx': 2, 'name': 'yrs_married'},
  {'idx': 3, 'name': 'children'},
  {'idx': 4, 'name': 'religious'}}
```  
So, rate_marriage is the most important feature from a prediction
standpoint followed by yrs_married. The least significant variable seems to
be Age.

8: Saving the Model
Sometimes, after training the model, we just need to call the model for
preditions, and hence it makes a lot of sense to persist the model object
and reuse it for predictions. There are two parts to this.
1. Save the ML model
2. Load the ML model
```
from pyspark.ml.classification import RandomForestClassificationModel
rf_classifier.save("/home/jovyan/work/RF_model")
```
This way we saved the model as object locally.The next
step is to load the model again for predictions
```
rf=RandomForestClassificationModel.load("/home/jovyan/work/RF_model")
new_preditions=rf.transform(new_df)
```
A new predictions table would contain the column with the model
predictions

Example Output:

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo6-Spark-ML/jupyter/ipynb/Random_Forests.ipynb

## Recommender Systems

The dataset that we are going to use for this demo is a subset from
a famous open sourced movie lens dataset and contains a total of 0.1
million records with three columns (User_Id,title,rating). We will train our
recommender model using 75% of the data and test it on the rest of the
25% user ratings.

Upload `./jupyter/dataset/movie_ratings_df.csv` into Jupyter env:

1: Create the SparkSession Object
We start the Jupyter Notebook and import SparkSession and create a new
SparkSession object to use Spark:
```
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('recommender').getOrCreate()
```

2: Read the Dataset
We then load and read the dataset within Spark using a dataframe. We
have to make sure we have opened the PySpark from the same directory
folder where the dataset is available or else we have to mention the
directory path of the data folder.
```
df=spark.read.csv('movie_ratings_df.csv',inferSchema=True,header=True)
```

3: Exploratory Data Analysis
In this section, we explore the dataset by viewing the dataset, validating
the shape of the dataset, and getting a count of the number of movies rated
and the number of movies that each user rated.
```
print((df.count(), len(df.columns)))
```
So, the above output confirms the size of our dataset and we can then
validate the datatypes of the input values to check if we need to change/
cast any columns’ datatypes.
```
df.printSchema()
```

There is a total of three columns out of which two are numerical and
the title is categorical. The critical thing with using PySpark for building
RS is that we need to have user_id and item_id in numerical form. Hence,
we will convert the movie title to numerical values later. We now view a
few rows of the dataframe using the rand function to shuffle the records in
random order.

```
#df.orderBy(rand()).show(10,False)
df.groupBy('userId').count().orderBy('count',ascending=False).show(10,False)
df.groupBy('userId').count().orderBy('count',ascending=True).show(10,False)
```

The user with the highest number of records has rated 737 movies, and
each user has rated at least 20 movies.
```
df.groupBy('title').count().orderBy('count',ascending=False).show(10,False)
```
The movie with highest number of ratings is Star Wars (1977) and has
been rated 583 times, and each movie has been rated by at least by 1 user.

4: Feature Engineering
We now convert the movie title column from categorical to numerical
values using StringIndexer. We import the stringIndexer and Indextostring
from the PySpark library.
```
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer,IndexToString
```
Next, we create the stringindexer object by mentioning the input
column and output column. Then we fit the object on the dataframe and
apply it on the movie title column to create new dataframe with numerical
values.
```
stringIndexer = StringIndexer(inputCol="title",outputCol="title_new")
model = stringIndexer.fit(df)
indexed = model.transform(df)
```
Let’s validate the numerical values of the title column by viewing few
rows of the new dataframe (indexed).
```
indexed.show(10)
```
As we can see, we now we have an additional column (title_new) with
numerical values representing the movie titles. We have to repeat the same
procedure in case the user_id is also a categorical type. Just to validate the
movie counts, we rerun the groupBy on a new dataframe.
```
indexed.groupBy('title_new').count().orderBy('count',ascending=False).show(10,False)
```

5: Splitting the Dataset
Now that we have prepared the data for building the recommender model,
we can split the dataset into training and test sets. We split it into a 75 to 25
ratio to train the model and test its accuracy.
```
train,test=indexed.randomSplit([0.75,0.25])
train.count()
test.count()
```
6: Build and Train Recommender Model
We import the ALS function from the PySpark ml library and build the
model on the training dataset. There are multiple hyperparameters
that can be tuned to improve the performance of the model. Two of the
important ones are nonnegative =‘True’ doesn’t create negative ratings in
recommendations and coldStartStrategy=‘drop’ to prevent any NaN ratings
predictions.
```
from pyspark.ml.recommendation import ALS
rec=ALS(maxIter=10,regParam=0.01,userCol='userId',itemCol='title_new',ratingCol='rating',nonnegative=True,coldStartStrategy="drop")
rec_model=rec.fit(train)
```
7: Predictions and Evaluation on Test Data
The final part of the entire exercise is to check the performance of the
model on unseen or test data. We use the transform function to make
predictions on the test data and RegressionEvaluate to check the RMSE
value of the model on test data.
```
predicted_ratings=rec_model.transform(test)
predicted_ratings.printSchema()
predicted_ratings.orderBy(rand()).show(10)
from pyspark.ml.evaluation import RegressionEvaluator
evaluator=RegressionEvaluator(metricName='rmse',predictionCol='prediction',labelCol='rating')
rmse=evaluator.evaluate(predicted_ratings)
print(rmse)
```
The RMSE is not very high; we are making an error of one point in the
actual rating and predicted rating. This can be improved further by tuning
the model parameters and using the hybrid approach.

8: Recommend Top Movies That Active User Might Like
After checking the performance of the model and tuning the hyperparameters,
we can move ahead to recommend top movies to users that they have not
seen and might like. The first step is to create a list of unique movies in the
dataframe.
```
unique_movies=indexed.select('title_new').distinct()
unique_movies.count()

```
So, we have in total 1,664 distinct movies in the dataframe.
```
a = unique_movies.alias('a')
```
We can select any user within the dataset for which we need to
recommend other movies. In our case, we go ahead with userId = 85.
```
user_id=85
```

We will filter the movies that this active user has already rated or seen.
```
watched_movies=indexed.filter(indexed['userId'] == user_id).select('title_new').distinct()
watched_movies.count()
b=watched_movies.alias('b')
```
So, there are total of 287 unique movies out of 1,664 movies that this
active user has already rated. So, we would want to recommend movies
from the remaining 1,377 items. We now combine both the tables to find
the movies that we can recommend by filtering null values from the joined
table.
```
total_movies = a.join(b, a.title_new == b.title_new,how='left')
total_movies.show(10,False)
remaining_movies=total_movies.where(col("b.title_new").isNull()).select(a.title_new).distinct()
remaining_movies.count()
remaining_movies=remaining_movies.withColumn("userId",lit(int(user_id)))
remaining_movies.show(10,False)

```
Finally, we can now make the predictions on this remaining movie’s
dataset for the active user using the recommender model that we built
earlier. We filter only a few top recommendations that have the highest
predicted ratings.
```
recommendations=rec_model.transform(remaining_movies).orderBy('prediction',ascending=False)
recommendations.show(5,False)
```

So, movie titles 1433 and 1322 have the highest predicted rating for this
active user (85). We can make it more intuitive by adding the movie title
back to the recommendations. We use Indextostring function to create an
additional column that returns the movie title.
```
movie_title = IndexToString(inputCol="title_new",outputCol="title",labels=model.labels)
final_recommendations=movie_title.transform(recommendations)
final_recommendations.show(10,False)
```

So, the recommendations for the userId (85) are Boys, Les (1997)
and Faust (1994). This can be nicely wrapped in a single function that
executes the above steps in sequence and generates recommendations for
active users. The complete code is available on the GitHub repo with this
function built in.

Example Output: 
https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo6-Spark-ML/jupyter/ipynb/Recommender_Systems.ipynb

## Clustering

The dataset that we are going to use for this demo is the famous open
sourced IRIS dataset and contains a total of 150 records with 5 columns
(sepal length, sepal width, petal length, petal width, species). There are
50 records for each type of species. We will try to group these into clusters
without using the species label information.

Upload `./jupyter/dataset/iris_dataset.csv` into Jupyter env:

1: Create the SparkSession Object
We start Jupyter Notebook and import SparkSession and create a new
SparkSession object to use Spark:
```
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('K_means').getOrCreate()
```
2: Read the Dataset
We then load and read the dataset within Spark using a dataframe. We
have to make sure we have opened PySpark from the same directory folder
where the dataset is available or else we have to mention the directory path
of the data folder.
```
df=spark.read.csv('iris_dataset.csv',inferSchema=True,header=True)
```
3: Exploratory Data Analysis
In this section, we explore the dataset by viewing it and validating its
shape.
```
print((df.count(), len(df.columns)))
```
So, the above output confirms the size of our dataset and we can then
validate the datatypes of the input values to check if we need to change/
cast any columns' datatypes.
```
df.printSchema()
```
There is a total of five columns out of which four are numerical and the
label column is categorical.
```
from pyspark.sql.functions import rand
df.orderBy(rand()).show(10,False)
df.groupBy('species').count().orderBy('count').show(10,False)
```
So, it confirms that there are an equal number of records for each
species available in the dataset

4: Feature Engineering
This is the part where we create a single vector combining all input
features by using Spark’s VectorAssembler. It creates only a single
feature that captures the input values for that particular row. So,
instead of four input columns (we are not considering a label column
since it's an unsupervised machine learning technique), it essentially
translates it into a single column with four input values in the form
of a list.
```
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
input_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
vec_assembler = VectorAssembler(inputCols = input_cols,outputCol='features')
final_data = vec_assembler.transform(df)
```
5: Build K-Means Clustering Model
The final data contains the input vector that can be used to run K-means
clustering. Since we need to declare the value of ‘K’ in advance before
using K-means, we can use elbow method to figure out the right value
of ‘K’. In order to use the elbow method, we run K-means clustering for
different values of ‘K’. First, we import K-means from the PySpark library
and create an empty list that would capture the variability or SSE (within
cluster distance) for each value of K.
```
from pyspark.ml.clustering import KMeans
errors=[]
for k in range(2,10):
    kmeans = KMeans(featuresCol='features',k=k)
    model = kmeans.fit(final_data)
    intra_distance = model.computeCost(final_data)
    errors.append(intra_distance)
```  

computeCost is deprecated and removed in 3.0.0. It causes the failure in https://github.com/rstudio/sparklyr/blob/master/tests/testthat/test-ml-clustering-kmeans-ext.R#L87

Note The ‘K’ should have a minimum value of 2 to be able to build
clusters.

Now, we can plot the intracluster distance with the number of clusters
using numpy and matplotlib.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cluster_number = range(2,10)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE')
plt.scatter(cluster_number,errors)
plt.show()
```
In this case, k=3 seems to be the best number of clusters as we can see
a sort of elbow formation between three and four values. We build final
clusters using k=3.
```
kmeans = KMeans(featuresCol='features',k=3)
model = kmeans.fit(final_data)
model.transform(final_data).groupBy('prediction').count().show()
```
K-Means clustering gives us three different clusters based on the IRIS
data set. We certainly are making a few of the allocations wrong as only
one category has 50 records in the group, and the rest of the categories
are mixed up. We can use the transform function to assign the cluster
number to the original dataset and use a groupBy function to validate the
groupings.
```
predictions=model.transform(final_data)
predictions.groupBy('species','prediction').count().show()
```
As it can be observed, the setosa species is perfectly grouped
along with versicolor, almost being captured in the same cluster,
but verginica seems to fall within two different groups. K-means can
produce different results every time as it chooses the starting point
(centroid) randomly every time. Hence, the results that you might get
in you K-means clustering might be totally different from these results
unless we use a seed to reproduce the results. The seed ensures the
split and the initial centroid values remain consistent throughout the
analysis.

6: Visualization of Clusters
In the final step, we can visualize the new clusters with the help of Python’s
matplotlib library. In order to do that, we convert our Spark dataframe into
a Pandas dataframe first.
```
pandas_df = predictions.toPandas()
pandas_df.head()
```
We import the required libraries to plot the third visualization and
observe the clusters.
```
from mpl_toolkits.mplot3d import Axes3D
cluster_vis = plt.figure(figsize=(12,10)).gca(projection='3d')
cluster_vis.scatter(pandas_df.sepal_length, pandas_df.sepal_width, pandas_df.petal_length, c=pandas_df.prediction,depthshade=False)
plt.show()
```
Example Output: https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo6-Spark-ML/jupyter/ipynb/Clustering.ipynb

## Natural Language Processing

# Appendix 4: Managing, Deploying, and Scaling Machine Learning Pipelines with Apache Spark (MLOps)

End to End MLOps examples: Build machine learning pipelines with MLlib. Manage and deploy the models we train. Utilize MLflow to track, reproduce, and
deploy our MLlib models using various model deployment scenarios, and architect scalable machine learning solutions.


MLflow is an open source platform that helps developers reproduce and share experiments, manage models, and much more. It provides interfaces in Python, R, and Java/Scala, as well as a REST API. MLflow has four main components:

- Tracking
Provides APIs to record parameters, metrics, code versions, models, and artifacts
such as plots, and text.
- Projects
A standardized format to package your data science projects and their dependen‐
cies to run on other platforms. It helps you manage the model training process.
- Models
A standardized format to package models to deploy to diverse execution environ‐
ments. It provides a consistent API for loading and applying models, regardless
of the algorithm or library used to build the model.
- Registry
A repository to keep track of model lineage, model versions, stage transitions,
and annotations.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-MLOps-MLFlow-components.png" width="800">

The MLflow tracking server can host many experiments. You can log to the tracking server using a notebook, local app, or cloud job, as shown

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-MLOps-MLFlow-tracking-server.png" width="800">

Let’s examine a few things that can be logged to the tracking server:

- Parameters
Key/value inputs to your code—e.g., hyperparameters like num_trees ormax_depth in your random forest
- Metrics
Numeric values (can update over time)—e.g., RMSE or accuracy values
- Artifacts
Files, data, and models—e.g., matplotlib images, or Parquet files Metadata Information about the run, such as the source code that executed the run or the
version of the code (e.g., the Git commit hash string for the code version)
- Models
The model(s) you trained


Build new MLFlow docker image if needed and pushing it into DockerHub container registry.
```
$ cd mlflow
$ docker build -t davarski/mlflow:1.8.0-v4 .
$ docker push davarski/mlflow:1.8.0-v4
```

Deploy MLFlow on k8s and create MiniIO bucket
```
export KUBECONFIG=~/.kube/k3s-config-jupyter
sudo k3s crictl pull davarski/mlflow:1.8.0-v4
kubectl create -f ../003-data/800-mlflow/60-ingress.yml -f ../003-data/800-mlflow/50-service.yml -f ../003-data/800-mlflow/40-statefulset.yml
mc rb minio-cluster/mlflow --force
mc mb minio-cluster/mlflow 
mc ls minio-cluster/mlflow
```

Upload ./mlflow/data/sf-airbnb-clean.parquet into  jupyter ./data folder (the same path as jupyter notebook)


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-MLOps-airbnb-dataset.png" width="800">

Cells:
```
!pip install mlflow==1.8.0
# Restart the kernel
```
```
import os
# api and object access
os.environ['MLFLOW_TRACKING_URI'] = "http://mlflow.data:5000"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio-service.data:9000"
# minio credentials
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"

```

```
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark
import pandas as pd

spark = SparkSession.builder.appName("airbnb").getOrCreate()
```

```
def mlflow_rf(file_path, num_trees, max_depth):
  with mlflow.start_run(run_name="random-forest") as run:
    # Create train/test split
    spark = SparkSession.builder.appName("App").getOrCreate()
    airbnbDF = spark.read.parquet("./data/")
    (trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)

    # Prepare the StringIndexer and VectorAssembler
    categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
    indexOutputCols = [x + "Index" for x in categoricalCols]

    stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

    numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price"))]
    assemblerInputs = indexOutputCols + numericCols
    vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    
    # Log params: Num Trees and Max Depth
    mlflow.log_param("num_trees", num_trees)
    mlflow.log_param("max_depth", max_depth)

    rf = RandomForestRegressor(labelCol="price",
                               maxBins=40,
                               maxDepth=max_depth,
                               numTrees=num_trees,
                               seed=42)

    pipeline = Pipeline(stages=[stringIndexer, vecAssembler, rf])

    # Log model
    pipelineModel = pipeline.fit(trainDF)
    mlflow.spark.log_model(pipelineModel, "model")

    # Log metrics: RMSE and R2
    predDF = pipelineModel.transform(testDF)
    regressionEvaluator = RegressionEvaluator(predictionCol="prediction",
                                            labelCol="price")
    rmse = regressionEvaluator.setMetricName("rmse").evaluate(predDF)
    r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
    mlflow.log_metrics({"rmse": rmse, "r2": r2})

    # Log artifact: Feature Importance Scores
    rfModel = pipelineModel.stages[-1]
    pandasDF = (pd.DataFrame(list(zip(vecAssembler.getInputCols(),
                                    rfModel.featureImportances)),
                          columns=["feature", "importance"])
              .sort_values(by="importance", ascending=False))
    # First write to local filesystem, then tell MLflow where to find that file
    pandasDF.to_csv("/tmp/feature-importance.csv", index=False)
    os.makedirs("data", exist_ok=True)
    mlflow.log_artifact("data", artifact_path="airbnb.ipynb")
```

Run experiment 1 (about 10 minutes to FINISH)

```
if __name__ == "__main__":
  mlflow_rf("./data",3,3)
```

Check MinIO bucket:

```
$ mc ls minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/
[2020-12-23 11:46:00 EET]     0B model/
$ mc ls minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/model/
[2020-12-23 11:44:58 EET]   293B MLmodel
[2020-12-23 11:44:58 EET]   107B conda.yaml
[2020-12-23 11:46:06 EET]     0B sparkml/
$ mc cat minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/model/conda.yaml
channels:
- defaults
dependencies:
- python=3.8.5
- pyspark=3.0.1
- pip
- pip:
  - mlflow
name: mlflow-env
$ mc cat minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/model/MLmodel
artifact_path: model
flavors:
  python_function:
    data: sparkml
    env: conda.yaml
    loader_module: mlflow.spark
    python_version: 3.8.5
  spark:
    model_data: sparkml
    pyspark_version: 3.0.1
run_id: f547a49a51df403ba21766c53c6eca6b
utc_time_created: '2020-12-23 09:44:56.119994'
$ mc ls minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/model/sparkml/
[2020-12-23 11:47:17 EET]     0B stages/
$ mc ls minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/model/sparkml/stages
[2020-12-23 11:47:25 EET]     0B stages/
$ mc ls minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/model/sparkml/stages/
[2020-12-23 11:47:31 EET]     0B 2_RandomForestRegressor_93839c6d60c6/
$ mc ls minio-cluster/mlflow/artifacts/0/f547a49a51df403ba21766c53c6eca6b/artifacts/model/sparkml/stages/2_RandomForestRegressor_93839c6d60c6/
[2020-12-23 11:47:47 EET]     0B data/
[2020-12-23 11:47:47 EET]     0B metadata/
[2020-12-23 11:47:47 EET]     0B treesMetadata/

```
Run experiment 2:

```
if __name__ == "__main__":
  mlflow_rf("./data",4,5) 
```

Example Ouptut:

https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/jupyter-v1.0.0/ipynb/spark-airbnb-random-forest-mlflow.ipynb

Check MinIO bucket:
```
$ mc ls minio-cluster/mlflow/artifacts/0/
[2020-12-23 12:05:29 EET]     0B 5b6431f76cfd4b8e990258d72aa6becf/
[2020-12-23 12:05:29 EET]     0B ed637f990f9b4e38bc7dc15d4f6839bb/
[2020-12-23 12:05:29 EET]     0B f547a49a51df403ba21766c53c6eca6b/

$ mc cat minio-cluster/mlflow/artifacts/0/ed637f990f9b4e38bc7dc15d4f6839bb/artifacts/model/MLmodel
artifact_path: model
flavors:
  python_function:
    data: sparkml
    env: conda.yaml
    loader_module: mlflow.spark
    python_version: 3.8.5
  spark:
    model_data: sparkml
    pyspark_version: 3.0.1
run_id: ed637f990f9b4e38bc7dc15d4f6839bb
utc_time_created: '2020-12-23 09:54:09.674812'
```


Check MLFlow UI http://mlflow.data.davar.com

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-MLOps-airbnb-experiments.png" width="800">


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-MLOps-airbnb-experiment-FINISHED.png" width="800">

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-MLOps-airbnb-experiment-UNFINISHED.png" width="800">

Register Model :

You can use the [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) to keep track of the models you are
using and control how they are transitioned to/from staging, production, and archived. You can see a screenshot of the Model Registry 

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo6-Spark-ML/pictures/Spark-MLOps-registered-model.png" width="800">

You can use the Model Registry with the other deployment options too.

Seldon Core deploy (Ref: https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo3-AutoML-MLFlow-SeldonCore)




