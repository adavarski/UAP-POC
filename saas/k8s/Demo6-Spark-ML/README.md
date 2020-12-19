
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
## Deployment modes

An attractive feature of Spark is its support for myriad deployment modes, enabling Spark to run in different configurations and environments. Because the cluster manager is agnostic to where it runs (as long as it can manage Spark’s executors and fulfill resource requests), Spark can be deployed in some of the most popular environments, such as Apache Hadoop YARN and Kubernetes, and can operate in different modes. 

Summarizes the available deployment modes.

<img src="https://github.com/adavarski/DataScience-DataOps_MLOps-Playground/blob/main/k8s/Demo6-Spark-ML/pictures/k8s-spark-cheatsheet-spark-deployment-models.png" width="800">


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



