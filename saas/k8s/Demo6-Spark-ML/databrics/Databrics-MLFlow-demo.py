# Databricks notebook source
from pyspark.ml.regression import LinearRegression
import mlflow
for i in range(1,4):
  with mlflow.start_run():
    mi = 10 * i
    rp = 0.1 * i
    enp = 0.5
    mlflow.log_param('maxIter',mi)
    mlflow.log_param('regParam',rp)
    mlflow.log_param('elasticNetParam',enp)
    lr = LinearRegression(maxIter=mi
        ,regParam=rp
        ,elasticNetParam=enp
        ,labelCol="target")
    model = lr.fit(traindata)
    pred = model.transform(testdata)
    r = pred.stat.corr("prediction", "target")
    mlflow.log_metric("rsquared", r**2, step=i)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
va = VectorAssembler(inputCols = basedata.feature_names
        ,outputCol = 'features')
testdata = va.transform(testdata)['features','target']
traindata = va.transform(traindata)['features','target']

# COMMAND ----------

import pandas as pd
import sklearn
from sklearn.datasets import load_boston
basedata = load_boston()
pddf = pd.DataFrame(basedata.data
        ,columns=basedata.feature_names)
pddf['target'] = pd.Series(basedata.target)
pct20 = int(pddf.shape[0]*.2)
testdata = spark.createDataFrame(pddf[:pct20])
traindata = spark.createDataFrame(pddf[pct20:])

# COMMAND ----------

dbutils.library.installPyPI('mlflow')
dbutils.library.installPyPI('scikit-learn')
dbutils.library.restartPython()
