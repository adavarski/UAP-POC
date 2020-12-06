Continuous Delivery for Machine Learning:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo3-AutoML-MLFlof-SeldonCore/pictures/cd4ml-end-to-end.png" width="800">

Model Building/Deploiyment High Level:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo3-AutoML-MLFlof-SeldonCore/pictures/ml-silos.png" width="800">

## Model Development

Example: 
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo3-AutoML-MLFlof-SeldonCore/pictures/overview.png" width="800">


### Build new MLFlow docker image if needed

```
$ cd mlflow
$ docker build -t davarski/mlflow:1.8.0-v4 .
$ docker push davarski/mlflow:1.8.0-v4
```
Edit MLflow StatefulSet ---> image: davarski/mlflow:1.8.0-v4

Redeploy:
```
kubectl delete -f 60-ingress.yml -f 50-service.yml -f 40-statefulset.yml
kubectl create -f 60-ingress.yml -f 50-service.yml -f 40-statefulset.yml

```

### Training/Tracking Models

MLflow API endpoint for tracking model development at http://mlflow.data:5000.

Jupyter environment: 

Note1: The following exercise is an adaptation of an official MLflow tutorial (https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.ipynb; https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html) using the scikit-learn ElasticNet (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) linear regression model using the Wine Quality Data Set.

Note2: The DataSet used here is Wine Quality Data set from UCI Machine Learning Repository. The csv file needed "winequality-red.csv" is attached in the repository. The same can also be found here https://archive.ics.uci.edu/ml/datasets/Wine+Quality
```
$ wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv' -qO - |head -n5
"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5
7.8;0.76;0.04;2.3;0.092;15;54;0.997;3.26;0.65;9.8;5
11.2;0.28;0.56;1.9;0.075;17;60;0.998;3.16;0.58;9.8;6

```

Jupyter Notebooks are a browser-based (or web-based) IDE (integrated development environments)

Build custom JupyterLab docker image
```
$ cd ./jupyterlab
$ docker build -t jupyterlab-eth .
$ docker tag jupyterlab-eth:latest davarski/jupyterlab-eth:latest
$ docker login 
$ docker push davarski/jupyterlab-eth:latest
```
Run Jupyter Notebook inside k8s:

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


First, install the following packages:
```
!pip install mlflow==1.8.0
!pip install scikit-learn==0.23.1
!pip install boto3==1.10.35
```
Setup environment variables: MLFLOW_TRACKING_URI to access the API server; MLFLOW_S3_ENDPOINT_URL to upload models and artifacts; and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY containing credentials to the MLFLOW_S3_ENDPOINT_URL.
```
import os
# api and object access
os.environ['MLFLOW_TRACKING_URI'] = "http://mlflow.data:5000"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio-service:9000"
# minio credentials
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
```
Import required packages and set a seed for NumPy random toaid in reproducing results:

```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
np.random.seed(70)
```
Create a function for evaluating model performance:

```
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
```
Download and split data into training and test sets:

```
csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(csv_url, sep=';')
train, test = train_test_split(data)
```
Prepare test and training sets by separating the quality column:
```
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]
```
Create a new MLflow experiment if one does not exist:
```
experiment_name = 'SkLearnWineQuality'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment == None:
    mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment.name)
```
Train the model, logging metrics, and parameters to MLflow,
along with trained model and source code:

```
alpha = 1
l1_ratio = 1
with mlflow.start_run() as run:
    mlflow.set_tags({
        "mlflow.user": "davar",
        "mlflow.source.type": "NOTEBOOK",
        "mlflow.source.name": "SkLearnWineQuality",
    })
    lr = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42
    )
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(
        test_y, predicted_qualities)
    print("Elasticnet model (alpha=%f, l1_ratio=%f):"
          % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    os.makedirs("data", exist_ok=True)
    mlflow.log_artifact("data", artifact_path="SkLearnWineQuality.ipynb")
    mlflow.sklearn.log_model(lr, "model",
                             registered_model_name="SkLearnWineModel")
mlflow.end_run()
```
Example ouput:
```
Elasticnet model (alpha=5.000000, l1_ratio=5.000000):
  RMSE: 0.7740183215265226
  MAE: 0.6624353628023353
  R2: -0.0020143202186577724
```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo3-AutoML-MLFlof-SeldonCore/pictures/jupyterlab-python-mlflow-notebook.png" width="800">


Each run of the previous code results in a new entry into the SkLearnWineQuality experiment. Browse to https://mlflow.data.davar.com and navigate to the experiment. From there, observe the various runs and their results.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo3-AutoML-MLFlof-SeldonCore/pictures/mlflow-ui-experiments.png" width="800">

Click a run entry to view details along with artifacts associated with the run, including, in this case, a model package and source code 

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo3-AutoML-MLFlof-SeldonCore/pictures/mlflow-ui-packaged-model.png" width="800">


Check MinIO bucket:
```

$ mc ls minio-cluster//mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model/
[2020-12-03 18:48:15 EET]   343B MLmodel
[2020-12-03 18:48:15 EET]   136B conda.yaml
[2020-12-03 18:48:15 EET]   633B model.pkl

$ mc cat minio-cluster//mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model/MLmodel
artifact_path: model
flavors:
  python_function:
    data: model.pkl
    env: conda.yaml
    loader_module: mlflow.sklearn
    python_version: 3.7.3
  sklearn:
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 0.23.1
run_id: e22b3108e7b04c269d65b3f081f44166
utc_time_created: '2020-12-03 16:48:15.286116'

$ mc cat minio-cluster//mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model/conda.yaml
channels:
- defaults
dependencies:
- python=3.7.3
- scikit-learn=0.23.1
- pip
- pip:
  - mlflow
  - cloudpickle==1.6.0
name: mlflow-env

```

.
MLflow brings essential Machine Learning components, further closing the gap between raw data and machine learning–based artificial intelligence. At this point, the k8s cluster supports the gathering of raw data The final step in Machine Learning development is production deployment, covered bellow.


## Deploy Artificial Intelligence
The method of deployment for machine learning models often depends on the problem domain, business requirements, and existing infrastructure.
However, a few projects have gained significant traction in moving toward standardization, specifically the open source project Seldon Core.

### Seldon Core

Deployment of Machine Learning–based models with Seldon Core. Seldon Core is an open source model deployment controller for Kubernetes. Seldon Core integrates well with established model packing standards, offering prebuilt inference servers, including supporting MLflow, scikit-learn, TensorFlow, and XGBoost, and provides an interface for building custom inference servers. This section uses only a small set of Seldon Core’s features needed to deploy the simple machine learning model built in the previous section.

Create a Namespace for Seldon Core:
```
$ kubectl create namespace seldon-system
```
Install Seldon Core Operator with Helm:
```
$ helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --namespace seldon-system
```
```
$ kubectl apply -f 000-sd-s3-secret.yml
```
Change the modelUri: value to the location of the MLflow model configuration. The additional componentSpecs: are optional and configured with more extended wait periods for the readiness and liveness probes to better account for the resource constrained k8s demo cluster.

100-sd-quality.yml ---> modelUri: s3://mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model


Apply the SeldonDeployment and Ingress:
```
$ kubectl apply -f 100-sd-quality.yml
```
It may take several minutes to deploy the model. Monitor the newly generated Pod in the default namespace for status; 
```
$ kubectl get po
NAME                                         READY   STATUS    RESTARTS   AGE
busybox                                      1/1     Running   154        13d
dnsutils                                     1/1     Running   150        13d
quality-default-0-quality-6d4664bd99-pl28n   2/2     Running   0          3m39s
$ kubectl logs quality-default-0-quality-6d4664bd99-pl28n -c seldon-container-engine
$ kubectl logs quality-default-0-quality-6d4664bd99-pl28n -c quality-model-initializer
[I 201204 11:27:40 initializer-entrypoint:13] Initializing, args: src_uri [s3://mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model] dest_path[ [/mnt/models]
[I 201204 11:27:40 storage:35] Copying contents of s3://mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model to local
[I 201204 11:27:40 storage:60] Successfully copied s3://mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model to /mnt/models
$ kubectl logs quality-default-0-quality-6d4664bd99-pl28n -c quality
Executing before-run script
---> Creating environment with Conda...
INFO:root:Copying contents of /mnt/models to local
INFO:root:Reading MLmodel file
INFO:root:Creating Conda environment 'mlflow' from conda.yaml
Collecting package metadata (repodata.json): ...working... done
Solving environment: ...working... done

Downloading and Extracting Packages
intel-openmp-2020.2  | 786 KB    | ########## | 100% 
scipy-1.5.2          | 14.3 MB   | ########## | 100% 
_libgcc_mutex-0.1    | 3 KB      | ########## | 100% 
mkl_random-1.1.1     | 322 KB    | ########## | 100% 
mkl-2020.2           | 138.3 MB  | ########## | 100% 
python-3.7.3         | 32.1 MB   | ########## | 100% 
tk-8.6.10            | 3.0 MB    | ########## | 100% 
sqlite-3.33.0        | 1.1 MB    | ########## | 100% 
blas-1.0             | 6 KB      | ########## | 100% 
pip-20.3             | 1.7 MB    | ########## | 100% 
mkl_fft-1.2.0        | 148 KB    | ########## | 100% 
libgfortran-ng-7.3.0 | 1006 KB   | ########## | 100% 
six-1.15.0           | 27 KB     | ########## | 100% 
scikit-learn-0.23.1  | 5.0 MB    | ########## | 100% 
threadpoolctl-2.1.0  | 17 KB     | ########## | 100% 
libffi-3.2.1         | 48 KB     | ########## | 100% 
numpy-1.19.2         | 22 KB     | ########## | 100% 
joblib-0.17.0        | 206 KB    | ########## | 100% 
wheel-0.36.0         | 32 KB     | ########## | 100% 
ncurses-6.2          | 817 KB    | ########## | 100% 
setuptools-50.3.2    | 723 KB    | ########## | 100% 
xz-5.2.5             | 341 KB    | ########## | 100% 
mkl-service-2.3.0    | 52 KB     | ########## | 100% 
libedit-3.1.20191231 | 116 KB    | ########## | 100% 
numpy-base-1.19.2    | 4.1 MB    | ########## | 100% 
Preparing transaction: ...working... done
Verifying transaction: ...working... done
Executing transaction: ...working... done
Installing pip dependencies: ...working... Ran pip subprocess with arguments:
['/opt/conda/envs/mlflow/bin/python', '-m', 'pip', 'install', '-U', '-r', '/tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt']
Pip subprocess output:
Collecting cloudpickle==1.6.0
  Downloading cloudpickle-1.6.0-py3-none-any.whl (23 kB)
Collecting mlflow
  Downloading mlflow-1.12.1-py3-none-any.whl (13.9 MB)
Requirement already satisfied: numpy in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.19.2)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting alembic<=1.4.1
  Downloading alembic-1.4.1.tar.gz (1.1 MB)
Collecting azure-storage-blob
  Downloading azure_storage_blob-12.6.0-py2.py3-none-any.whl (328 kB)
Collecting azure-core<2.0.0,>=1.9.0
  Downloading azure_core-1.9.0-py2.py3-none-any.whl (124 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting click>=7.0
  Downloading click-7.1.2-py2.py3-none-any.whl (82 kB)
Collecting cryptography>=2.1.4
  Downloading cryptography-3.2.1-cp35-abi3-manylinux2010_x86_64.whl (2.6 MB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting cffi!=1.11.3,>=1.8
  Downloading cffi-1.14.4-cp37-cp37m-manylinux1_x86_64.whl (402 kB)
Collecting databricks-cli>=0.8.7
  Downloading databricks-cli-0.14.1.tar.gz (54 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting docker>=4.0.0
  Downloading docker-4.4.0-py2.py3-none-any.whl (146 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting entrypoints
  Downloading entrypoints-0.3-py2.py3-none-any.whl (11 kB)
Collecting Flask
  Downloading Flask-1.1.2-py2.py3-none-any.whl (94 kB)
Collecting gitpython>=2.1.0
  Downloading GitPython-3.1.11-py3-none-any.whl (159 kB)
Collecting gitdb<5,>=4.0.1
  Downloading gitdb-4.0.5-py3-none-any.whl (63 kB)
Collecting gunicorn
  Downloading gunicorn-20.0.4-py2.py3-none-any.whl (77 kB)
Requirement already satisfied: setuptools>=3.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from gunicorn->mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (50.3.2.post20201201)
Collecting itsdangerous>=0.24
  Downloading itsdangerous-1.1.0-py2.py3-none-any.whl (16 kB)
Collecting Jinja2>=2.10.1
  Downloading Jinja2-2.11.2-py2.py3-none-any.whl (125 kB)
Collecting Mako
  Downloading Mako-1.1.3-py2.py3-none-any.whl (75 kB)
Collecting MarkupSafe>=0.23
  Downloading MarkupSafe-1.1.1-cp37-cp37m-manylinux1_x86_64.whl (27 kB)
Collecting msrest>=0.6.10
  Downloading msrest-0.6.19-py2.py3-none-any.whl (84 kB)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from msrest>=0.6.10->azure-storage-blob->mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (2020.11.8)
Collecting isodate>=0.6.0
  Downloading isodate-0.6.0-py2.py3-none-any.whl (45 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting pandas
  Downloading pandas-1.1.4-cp37-cp37m-manylinux1_x86_64.whl (9.5 MB)
Requirement already satisfied: numpy in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.19.2)
Collecting prometheus-flask-exporter
  Downloading prometheus_flask_exporter-0.18.1.tar.gz (21 kB)
Collecting prometheus_client
  Downloading prometheus_client-0.9.0-py2.py3-none-any.whl (53 kB)
Collecting protobuf>=3.6.0
  Downloading protobuf-3.14.0-cp37-cp37m-manylinux1_x86_64.whl (1.0 MB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting pycparser
  Downloading pycparser-2.20-py2.py3-none-any.whl (112 kB)
Collecting python-dateutil
  Downloading python_dateutil-2.8.1-py2.py3-none-any.whl (227 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting python-editor>=0.3
  Downloading python_editor-1.0.4-py3-none-any.whl (4.9 kB)
Collecting pytz>=2017.2
  Downloading pytz-2020.4-py2.py3-none-any.whl (509 kB)
Collecting pyyaml
  Downloading PyYAML-5.3.1.tar.gz (269 kB)
Collecting querystring-parser
  Downloading querystring_parser-1.2.4-py2.py3-none-any.whl (7.9 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting requests>=2.17.3
  Downloading requests-2.25.0-py2.py3-none-any.whl (61 kB)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from msrest>=0.6.10->azure-storage-blob->mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (2020.11.8)
Collecting chardet<4,>=3.0.2
  Downloading chardet-3.0.4-py2.py3-none-any.whl (133 kB)
Collecting idna<3,>=2.5
  Downloading idna-2.10-py2.py3-none-any.whl (58 kB)
Collecting requests-oauthlib>=0.5.0
  Downloading requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)
Collecting oauthlib>=3.0.0
  Downloading oauthlib-3.1.0-py2.py3-none-any.whl (147 kB)
Collecting smmap<4,>=3.0.1
  Downloading smmap-3.0.4-py2.py3-none-any.whl (25 kB)
Collecting sqlalchemy
  Downloading SQLAlchemy-1.3.20-cp37-cp37m-manylinux2010_x86_64.whl (1.3 MB)
Collecting sqlparse>=0.3.1
  Downloading sqlparse-0.4.1-py3-none-any.whl (42 kB)
Collecting tabulate>=0.7.7
  Downloading tabulate-0.8.7-py3-none-any.whl (24 kB)
Collecting urllib3<1.27,>=1.21.1
  Downloading urllib3-1.26.2-py2.py3-none-any.whl (136 kB)
Collecting websocket-client>=0.32.0
  Downloading websocket_client-0.57.0-py2.py3-none-any.whl (200 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow->-r /tmp/tmpw85l7lc1/condaenv.ucgtv5hk.requirements.txt (line 1)) (1.15.0)
Collecting Werkzeug>=0.15
  Downloading Werkzeug-1.0.1-py2.py3-none-any.whl (298 kB)
Building wheels for collected packages: alembic, databricks-cli, prometheus-flask-exporter, pyyaml
  Building wheel for alembic (setup.py): started
  Building wheel for alembic (setup.py): finished with status 'done'
  Created wheel for alembic: filename=alembic-1.4.1-py2.py3-none-any.whl size=158156 sha256=cf08323834b233b106f710ab4669f58c0b002d48de6691f30460fbb54f96a2c7
  Stored in directory: /tmp/pip-ephem-wheel-cache-x3m1nyz8/wheels/be/5d/0a/9e13f53f4f5dfb67cd8d245bb7cdffe12f135846f491a283e3
  Building wheel for databricks-cli (setup.py): started
  Building wheel for databricks-cli (setup.py): finished with status 'done'
  Created wheel for databricks-cli: filename=databricks_cli-0.14.1-py3-none-any.whl size=100578 sha256=889e699dfe1ea6213388f1897ff47d690085ef5fc11f34ade5cefadd534b6264
  Stored in directory: /tmp/pip-ephem-wheel-cache-x3m1nyz8/wheels/7f/d9/25/baefac3eda0e7dbf143008d2b9865e0d923d4b7306136244fe
  Building wheel for prometheus-flask-exporter (setup.py): started
  Building wheel for prometheus-flask-exporter (setup.py): finished with status 'done'
  Created wheel for prometheus-flask-exporter: filename=prometheus_flask_exporter-0.18.1-py3-none-any.whl size=17157 sha256=8d0cfad5ca828b59c8932080b3978ceff79a56f3325dad90b797913803408f71
  Stored in directory: /tmp/pip-ephem-wheel-cache-x3m1nyz8/wheels/c4/b6/b5/e76659f3b2a3a226565e27f0a7eb7a3ac93c3f4d68acfbe617
  Building wheel for pyyaml (setup.py): started
  Building wheel for pyyaml (setup.py): finished with status 'done'
  Created wheel for pyyaml: filename=PyYAML-5.3.1-cp37-cp37m-linux_x86_64.whl size=44619 sha256=89f0985ac9c07b673226c30d13503db46a87c934c8c899c7b19827ee2ead4bc5
  Stored in directory: /tmp/pip-ephem-wheel-cache-x3m1nyz8/wheels/5e/03/1e/e1e954795d6f35dfc7b637fe2277bff021303bd9570ecea653
Successfully built alembic databricks-cli prometheus-flask-exporter pyyaml
Installing collected packages: urllib3, idna, chardet, requests, pycparser, oauthlib, MarkupSafe, Werkzeug, smmap, requests-oauthlib, Jinja2, itsdangerous, isodate, click, cffi, websocket-client, tabulate, sqlalchemy, pytz, python-editor, python-dateutil, prometheus-client, msrest, Mako, gitdb, Flask, cryptography, azure-core, sqlparse, querystring-parser, pyyaml, protobuf, prometheus-flask-exporter, pandas, gunicorn, gitpython, entrypoints, docker, databricks-cli, cloudpickle, azure-storage-blob, alembic, mlflow
Successfully installed Flask-1.1.2 Jinja2-2.11.2 Mako-1.1.3 MarkupSafe-1.1.1 Werkzeug-1.0.1 alembic-1.4.1 azure-core-1.9.0 azure-storage-blob-12.6.0 cffi-1.14.4 chardet-3.0.4 click-7.1.2 cloudpickle-1.6.0 cryptography-3.2.1 databricks-cli-0.14.1 docker-4.4.0 entrypoints-0.3 gitdb-4.0.5 gitpython-3.1.11 gunicorn-20.0.4 idna-2.10 isodate-0.6.0 itsdangerous-1.1.0 mlflow-1.12.1 msrest-0.6.19 oauthlib-3.1.0 pandas-1.1.4 prometheus-client-0.9.0 prometheus-flask-exporter-0.18.1 protobuf-3.14.0 pycparser-2.20 python-dateutil-2.8.1 python-editor-1.0.4 pytz-2020.4 pyyaml-5.3.1 querystring-parser-1.2.4 requests-2.25.0 requests-oauthlib-1.3.0 smmap-3.0.4 sqlalchemy-1.3.20 sqlparse-0.4.1 tabulate-0.8.7 urllib3-1.26.2 websocket-client-0.57.0

done
#
# To activate this environment, use
#
#     $ conda activate mlflow
#
# To deactivate an active environment, use
#
#     $ conda deactivate

INFO:root:Install additional package from requirements.txt
Requirement already satisfied: pyyaml<5.4.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from -r /microservice/requirements.txt (line 1)) (5.3.1)
Requirement already satisfied: pandas<1.2.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from -r /microservice/requirements.txt (line 4)) (1.1.4)
Processing ./python
Collecting mlflow<1.12.0
  Downloading mlflow-1.11.0-py3-none-any.whl (13.9 MB)
Requirement already satisfied: gitpython>=2.1.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (3.1.11)
WARNING: The directory '/.cache/pip' or its parent directory is not owned or is not writable by the current user. The cache has been disabled. Check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.

Requirement already satisfied: protobuf>=3.6.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (3.14.0)
Requirement already satisfied: databricks-cli>=0.8.7 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.14.1)
Requirement already satisfied: sqlparse in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.4.1)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: click>=7.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (7.1.2)
Requirement already satisfied: python-dateutil in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (2.8.1)
Requirement already satisfied: docker>=4.0.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (4.4.0)
Requirement already satisfied: pyyaml<5.4.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from -r /microservice/requirements.txt (line 1)) (5.3.1)
Requirement already satisfied: numpy in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.19.2)
Requirement already satisfied: cloudpickle in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.6.0)
Requirement already satisfied: gunicorn in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (20.0.4)
Requirement already satisfied: entrypoints in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.3)
Requirement already satisfied: querystring-parser in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.2.4)
Requirement already satisfied: azure-storage-blob>=12.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (12.6.0)
Requirement already satisfied: alembic<=1.4.1 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.4.1)
Requirement already satisfied: prometheus-flask-exporter in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.18.1)
Requirement already satisfied: Flask in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.2)
Requirement already satisfied: pandas<1.2.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from -r /microservice/requirements.txt (line 4)) (1.1.4)
Requirement already satisfied: numpy in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.19.2)
Requirement already satisfied: pytz>=2017.2 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from pandas<1.2.0->-r /microservice/requirements.txt (line 4)) (2020.4)
Requirement already satisfied: python-dateutil in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (2.8.1)
Collecting requests<2.25.0
  Downloading requests-2.24.0-py2.py3-none-any.whl (61 kB)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from requests<2.25.0->-r /microservice/requirements.txt (line 2)) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from requests<2.25.0->-r /microservice/requirements.txt (line 2)) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from requests<2.25.0->-r /microservice/requirements.txt (line 2)) (2020.11.8)
Requirement already satisfied: Flask in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.2)
Requirement already satisfied: numpy in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.19.2)
Requirement already satisfied: protobuf>=3.6.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (3.14.0)
Requirement already satisfied: pyyaml<5.4.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from -r /microservice/requirements.txt (line 1)) (5.3.1)
Requirement already satisfied: gunicorn in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (20.0.4)
Requirement already satisfied: setuptools>=41.0.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from seldon-core==1.5.0->-r /microservice/requirements.txt (line 7)) (50.3.2.post20201201)
Requirement already satisfied: Mako in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from alembic<=1.4.1->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.3)
Requirement already satisfied: python-dateutil in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (2.8.1)
Requirement already satisfied: python-editor>=0.3 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from alembic<=1.4.1->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.0.4)
Requirement already satisfied: msrest>=0.6.10 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.6.19)
Requirement already satisfied: azure-core<2.0.0,>=1.9.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.9.0)
Requirement already satisfied: cryptography>=2.1.4 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (3.2.1)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: cffi!=1.11.3,>=1.8 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from cryptography>=2.1.4->azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.14.4)
Requirement already satisfied: pycparser in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from cffi!=1.11.3,>=1.8->cryptography>=2.1.4->azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (2.20)
Requirement already satisfied: tabulate>=0.7.7 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from databricks-cli>=0.8.7->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.8.7)
Requirement already satisfied: click>=7.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (7.1.2)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: websocket-client>=0.32.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from docker>=4.0.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.57.0)
Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from Flask->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.0.1)
Requirement already satisfied: click>=7.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (7.1.2)
Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from Flask->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (2.11.2)
Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from Flask->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.0)
Collecting Flask-cors<4.0.0
  Downloading Flask_Cors-3.0.9-py2.py3-none-any.whl (14 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: Flask in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.2)
Collecting Flask-OpenTracing<1.2.0,>=1.1.0
  Downloading Flask-OpenTracing-1.1.0.tar.gz (8.2 kB)
Requirement already satisfied: Flask in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.2)
Collecting flatbuffers<2.0.0
  Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from gitpython>=2.1.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (4.0.5)
Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython>=2.1.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (3.0.4)
Collecting gorilla
  Downloading gorilla-0.3.0-py2.py3-none-any.whl (11 kB)
Collecting grpcio<2.0.0
  Downloading grpcio-1.34.0-cp37-cp37m-manylinux2014_x86_64.whl (3.9 MB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Collecting grpcio-opentracing<1.2.0,>=1.1.4
  Downloading grpcio_opentracing-1.1.4-py3-none-any.whl (14 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: setuptools>=41.0.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from seldon-core==1.5.0->-r /microservice/requirements.txt (line 7)) (50.3.2.post20201201)
Collecting jaeger-client<4.4.0,>=4.1.0
  Downloading jaeger-client-4.3.0.tar.gz (81 kB)
Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from Jinja2>=2.10.1->Flask->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.1)
Collecting jsonschema<4.0.0
  Downloading jsonschema-3.2.0-py2.py3-none-any.whl (56 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: setuptools>=41.0.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from seldon-core==1.5.0->-r /microservice/requirements.txt (line 7)) (50.3.2.post20201201)
Collecting attrs>=17.4.0
  Downloading attrs-20.3.0-py2.py3-none-any.whl (49 kB)
Collecting importlib-metadata
  Downloading importlib_metadata-3.1.1-py3-none-any.whl (9.6 kB)
Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from Jinja2>=2.10.1->Flask->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.1)
Requirement already satisfied: requests-oauthlib>=0.5.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from msrest>=0.6.10->azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.3.0)
Requirement already satisfied: isodate>=0.6.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from msrest>=0.6.10->azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (0.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from requests<2.25.0->-r /microservice/requirements.txt (line 2)) (2020.11.8)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Collecting opentracing<2.5.0,>=2.2.0
  Downloading opentracing-2.4.0.tar.gz (46 kB)
Collecting prometheus_client<0.9.0,>=0.7.1
  Downloading prometheus_client-0.8.0-py2.py3-none-any.whl (53 kB)
Requirement already satisfied: Flask in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.1.2)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Collecting pyrsistent>=0.14.0
  Downloading pyrsistent-0.17.3.tar.gz (106 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Collecting redis<4.0.0
  Downloading redis-3.5.3-py2.py3-none-any.whl (72 kB)
Requirement already satisfied: oauthlib>=3.0.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from requests-oauthlib>=0.5.0->msrest>=0.6.10->azure-storage-blob>=12.0->mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (3.1.0)
Collecting sqlalchemy<=1.3.13
  Downloading SQLAlchemy-1.3.13.tar.gz (6.0 MB)
Collecting threadloop<2,>=1
  Downloading threadloop-1.0.2.tar.gz (4.9 kB)
Collecting thrift
  Downloading thrift-0.13.0.tar.gz (59 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Collecting tornado>=4.3
  Downloading tornado-6.1-cp37-cp37m-manylinux2010_x86_64.whl (428 kB)
Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1
  Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/mlflow/lib/python3.7/site-packages (from mlflow<1.12.0->-r /microservice/requirements.txt (line 3)) (1.15.0)
Collecting zipp>=0.5
  Downloading zipp-3.4.0-py3-none-any.whl (5.2 kB)
Building wheels for collected packages: seldon-core, Flask-OpenTracing, jaeger-client, opentracing, pyrsistent, sqlalchemy, threadloop, thrift
  Building wheel for seldon-core (setup.py): started
  Building wheel for seldon-core (setup.py): finished with status 'done'
  Created wheel for seldon-core: filename=seldon_core-1.5.0-py3-none-any.whl size=109970 sha256=8797fb8bc22997c8b8f623a05e6206b80f830b846a3b3593c8484843cc225d54
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/e8/4a/91/631e36445684dcfa197d1b049d70c723f75072b5a5421c0ba0
  Building wheel for Flask-OpenTracing (setup.py): started
  Building wheel for Flask-OpenTracing (setup.py): finished with status 'done'
  Created wheel for Flask-OpenTracing: filename=Flask_OpenTracing-1.1.0-py3-none-any.whl size=9069 sha256=258faeb30872c0151166f42f026ba1d4504a6417b23dd95684070b25400be1cc
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/42/22/cd/ccb93fa68f4a01fb6c10082f97bcb2af9eb8e43565ce38a292
  Building wheel for jaeger-client (setup.py): started
  Building wheel for jaeger-client (setup.py): finished with status 'done'
  Created wheel for jaeger-client: filename=jaeger_client-4.3.0-py3-none-any.whl size=64291 sha256=1c65c9f7b5a57fa2eb2ca68a931ab5e3d642053e882dea3320969a4e1fc80e6d
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/4b/b9/d9/efe18893b02a4bc5abb68e0174d4ab10147f7f184dd170758e
  Building wheel for opentracing (setup.py): started
  Building wheel for opentracing (setup.py): finished with status 'done'
  Created wheel for opentracing: filename=opentracing-2.4.0-py3-none-any.whl size=51400 sha256=6814384644bba11fa24bb1342c22b539a48942cec02734f37058f85facd66348
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/de/d6/d7/bbfcdb96ef12eb5bc0f379947c16bcb446c55ba0fe28424064
  Building wheel for pyrsistent (setup.py): started
  Building wheel for pyrsistent (setup.py): finished with status 'done'
  Created wheel for pyrsistent: filename=pyrsistent-0.17.3-cp37-cp37m-linux_x86_64.whl size=124199 sha256=a164d36ca417d1e11b2fea61fb33fc680c93da0b7d5661f68da4ed342430e4e2
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/a5/52/bf/71258a1d7b3c8cbe1ee53f9314c6f65f20385481eaee573cc5
  Building wheel for sqlalchemy (setup.py): started
  Building wheel for sqlalchemy (setup.py): finished with status 'done'
  Created wheel for sqlalchemy: filename=SQLAlchemy-1.3.13-cp37-cp37m-linux_x86_64.whl size=1224403 sha256=e251892f2d51e0156eac400a8e30e52c92506906d0054a429bb3de160b7857d5
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/b9/ba/77/163f10f14bd489351530603e750c195b0ceceed2f3be2b32f1
  Building wheel for threadloop (setup.py): started
  Building wheel for threadloop (setup.py): finished with status 'done'
  Created wheel for threadloop: filename=threadloop-1.0.2-py3-none-any.whl size=3424 sha256=bb121972b67d6af1d8cc3a58e8678b717001565778a3d7cfd7679b4647ea0fb7
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/08/93/e3/037c2555d98964d9ca537dabb39827a2b72470a679b5c0de37
  Building wheel for thrift (setup.py): started
  Building wheel for thrift (setup.py): finished with status 'done'
  Created wheel for thrift: filename=thrift-0.13.0-cp37-cp37m-linux_x86_64.whl size=487215 sha256=f0ba610a91131f2fa471c0017c85f959caca2a5f2f2df1ee536aa7d0c2995a4e
  Stored in directory: /tmp/pip-ephem-wheel-cache-mz663v2c/wheels/79/35/5a/19f5dadf91f62bd783aaa8385f700de9bc14772e09ab0f006a
Successfully built seldon-core Flask-OpenTracing jaeger-client opentracing pyrsistent sqlalchemy threadloop thrift
Installing collected packages: urllib3, requests, zipp, tornado, thrift, threadloop, sqlalchemy, pyrsistent, prometheus-client, opentracing, importlib-metadata, grpcio, attrs, redis, jsonschema, jaeger-client, grpcio-opentracing, gorilla, flatbuffers, Flask-OpenTracing, Flask-cors, seldon-core, mlflow
  Attempting uninstall: urllib3
    Found existing installation: urllib3 1.26.2
    Uninstalling urllib3-1.26.2:
      Successfully uninstalled urllib3-1.26.2
  Attempting uninstall: requests
    Found existing installation: requests 2.25.0
    Uninstalling requests-2.25.0:
      Successfully uninstalled requests-2.25.0
  Attempting uninstall: sqlalchemy
    Found existing installation: SQLAlchemy 1.3.20
    Uninstalling SQLAlchemy-1.3.20:
      Successfully uninstalled SQLAlchemy-1.3.20
  Attempting uninstall: prometheus-client
    Found existing installation: prometheus-client 0.9.0
    Uninstalling prometheus-client-0.9.0:
      Successfully uninstalled prometheus-client-0.9.0
  Attempting uninstall: mlflow
    Found existing installation: mlflow 1.12.1
    Uninstalling mlflow-1.12.1:
      Successfully uninstalled mlflow-1.12.1
Successfully installed Flask-OpenTracing-1.1.0 Flask-cors-3.0.9 attrs-20.3.0 flatbuffers-1.12 gorilla-0.3.0 grpcio-1.34.0 grpcio-opentracing-1.1.4 importlib-metadata-3.1.1 jaeger-client-4.3.0 jsonschema-3.2.0 mlflow-1.11.0 opentracing-2.4.0 prometheus-client-0.8.0 pyrsistent-0.17.3 redis-3.5.3 requests-2.24.0 seldon-core-1.5.0 sqlalchemy-1.3.13 threadloop-1.0.2 thrift-0.13.0 tornado-6.1 urllib3-1.25.11 zipp-3.4.0

Activating Conda environment 'mlflow'
starting microservice
2020-12-04 11:30:46,041 - seldon_core.microservice:main:201 - INFO:  Starting microservice.py:main
2020-12-04 11:30:46,041 - seldon_core.microservice:main:202 - INFO:  Seldon Core version: 1.5.0
2020-12-04 11:30:46,042 - seldon_core.microservice:main:314 - INFO:  Parse JAEGER_EXTRA_TAGS []
2020-12-04 11:30:46,043 - seldon_core.microservice:load_annotations:153 - INFO:  Found annotation kubernetes.io/config.seen:2020-12-04T13:27:37.316214907+02:00 
2020-12-04 11:30:46,043 - seldon_core.microservice:load_annotations:153 - INFO:  Found annotation kubernetes.io/config.source:api 
2020-12-04 11:30:46,043 - seldon_core.microservice:load_annotations:153 - INFO:  Found annotation prometheus.io/path:/prometheus 
2020-12-04 11:30:46,043 - seldon_core.microservice:load_annotations:153 - INFO:  Found annotation prometheus.io/scrape:true 
2020-12-04 11:30:46,043 - seldon_core.microservice:main:317 - INFO:  Annotations: {'kubernetes.io/config.seen': '2020-12-04T13:27:37.316214907+02:00', 'kubernetes.io/config.source': 'api', 'prometheus.io/path': '/prometheus', 'prometheus.io/scrape': 'true'}
2020-12-04 11:30:46,043 - seldon_core.microservice:main:321 - INFO:  Importing MLFlowServer
2020-12-04 11:30:46,571 - root:__init__:21 - INFO:  Creating MLFLow server with URI /mnt/models
2020-12-04 11:30:46,571 - root:__init__:22 - INFO:  xtype: ndarray
2020-12-04 11:30:46,587 - seldon_core.microservice:main:403 - INFO:  REST gunicorn microservice running on port 9000
2020-12-04 11:30:46,588 - seldon_core.microservice:main:457 - INFO:  REST metrics microservice running on port 6000
2020-12-04 11:30:46,588 - seldon_core.microservice:main:467 - INFO:  Starting servers
2020-12-04 11:30:46,601 - seldon_core.wrapper:_set_flask_app_configs:208 - INFO:  App Config:  <Config {'ENV': 'production', 'DEBUG': False, 'TESTING': False, 'PROPAGATE_EXCEPTIONS': None, 'PRESERVE_CONTEXT_ON_EXCEPTION': None, 'SECRET_KEY': None, 'PERMANENT_SESSION_LIFETIME': datetime.timedelta(days=31), 'USE_X_SENDFILE': False, 'SERVER_NAME': None, 'APPLICATION_ROOT': '/', 'SESSION_COOKIE_NAME': 'session', 'SESSION_COOKIE_DOMAIN': None, 'SESSION_COOKIE_PATH': None, 'SESSION_COOKIE_HTTPONLY': True, 'SESSION_COOKIE_SECURE': False, 'SESSION_COOKIE_SAMESITE': None, 'SESSION_REFRESH_EACH_REQUEST': True, 'MAX_CONTENT_LENGTH': None, 'SEND_FILE_MAX_AGE_DEFAULT': datetime.timedelta(seconds=43200), 'TRAP_BAD_REQUEST_ERRORS': None, 'TRAP_HTTP_EXCEPTIONS': False, 'EXPLAIN_TEMPLATE_LOADING': False, 'PREFERRED_URL_SCHEME': 'http', 'JSON_AS_ASCII': True, 'JSON_SORT_KEYS': True, 'JSONIFY_PRETTYPRINT_REGULAR': False, 'JSONIFY_MIMETYPE': 'application/json', 'TEMPLATES_AUTO_RELOAD': None, 'MAX_COOKIE_SIZE': 4093}>
2020-12-04 11:30:46,609 - root:load:28 - INFO:  Downloading model from /mnt/models
2020-12-04 11:30:46,610 - root:download:31 - INFO:  Copying contents of /mnt/models to local
2020-12-04 11:30:46,613 - seldon_core.wrapper:_set_flask_app_configs:208 - INFO:  App Config:  <Config {'ENV': 'production', 'DEBUG': False, 'TESTING': False, 'PROPAGATE_EXCEPTIONS': None, 'PRESERVE_CONTEXT_ON_EXCEPTION': None, 'SECRET_KEY': None, 'PERMANENT_SESSION_LIFETIME': datetime.timedelta(days=31), 'USE_X_SENDFILE': False, 'SERVER_NAME': None, 'APPLICATION_ROOT': '/', 'SESSION_COOKIE_NAME': 'session', 'SESSION_COOKIE_DOMAIN': None, 'SESSION_COOKIE_PATH': None, 'SESSION_COOKIE_HTTPONLY': True, 'SESSION_COOKIE_SECURE': False, 'SESSION_COOKIE_SAMESITE': None, 'SESSION_REFRESH_EACH_REQUEST': True, 'MAX_CONTENT_LENGTH': None, 'SEND_FILE_MAX_AGE_DEFAULT': datetime.timedelta(seconds=43200), 'TRAP_BAD_REQUEST_ERRORS': None, 'TRAP_HTTP_EXCEPTIONS': False, 'EXPLAIN_TEMPLATE_LOADING': False, 'PREFERRED_URL_SCHEME': 'http', 'JSON_AS_ASCII': True, 'JSON_SORT_KEYS': True, 'JSONIFY_PRETTYPRINT_REGULAR': False, 'JSONIFY_MIMETYPE': 'application/json', 'TEMPLATES_AUTO_RELOAD': None, 'MAX_COOKIE_SIZE': 4093}>
[2020-12-04 11:30:46 +0000] [376] [INFO] Starting gunicorn 20.0.4
[2020-12-04 11:30:46 +0000] [376] [INFO] Listening at: http://0.0.0.0:6000 (376)
[2020-12-04 11:30:46 +0000] [376] [INFO] Using worker: sync
[2020-12-04 11:30:46 +0000] [7] [INFO] Starting gunicorn 20.0.4
[2020-12-04 11:30:46 +0000] [7] [INFO] Listening at: http://0.0.0.0:9000 (7)
[2020-12-04 11:30:46 +0000] [7] [INFO] Using worker: threads
[2020-12-04 11:30:46 +0000] [382] [INFO] Booting worker with pid: 382
[2020-12-04 11:30:46 +0000] [383] [INFO] Booting worker with pid: 383
2020-12-04 11:30:46,641 - seldon_core.app:load:82 - INFO:  Tracing branch is active
2020-12-04 11:30:46,650 - seldon_core.utils:setup_tracing:724 - INFO:  Initializing tracing
2020-12-04 11:30:46,699 - seldon_core.utils:setup_tracing:731 - INFO:  Using default tracing config
2020-12-04 11:30:46,699 - jaeger_tracing:_create_local_agent_channel:446 - INFO:  Initializing Jaeger Tracer with UDP reporter
2020-12-04 11:30:46,703 - jaeger_tracing:new_tracer:384 - INFO:  Using sampler ConstSampler(True)
2020-12-04 11:30:46,704 - jaeger_tracing:_initialize_global_tracer:436 - INFO:  opentracing.tracer initialized to <jaeger_client.tracer.Tracer object at 0x7f1e97398208>[app_name=MLFlowServer]
2020-12-04 11:30:46,704 - seldon_core.app:load:87 - INFO:  Set JAEGER_EXTRA_TAGS []
2020-12-04 11:30:46,705 - root:load:28 - INFO:  Downloading model from /mnt/models
2020-12-04 11:30:46,705 - root:download:31 - INFO:  Copying contents of /mnt/models to local
2020-12-04 11:30:47,214 - seldon_core.microservice:grpc_prediction_server:433 - INFO:  GRPC microservice Running on port 9500


```

once two of two containers report ready, the Pod can accept posted data and serve predictions. Test the deployment with curl by posting the model’s expected input, in this case a two-dimensional array (or an array of arrays), each containing the 11 values required to make a prediction. The model returns one prediction per inner array:

Example output:
```
$ curl -k -X POST https://quality.data.davar.com/api/v1.0/predictions -H 'Content-Type: application/json' -d '{ "data": { "ndarray": [[ 6.4, 0.57, 0.02, 1.8, 0.067, 4.0, 11.0, 0.997, 3.46, 0.68, 9.5]] } }'
{"data":{"names":[],"ndarray":[5.644703919933278]},"meta":{"requestPath":{"quality":"seldonio/mlflowserver:1.5.0"}}}

```
Returned prediction is 5.644703919933278 ("ndarray":[5.644703919933278]})


Note: Given a set of features as inputs, the task here is to predict the quality of wine on a scale of [0-10]. 

Input variables (based on physicochemical tests):

    fixed acidity
    volatile acidity
    citric acid
    residual sugar
    chlorides
    free sulfur dioxide
    total sulfur dioxide
    density
    pH
    sulphates
    alcohol

Output variable (based on sensory data): quality (score between 0 and 10)


This k8s+MLFlow/Seldon Core Demo moved quickly, lightly scratching the surface of Seldon Core’s capabilities. However, it demonstrated nearly seamless interoperability between a range of diverse components, from building scikit-learn models in Jupyter Notebooks and tracking and serving the models in MLflow to their final deployment with Seldon Core, all integrated atop Kubernetes.

### TODO:WIP MLOps CI/CD --->  Jenkins CI/CD and Argo CD GitOps ---> Jenkins: Automate with MinIO(notifications/events:notify_webhook) + Jenkins(generic-webhook-trigger plugin)  && Argo CD (GitOps).

One of the core tenets of data science that differentiates it from software engineering is its focus on experimentation. With software, you develop, you test, and you push features that are primarily code-based. In data science, on the other hand, you conduct heaps of experiments while making changes in configuration, data, features, etc. The output isn’t even necessarily “completed code,” but artifacts such as model weights.

Git-oriented source control isn’t the right tool for this problem. To create a robust training pipeline, you’d have to build quite a bit. Here, we’ll just ignore any restrictions on the model training approach and offload the results of our experiments in a central, shared repository (MinIO). Mlflow has become one of the most popular tools for experiment management. While it does offer a lot more than simple tracking functions.

We definitely don’t want to run things manually every time we update our code or model, so let’s welcome some automation.
Jenkins for CI/CD

CI/CD refer to continuous integration and continuous delivery. The former involves frequent and iterative merge/build/testing of code to mitigate the risk of things breaking. The latter aims to ensure rapid and easy production deployments. There’s probably some overlap and ambiguity in the responsibilities of each, at least from my persepctive. I recommend looking it up if you’re interested in the specifics.

#### Jenkins

Jenkins is typically used in tandem with a source control management (SCM) tool like Github. Jenkins projects are often configured to trigger builds or run scripts when people push or merge branches in their repo. However, there are a number of available hooks. It can also be used to automate:

    Building/tagging/pushing of software (Docker images)
    Unit tests, integration tests, etc…
    Deployment to dev/stage/prod environments (if you use it for CD)

Run jenkins pod inside k8s 

Install J.plugin 
```
def pluginParameter="generic-webhook-trigger"
	def plugins = pluginParameter.split()
	println(plugins)
	def instance = Jenkins.getInstance()
	def pm = instance.getPluginManager()
	def uc = instance.getUpdateCenter()
	def installed = false
	
	plugins.each {
	  if (!pm.getPlugin(it)) {
	    def plugin = uc.getPlugin(it)
	    if (plugin) {
	      println("Installing " + it)
	      plugin.deploy()
	      installed = true
	    }
	  }
	}
	
	instance.save()
```

Configure MinIO notifications:

```
mc alias set s3 http://localhost:9000
mc ls s3/mlflow
mc admin config set s3 notify_webhook:1 queue_limit="0" queue_dir='.' endpoint="http://JENKINS_URL/generic-webhook-trigger/invoke"
mc admin service restart s3
mc event add s3/mlflow arn:minio:sqs::1:webhook --suffix .pkl
```

  - Configure Jenkins to build when new models are trained and pushed to S3 or GCS
  - Configure ml-app code so that it always pulls the latest model
  - Build and test the model container with the new artifact so that it’s verified to escalate to production!

### Argo CD 
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo3-AutoML-MLFlof-SeldonCore/pictures/ML_mlflow-1024x652.png" width="800">


Argo CD is a declarative, GitOps continuous delivery tool for Kubernetes.

Setup Argo CD

The other thing we need to install is argo CD. We could’ve done the deployment to k8s using Jenkins, but I really wanted to try this one out.
```
kubectl create ns argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```
This will create a bunch more resources.


We can configure argo to run deployments for us instead of kubectl apply -f. The goal here to always have our deployment in sync with the representation of our deployment (YAML).

We’re going to make it so we can access Argo via HTTP. In a new terminal, run:

`kubectl port-forward -n argocd svc/argocd-server 8080:80`

Wait until the pods come up. You can go to localhost:8080 to access Argo. By default, the user is “admin” and the password is the name of the server pod. You can get this by running: `kubectl get po -n argocd | grep argocd-server`
