

## Model Development

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

### Tracking Models

MLflow API endpoint for tracking model development at http://mlflow.data:5000.

Jupyter environment:

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

100-sd-quality.yml ---> modelUri: s3://mlflow/artifacts/1/e22b3108e7b04c269d65b3f081f44166/artifacts/model/MLmodel


Apply the SeldonDeployment and Ingress:
```
$ kubectl apply -f 100-sd-quality.yml
```
It may take several minutes to deploy the model. Monitor the newly generated Pod in the default namespace for status; once two of two containers report ready, the Pod can accept posted data and serve predictions. Test the deployment with curl by posting the model’s expected input, in this case a two-dimensional array (or an array of arrays), each containing the 11 values required to make a prediction. The model returns one prediction per inner array:
```
$ curl -X POST https://quality.data.davar.com/api/v1.0/
predictions \
      -H 'Content-Type: application/json' \
      -d '{ "data": { "ndarray": [[ 6.4, 0.57, 0.02, 1.8,
0.067, 4.0, 11.0, 0.997, 3.46, 0.68, 9.5]] } }'

Returned prediction is 5.703684339252623:
{"data":{"names":[],"ndarray":[5.703684339252623]},"meta":{}}
```
This k8s+MLFlow/Seldon Core Demo moved quickly, lightly scratching the surface of Seldon Core’s capabilities. However, it demonstrated nearly seamless interoperability between a range of diverse components, from building scikit-learn models in Jupyter Notebooks and tracking and serving the models in MLflow to their final deployment with Seldon Core, all integrated atop Kubernetes.


