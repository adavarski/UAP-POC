

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

Open a Jupyter environment:

