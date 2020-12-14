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
