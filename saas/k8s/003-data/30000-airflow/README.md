# Production-like Airflow Development Environment on k3s
This is a everything you need to deploy Airflow on k3s.

Image names are: davarski/airflow-base and davarski/airflow-dag.

Airflow UI password can be found in helm/files/secrets/airflow/AFPW

## From here you can copy files and set AWS creds manually 

### Set up everyhting by hand

Go to the docker/base and build the base img after that go to docker/dag and build the dag image

Before you build the your dag image don't forget to add/change the first line to this: FROM davarski/airflow-base
```
cd ./docker/base
docker build -t davarski/airflow-base .
cd ../dag
docker build -t davarski/airflow-dag .
docker tag davarski/airflow-dag davarski/airflow-dag:1.0.0
docker login 
docker push davarski/airflow-base
docker push davarski/airflow-dag
docker push  davarski/airflow-dag:1.0.0
```
## Create namespace for dev
```
kubectl create namespace development
```
Deploy the PostgreSQL service to create the metadat-databse for Airflow
```
kubectl apply -f postgres/ --namespace development
```

## Logging

Choose according to your preference. By default the local logging option is used since this is a development environment. 


### Cloud storage logging: deployment.yml file and also modify the airflow.cfg file inside the docker/base folder.
Create the MyLogConn variable on the Airflow UI, use the same name what is inside the airflow.cfg file.
#### FOR AWS LOGGING ###
conn id: MyLogConn
conn type: S3
host: bucket name
login: AWS_ACCESS_KEY_ID
password: AWS_SECRET_ACCESS_KEY

#### FOR GCP
var name: MyLogConn
conn type: Google Cloud Platform
project id: gcp project name
scopes: https://www.googleapis.com/auth/cloud-platform
Keyfile JSON: the service_account.json  (upload the actual json file)

### Install Lens the Kubernetes IDE
With Lens you can connect to any of your Kubernetes clusters easily and you just have to click on the pod (Airflow task in this case) and click on the first icon from the left in the top right corner. After that in the built-in terminal you'll see the logs.

## The easy way to give cloud access to your scripts
### For GCP
Ref: https://airflow.apache.org/docs/apache-airflow-providers-google/stable/connections/gcp.html

### For AWS
Add these to your dag files
```
secret_id = os.getenv("AWS_ACCESS_KEY_ID", None)
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)

env_vars={
        'AWS_ACCESS_KEY_ID': variable_id,
        'AWS_SECRET_ACCESS_KEY': variable_secret_key}
```
And put your credentials into the helm/files/secrets/airflow appropriate files. (AWS_ACCESS_KEY_ID, AWS_SECRET_KEY)

## Deploy airflow
```

cd into ./helm and run the following cmd
```
helm install airflow-k3s-dev . --namespace development
```

Run the following cmd, kubectl port-forward and go to url http://localhost:8080 with your browser (user:Airflow password:adminadmin):
```
$kubectl get po -n development
NAME                        READY   STATUS    RESTARTS   AGE
postgres-6c869f86c5-dqw8j   1/1     Running   0          65m
airflow-77bfd6b5bb-w7628    2/2     Running   0          56s
$ kubectl port-forward airflow-77bfd6b5bb-w7628 8080:8080 -n development


```
Airflow UI password can be found in helm/files/secrets/airflow/AFPW

## Deploy new image to our cluster
### Helm upgrade
First you have to give some unique tag to your new dag image

Example:
```
docker build -t davarski/airflow-dag:modified .
docker tag davarski/airflow-dag:modified davarski/airflow-dag:latest
docker push davarski/airflow-dag:modified
docker push davarski/airflow-dag:latest
```
After this you have to change the image in you cluster
```
helm upgrade airflow-k3s-dev helm/ --install --wait --atomic --set dags_image.tag=modified -n development
```
Note: For DAG example create:

```
aws configure
cat ~/.aws/credentials
aws secretsmanager create-secret --name demo --description "Basic Create Secret for Airflow DAG" --secret-string S3@tt13R0cks
aws secretsmanager describe-secret --secret-id demo
#Clean after DAG test
aws secretsmanager delete-secret --secret-id  --recovery-window-in-days 7
```

## Clean up
```
helm uninstall airflow-k3s-dev --namespace development
kubectl delete namespace development
```

Note: CLI usage

With the help of the cli tool you can set aws creds, deploy everything with the base setup, copy dag/project/requirements files and change image on the cluster after you modified any of your project/dag files with only one command.
Important to check the copied files especially the dag files since this deployment created a namespace called development and deploys everything there. Which mean you have to change the namespace in the KubernetesPodOperator. Also now the image is called: davarski/airflow-dag.


$ python3 main.py --help
usage: main.py [-h] [--deploy] [--helm_update]
               [--set_aws_access_id SET_AWS_ACCESS_ID]
               [--set_aws_secret_key SET_AWS_SECRET_KEY]
               [--path_local_dags PATH_LOCAL_DAGS]
               [--path_project_folder PATH_PROJECT_FOLDER]
               [--path_local_requirements PATH_LOCAL_REQUIREMENTS]
               [--clean_up]

This is MiniDeployer - it will automatize your Apache Airflow DEV deployment

optional arguments:
  -h, --help            show this help message and exit
  --deploy              Runs the deployment script
  --helm_update         Rebuilds the dag img and change the image on the
                        cluster.
  --set_aws_access_id SET_AWS_ACCESS_ID
                        Sets the AWS access key id.
  --set_aws_secret_key SET_AWS_SECRET_KEY
                        Set the AWS secret access key.
  --path_local_dags PATH_LOCAL_DAGS
                        The absolute path to your DAGs which which you want to
                        copy into the image.
  --path_project_folder PATH_PROJECT_FOLDER
                        The absolute path to your local project folder which
                        you want to copy into the image.
  --path_local_requirements PATH_LOCAL_REQUIREMENTS
                        The absolute path to your requirements which you want
                        to copy into the image.
  --clean_up            Deletes all the Kubernetes objects inside the
                        development namspace.

