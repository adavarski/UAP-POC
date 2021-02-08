## Production-like Airflow Development Environment on k3s

This is a everything you need to deploy Airflow on k3s for development (DAGs: Directed Acyclic Graphs).

Image names are: davarski/airflow-base and davarski/airflow-dag.

Airflow UI password can be found in helm/files/secrets/airflow/AFPW

## Set up everyhting by hand

### Build docker images

In this Project, we work with 2 Docker Images that we call Base and Dag. The base image is used to contain Airflow itself without the Airflow DAGs, and the Dag is used for only containing the Airflow DAGs. They are separated, because the Dag image gets built a lot more during a project, so we save building time this way but not having them in the same image. We also use virtualenv for further separation of the dependencies between the two.

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
docker push davarski/airflow-dag:1.0.0
```
### Create namespace for development
```
kubectl create namespace development
```
Deploy the PostgreSQL service to create the metadat-database for Airflow
```
kubectl apply -f postgres/ --namespace development
```

### Logging

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

### The easy way to give cloud access to your scripts

#### For GCP
Ref: https://airflow.apache.org/docs/apache-airflow-providers-google/stable/connections/gcp.html

#### For AWS
Add these to your dag files
```
secret_id = os.getenv("AWS_ACCESS_KEY_ID", None)
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)

env_vars={
        'AWS_ACCESS_KEY_ID': variable_id,
        'AWS_SECRET_ACCESS_KEY': variable_secret_key}
```
And put your credentials into the helm/files/secrets/airflow appropriate files. (AWS_ACCESS_KEY_ID, AWS_SECRET_KEY)

### Deploy airflow helm chart
```

cd into ./helm 
# run the following cmd
helm install airflow-k3s-dev . --namespace development
```

Run the following cmds: kubectl port-forward and go to url http://localhost:8080 with your browser (user:Airflow password:adminadmin):
```
$kubectl get po -n development
NAME                        READY   STATUS    RESTARTS   AGE
postgres-6c869f86c5-dqw8j   1/1     Running   0          65m
airflow-77bfd6b5bb-w7628    2/2     Running   0          56s
$ kubectl port-forward airflow-77bfd6b5bb-w7628 8080:8080 -n development

```
Note: Airflow UI password can be found in helm/files/secrets/airflow/AFPW; postgres user and password are postgres (base64 encoded)

### Deploy new image to our cluster

#### Helm upgrade
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
Note: For example DAG create AWS resource:

```
aws configure
cat ~/.aws/credentials
aws secretsmanager create-secret --name demo --description "Basic Create Secret for Airflow DAG" --secret-string S3@tt13R0cks
aws secretsmanager describe-secret --secret-id demo
#Clean after DAG test
aws secretsmanager delete-secret --secret-id  --recovery-window-in-days 7
```

### Clean up
```
helm uninstall airflow-k3s-dev --namespace development
kubectl delete namespace development
```

## CLI usage

With the help of the cli tool you can set aws creds, deploy everything with the base setup, copy dag/project/requirements files and change image on the cluster after you modified any of your project/dag files with only one command. Important to check the copied files especially the dag files since this deployment created a namespace called development and deploys everything there. Which mean you have to change the namespace in the KubernetesPodOperator. The image is called: davarski/airflow-dag.

```
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

```
### Note: Apache Nifi vs Apache Airflow

They seems to be completely different animals. Nifi is scalable stream ingestion/processing framework. It provides some critical features like back pressure, gauranted delivery, data provence etc. You can use nifi to push data form millions of IOT device to a centralized cluster.

Airflow on other hand seems to be workflow orchestrator. It’s main use-case seems to be to schedule and run complex workflow’s.

Usecase in this SaaS ---> Workflows: reding the data from json sources, avro/parquet formats and keep the data in kafka and further picked up spark streaming to do some stream processing, which tool is better:Nifi or Airflow --- > Airflow is more of an orchestration tool, whereas NiFi is built for processing data in distributed fashion. NiFi is the perfect fit for this usecase/workflow, you can quickly spin up a NiFi flow without writing any code.


- Apache Nifi: The first in the list of the best ETL tools is an open source project, Apache NiFi. Developed by the Apache Software Foundation, it is based on the concept of Dataflow Programming. This means that this ETL tool allows us to visually assemble programs from boxes and run them almost without coding. So, you don't have to know any programming languages.
This ETL tool helps to create long-running jobs and is suited for processing both streaming data and periodic batches. As for manually managed jobs, they are also possible. However, there is a risk to face difficulties while setting them up.
```
Pros:

Perfect implementation of dataflow programming concept.
The opportunity to handle binary data.
Data provenance.

Cons:

Simplistic UI.
Lack of live monitoring and per-record statistics.
```

- Apache Airflow(definition!): This modern platform for designing, creating and tracking workflows is an open source ETL software. It can be used with cloud services, including GCP, Azure, and AWS. There is an opportunity to run Airflow on Kubernetes using Astronomer Enterprise.
You can code in Python, but not have to worry about XML or drag-and-drop GUIs. The workflows are written in Python, however, the steps themselves can be done in anything you want. Airflow was created as a perfectly flexible task scheduler. One of the top ETL tools is suitable for lots of different purposes. It is used to train ML models, send notifications, track systems, and power functions within various APIs.
```
Pros:

Suits for different types of tasks.
User-friendly interface for clear visualization.
Scalable solution.

Cons:

Isn't suitable for streaming jobs.
Requires additional operators.
```
Code-first: write code to generate DAGs dynamically, which is unlike the declarative style of all drag-and-drop and YAML/XML defined tools.
Python: allows for collaboration with data scientists, and it's more reasonable with Airflow to expect that data scientists could may author their own pipelines.

Airflow is cron on steroids with a nice UI and multiple methods for horizontal scaling. It allows for rapid iteration of ideas and cross-functional collaboration. Airflow is winning in the same way that Javascript is winning in the application development space - it's a good blend of accessible and useful.

In 2020, at least with Google searches, "Apache Airflow" is now overtaking "Apache NiFi" - so you'll probably hear about it more and more...

Ref: https://trends.google.com/trends/explore?date=all&q=apache%20airflow,apache%20nifi
