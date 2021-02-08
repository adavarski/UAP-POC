In this Project, we work with 2 Docker Images that we call Base and Dag. The base image is used to contain Airflow itself without the Airflow DAGs, and the Dag is used for only containing the Airflow DAGs. They are separated, because the Dag image gets built a lot more during a project, so we save building time this way but not having them in the same image. We also use virtualenv for further separation of the dependencies between the two.

### base
This folder contains the everything that is needed to build the Base Image, along with the Dockerfile, it has:

- The config file for Airflow called airflow.cfg
- A bootstrap.sh script which runs airflow initdb and creates an Airflow User. This executes once when the Helm Chart deployment for Airflow happens
(See helm/templates/deployment.yaml/initContainers section)
- requirements.txt, which contains Airflow's dependencies, e.g. the airflow-exporter, which is needed for using Prometheus with Airflow

The rebuilding of this image is only needed if something that is directly related to Airflow is changed, for example Airflow dependencies, Airflow version number.

### dag

This folder contains the everything that is needed to build the Dag Image, along with the Dockerfile, it has:

- A folder called airflow-dags, which is used to store files that define Airflow DAGs.
- A ci folder, for storing scripts that call Python scripts, and other DAG related .sh scripts.
- example(pipeline) folder, which is used for storing the Python files that we want to execute with our DAGs.
- The requirements.txt, where we put the dependencies of the Python scripts.

The rebuilding of this image is needed every time when a change to a DAG happens. Rebuilding this image is basically a refresh of DAGs and their related files in Airflow, so this process happens frequently in development stages.

### Build images

Before you build the your dag image don't forget to add/change the first line to this: FROM davarski/airflow-base
```
cd ./base
docker build -t davarski/airflow-base .
cd ../dag
docker build -t davarski/airflow-dag .
docker tag davarski/airflow-dag davarski/airflow-dag:1.0.0
docker login
docker push davarski/airflow-base
docker push davarski/airflow-dag
docker push davarski/airflow-dag:1.0.0
```

