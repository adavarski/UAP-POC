# mlflow-project-example

To run this code locally, you can clone the repo and execute:
* mlflow run .

If you want to pass in parameters, you can specify them as well:
* mlflow run . -P file_path=data/sf-airbnb-clean.parquet -P max_depth=5 -P num_trees=100

Or, you can run it this way:
* mlflow.run("https://github.com/adavarski/DataScience-DataOps_MLOps-Playground/tree/main/k8s/Demo6-Spark-ML/mlflow", parameters={"max_depth": 5, "num_trees": 100})
