# JupyterLab Images

This repository contains the Docker build files for the JupyterLab images that are part of the k8s integrated Data Science and Analytics platform. The container images package dependencies and libraries for computer vision, machine learning, and other Data Science work.

This repository is stuctured in a series of layers. Images at the bottom provide light-weight Spark and Dask executors, images in the middle provide components for running headless Spark driver applications within a Kubernetes environment, and images at the top provide Jupyter and high-level libraries for interactively working with data.

## Core Spark:


* `Dockerfile.k8s-minio.executor`: core foundation image for environment. Provides a minimal Spark environment with Python, Scala, and R runtimes. It also includes the dependencies needed to work with files stored in Amazon S3 or MinIO (via the s3a storage driver for Spark).
	- Tagged as `davarski/spark301-k8s-minio-base`
	- Parent: `nvidia/cuda:11.0-runtime-ubuntu20.04`


* `Dockerfile.k8s-minio.driver`: extension of the Spark executor image that provides additional components, such as kubectl so that the image can be used to run headless driver components on a Kubernetes cluster.
	- Tagged as `davarski/spark301-k8s-minio-driver`
	- Parent: `davarski/spark301-k8s-minio-base`

## Jupyter and general Data Science:

* `Dockerfile.k8s-minio.jupyter`: minimal Jupyter image that provdes the core components of the Scientific Python stack: NumPy, Pandas, Matplotlib, Seaborn, Boken, and SciPy.
	- Tagged as `davarski/spark301-k8s-minio-jupyter`
	- Parent: `davarski/spark301-k8s-minio-driver`

* `Dockerfile.hub-jupyter`: JupyterLab Python image used in deployments with a broad set of data utilities for working with Data Engineering and Data Science libraries for NLP, Machine Vision, Geographic Information Systems, and Medical Informatics.
	- Tagged as `davarski/spark301-k8s-minio-kafka:latesta`
	- Parent: `davarski/spark301-k8s-minio-jupyter`

* `Dockerfile.k8s-minio.deep-learning`: Deep Learning image that includes NVIDIA drivers, CUDA utilities, TensorFlow, and PyTorch.
        - Tagged as `davarski/spark301-k8s-minio-dl:latest`
        - Parent: `davarski/spark301-k8s-minio-kafka:latest`

* `Dockerfile.hub-polyglot`: Extension of the JupyterHub image that provides Python, Scala/Java, and R kernels.
	- Tagged as `davarski/spark301-k8s-minio-polyglot:latest`
	- Parent: `davarski/spark301-k8s-minio-dl:latest`

## Dask runtime (Python cluster computing):

* `Dockerfile.cluster-dask`: Dask distributed computing framework and associated libraries.
	- Tagged as `davarski/spark301-minio-dask:latest`
	- Parent: `davarski/spark301-k8s-minio-polyglot:latest`

## Computer Vision and image segmentation:

* `Dockerfile.itk`: Tools for image and volume visualization including the Insight Toolkit (ITK) and Visualization Toolkit (VTK).
	- Tagged as `davarski/spark301-minio-itk:latest`
	- Parent: `davarski/spark301-minio-dask:latest`

* `Dockerfile.cv`: Tools and dependencies useful for working on computer vision problems.
	- Tagged as `davarski/spark301-minio-cv:latest`
	- Parent: `davarski/spark301-minio-itk:latest`
