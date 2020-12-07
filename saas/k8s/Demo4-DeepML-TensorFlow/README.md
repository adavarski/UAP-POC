
### JupyterLab/Jupyter Notebooks environment


Jupyter Notebooks are a browser-based (or web-based) IDE (integrated development environments) applications that allows users to create, edit, run, and share their code with ease. This application gets its name from the main languages that it supports: Julia, Python, and R.


Build custom JupyterLab docker image and pushing it into DockerHub container registry.
```
$ cd ./jupyterlab
$ docker build -t jupyterlab-eth .
$ docker tag jupyterlab-eth:latest davarski/jupyterlab-eth:latest
$ docker login 
$ docker push davarski/jupyterlab-eth:latest
```
Run Jupyter Notebook inside k8s as pod:

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


### Python’s Machine Learning Libraries (Ref)

A library in the programming world is a collection of methods and functions that allow a user to perform various tasks without having to write the necessary code for it. We use libraries to save time while we program. Python has a huge array of open source machine learning libraries, including, but not limited to, the following:

• Pandas: The Pandas library provides users with the ability to handle large datasets. It provides tools for reading and writing data, cleaning and altering data,
and so on.

• Numpy: The Numpy, or Numerical Python, library provides users with a powerful array of computing abilities. It tackles the problem of slow mathematical computations and allows users to perform huge calculations with the help of multi-dimensional arrays.

• Scipy: The Scipy library is used for scientific and technical computations. It works on Numpy’s multi-dimensional arrays.

• Scikit-Learn: The Scikit-Learn library consists of various features and methods that have specially been made to assist users in their machine learning
requirements. It makes use of the Numpy library, specifically when it comes to array operations.

• TensorFlow: The TensorFlow library is an increasingly popular library that provides users with a large set of flexible and accessible tools for machine learning. 


Machine learning involves data science techniques (like cleaning, manipulating, and visualizing data), mathematical techniques, and statistical techniques. Keeping this in mind, some of the most commonly used Python libraries for [machine learning] include Matplotlib, Seaborn, Pandas, Scikit-learn, Numpy, Scipy, and so on.
These libraries have been tried and tested and were found to be easy to work with. They have thus gained popularity over the years, with numerous applications in various machine learning programs. With the growing enthusiasm toward [deep machine learning], there arose a need to create libraries that could assist with building multi-layered [neural networks]. Thus, libraries like [Theano], [PyTorch], [OpenCV], [Caffe], [Apache Spark], [Keras], and [TensorFlow] were released. These libraries enable programmers to develop large, multi-layered neural networks with less time and effort, and more efficiency. All these libraries, although varying in functionality and capability, have similar uses in deep machine learning. 


### TensorFlow library

TensorFlow consists of two main components, as follows:
1. Tensors, in which the data is held
2. Flow, referring to the computational graph

[Tensors] can be defined as multi-dimensional arrays. A single number is known as a scalar. More than one number arranged in a one-dimensional list (array) is known as a vector. More than one number arranged in a two-dimensional manner is known as a matrix.Technically speaking, scalars, vectors, and matrices are all tensors.

• Scalars are zero-dimensional tensors.

• Vectors are one-dimensional tensors.

• Matrices are two-dimensional tensors.

However, it is a universally accepted practice that when we have more than one number arranged in three or more dimensions, we refer to such an arrangement as a tensor. We can picture a tensor in the shape of a Rubik’s cube. From the picture, we can see that tensors have a great capacity for data
storage, as they have n dimensions. The n here is used as a proxy for the
actual number of dimensions, where n>=3. To better understand the relationship between scalars, vectors,
matrices, and tensors, we can depict them as shown

As you can see, the four data structures are quite similar to each other notation-wise as well, differing with respect to their capacity. Although tensors usually hold numbers, they can also hold text and strings. Tensors are capable of containing large amounts of data in a compact form. This makes it easier to handle the computation of our program, even when we have enormous amounts of data that we need to use to train our machine.

[Flow] The input of the program is taken in the form of tensors, which are then executed in distributed mode with the help of computational graphs. These graphs are used to set the flow of the entire program. A computational graph is a flowchart of operations and functions that are needed to be carried out on the input tensor. The tensor enters on one side, goes through a list of operations, then comes out the other side as the
output of the code. The Tensorflow Machine Learning Library This is how TensorFlow got its name—the input tensor follows a systematic flow, thus producing the necessary output. Now that we know what TensorFlow is, let’s examine how it is useful to machine learning developers.

#### Applications of TensorFlow
Despite being relatively new, TensorFlow has already served its purpose in several areas of artificial intelligence, and continues to do so. Some of its
applications include the following:

• Image recognition: Identifying objects or features from a photo or a video

• Image classification: Identifying and segregating objects from each other

• Text summarization: Condensing content into a few comprehensible words

• Sentiment analysis: Identifying whether a statement is positive, negative, or neutral

• Speech recognition: Recognizing and translating the spoken word into text

• Other deep learning projects

With TensorFlow, deep learning using neural networks becomes a piece of cake. Hence, most of the library’s applications are focused on this
area of artificial intelligence.

### TensorFlow’s Competitors
TensorFlow, although quite unique in its structure and usage, does have some competitors in the machine learning world. These are alternative frameworks that people use to perform the same functions that TensorFlow does. Some of these libraries include the following: Theano, OpenCV, PyTorch, Apache Spark, Keras. All these libraries, although varying in functionality and capability, have similar uses in machine learning. The Keras library can be used on top of TensorFlow to develop even more effective deep learning models.



