
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


### TensorFlow 2.0 library

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

[Flow] The input of the program is taken in the form of tensors, which are then executed in distributed mode with the help of computational graphs. These graphs are used to set the flow of the entire program. A computational graph is a flowchart of operations and functions that are needed to be carried out on the input tensor. The tensor enters on one side, goes through a list of operations, then comes out the other side as the output of the code. The Tensorflow Machine Learning Library This is how TensorFlow got its name—the input tensor follows a systematic flow, thus producing the necessary output. Now that we know what TensorFlow is, let’s examine how it is useful to machine learning developers. 

Note: TF 1.0 specific example: TensorFlow works with the help of a computational graph. This graph consists of all the variables that we declare, all the operations that we carry out, and so on. It basically works behind the scenes of a program. In TensorFlow, every node of the graph is known as an operation, even if it is just a command that initializes a variable. We will begin by acquiring the “default graph,” like this:`graph = tf.get_default_graph()` Now, let’s try to retrieve the operations from within this graph:`graph.get_operations()`We will get an output like this:`[ ]` This is because we’ve not carried out any operations yet, so the graph has nothing to display. We will now begin adding some nodes to this graph. Let us use some of the simple commands we have learned so far, like the following:

• Creating a constant a

• Creating another constant b

• Finding the sum of a and b as c

• Finding the product of c and a constant as d

```
a = tf.constant(300, name = "a")
b = tf.constant(65, name = "b")
c = tf.add(a, b, name = "c")
d = tf.multiply(c, 10, name = "d")
```

In each line, name is used just for visualization to help us understand the concept of the computational graph. We can give each node any other
name as well, but we have assigned our names to avoid confusion and to facilitate better understanding. Let us now see how our graph looks by entering the following two lines to get the operations from it:
```
operations = graph.get_operations()
operations
```
Executing this gives us the result shown

This shows us the number of nodes present in our graph. We had entered four different nodes, which are displayed here along with their names (a, b, c, and d) and their types (constant, constant, addition, multiplication, respectively)
We can now run any or all of these nodes in a session, as shown below. As you can see, we have executed node a and node e:
```
sess = tf.Session()
with tf.Session() as sess:
     result = sess.run(a, e)
     print result
```     
Here, we have run the session within a with block. This is a method that is used quite often, especially when multiple sessions are required. Instead of declaring the sess variable separately, and then typing the sess.run() command several times, we can just complete the entire process within a single loop. Thus, we can see how the computational graph works. Of course, we won’t necessarily need to develop this kind of program, especially in machine learning. However, in order to grasp the concept of graphs in TensorFlow, it is good to go through this.

Note: TesorFlow 2.0 vs TensorFlow 1.0 : Lazy Execution vs. Eager Execution && @tf.function Decorator && TF1.0: tf.global_variables_initializer(). 

Tf 2.0 Eager Execution
According to the official website, `TensorFlow’s eager execution is an imperative programming environment that evaluates operations immediately, without
building graphs: operations return concrete values instead of
constructing a computational graph to run later.` In other words, iteration occurs at once, and we need not create a computational graph or run separate sessions for each command. It has a natural and steady flow, and does not need to be controlled by a graph. It is intuitive because it ensures that the code follows the correct layout and structure. It also allows us to use regular Python debugging tools to identify and rectify any errors that may exist within the code. This is different from TensorFlow’s original “lazy” execution, where the programmer had to build a graph and run their lines of code within a session.

TensorFlow 1.0 followed lazy execution. It would not execute code immediately. Instead, it would wait for the particular node of the graph to be executed within a session, and only then would it run.

An example is shown next, where we have a code to print “Hello There,” to find the sum of 90 and 7, and to display the value of a variable that is declared to be 300.

TensorFlow 1.0
```
import tensorflow as tf
a = tf.constant("Hello There")
b = 9+70
c = tf.Variable(300)
init_op = tf.global_variables_initializer()
sess = tf.Session()
print(sess.run(init_op))
print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
```
Note: We need to create a session, which can execute an entire graph or a part of the graph. Accordingly, it will allocate resources and accommodate values and results.One very important point to be noted is that, when we are using variables, we need to initialize them before we can begin using them in sessions. This seems like an unnecessary step at first, but when we try to work with our variables without initializing them, our program gives us an error. This init_op is a node, or an operation, that needs to be executed in order to initialize the variables.

TensorFlow 2.0
In TensorFlow 2.0, lazy execution was replaced with eager execution. This means that the code is now executed directly. There is no need to first build a computational graph and then run each node in a session. Each line of code executes immediately.

We can see this here, where we write the same code as before, but using TensorFlow 2.0:
```
import tensorflow as tf
a = tf.constant("Hello There")
b = 9+70
c = tf.Variable(300)
tf.print(a)
tf.print(b)
tf.print(c)
```
As you can see, the first set of code followed a lazy manner of execution, using a distributed graph, while the second set of code did not.
It followed an eager manner of execution instead. The second code is also shorter than the first code, as it doesn’t have so many steps.

@tf.function Decorator
We know that, since TensorFlow 2.0 follows eager execution, there is no need to create a computational graph first, followed by a session to run our program. Does this mean that we can no longer run a program in a distributed manner? Not at all. We can still carry out a distributed execution for our program. All we need to do is write that piece of code in the form of a function, and then use the @tf.function decorator as a prefix to the code. TensorFlow will then understand that the code is meant to be executed in a distributed manner, and it will proceed to do so.

TensorFlow 2.0
```
import tensorflow as tf
x = 7
y = 8
z = 9
@tf.function
def result(x,y,z):
      a = x+y-z
      return a
b = result(x,y,x)
tf.print(b)
b = result(1,7,3)
tf.print(b)
```
As you may have already noticed, the function here is decorated with tf.function, which allows it to be executed like a graph.

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

#### TensorFlow’s Competitors
TensorFlow, although quite unique in its structure and usage, does have some competitors in the machine learning world. These are alternative frameworks that people use to perform the same functions that TensorFlow does. Some of these libraries include the following: Theano, OpenCV, PyTorch, Apache Spark, Keras. All these libraries, although varying in functionality and capability, have similar uses in machine learning. The Keras library can be used on top of TensorFlow to develop even more effective deep learning models.
Note: TensorFlow implemented the Keras API as a powerful tool that can be used for model building. It supports eager execution and several other functionalities of TensorFlow. It is versatile, reliable, and effective in its working. It has been added to TensorFlow 2.0 for this very reason.Keras used to be an independent package on its own, which users would download separately and use within their models. Slowly, TensorFlow added it to its framework, as tf.keras. This tf.keras sub-package was different from the main Keras package, so as to ensure compatibility and stability. Later, with the announcement of TensorFlow 2.0, the TensorFlow team stated that Keras would be the main high-level API of this version.



