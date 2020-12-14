
### AI/ML/DeepML Overview:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-DeepLearning-is-a-subset-of-ML-whithin-AI-sphere.png" width="500">

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


### TensorFlow 1.x/2.x DeepML library

TensorFlow consists of two main components, as follows:
1. Tensors, in which the data is held
2. Flow, referring to the computational graph

[Tensors] can be defined as multi-dimensional arrays. A single number is known as a scalar. More than one number arranged in a one-dimensional list (array) is known as a vector. More than one number arranged in a two-dimensional manner is known as a matrix.Technically speaking, scalars, vectors, and matrices are all tensors.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-a_scalar_vector_and_matrix.png" width="500">

• Scalars are zero-dimensional tensors.

• Vectors are one-dimensional tensors.

• Matrices are two-dimensional tensors.

However, it is a universally accepted practice that when we have more than one number arranged in three or more dimensions, we refer to such an arrangement as a tensor. We can picture a tensor in the shape of a Rubik’s cube. From the picture, we can see that tensors have a great capacity for data storage, as they have n dimensions. The n here is used as a proxy for the actual number of dimensions, where n>=3. To better understand the relationship between scalars, vectors, matrices, and tensors, we can depict them as shown:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-notational_representation_of_a_scalar_vector_matrix_and_tensor.png" width="500">


As you can see, the four data structures are quite similar to each other notation-wise as well, differing with respect to their capacity. Although tensors usually hold numbers, they can also hold text and strings. Tensors are capable of containing large amounts of data in a compact form. This makes it easier to handle the computation of our program, even when we have enormous amounts of data that we need to use to train our machine.

So a tensor is also a mathematical entity with which to represent different properties, similar to a scalar, vector, or matrix. It is true that a tensor is a generalization of a scalar or vector. In short, tensors are multidimensional arrays that have some dynamic properties. A vector is a one-dimensional tensor, whereas two-dimensional tensors are matrices:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Tensors.png" width="500">

Tensors can be of two types: constant or variable.

Rank

Ranking tensors can sometimes be confusing for some people, but in terms of tensors, rank simply indicates the number of directions required to
describe the properties of an object, meaning the dimensions of the array contained in the tensor itself. Breaking this down for different objects, a
scalar doesn’t have any direction and, hence, automatically becomes a rank 0 tensor, whereas a vector, which can be described using only one direction, becomes a first rank tensor. The next object, which is a matrix, requires two directions to describe it and becomes a second rank tensor.

Shape

The shape of a tensor represents the number of values in each dimension.
```
Scalar—32: The shape of the tensor would be [ ].
Vector—[3, 4, 5]: The shape of the first rank tensor
would be [3].
         1 2 3
Matrix = 4 5 6 : The second rank tensor would have a shape of [3, 3]
         7 8 9

```

[Flow] The input of the program is taken in the form of tensors, which are then executed in distributed mode with the help of computational graphs. These graphs are used to set the flow of the entire program. A computational graph is a flowchart of operations and functions that are needed to be carried out on the input tensor. The tensor enters on one side, goes through a list of operations, then comes out the other side as the output of the code. The Tensorflow Machine Learning Library This is how TensorFlow got its name—the input tensor follows a systematic flow, thus producing the necessary output. Now that we know what TensorFlow is, let’s examine how it is useful to machine learning developers. 

So flow is basically an underlying graph computation framework that uses tensors for its execution. A typical graph consists of two entities: nodes and edges, as
shown: (Nodes are also called vertices):

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-typical-graph.png" width="500">

The edges are essentially the connections between the nodes/vertices through which the data flows, and nodes are where actual computation
takes place. Now, in general, the graph can be cyclic or acyclic, but in TensorFlow, it is always acyclic. It cannot start and end at the same node.
Let’s consider a simple computational graph, as shown, and explore some of its attributes.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-computational-graph.png" width="500">

The nodes in the graph indicate some sort of computation, such as addition, multiplication, division, etc., except for the leaf nodes, which
contain the actual tensors with either constant or variable values to be operated upon. These tensors flow through the edges or connections between nodes, and the computation at the next node results in formation of a new tensor. So, in the sample graph, a new tensor m is created through a computation at the node using other tensors x and y. The thing to focus on in this graph is that computations take place only at the next stage after leaf nodes, as leaf nodes can only be simple tensors, which become input for next-node computation flowing through edges. We can also represent the computations at each node through a hierarchical structure. The nodes at the same level can be executed in parallel, as there is no interdependency between them. In this case, m and n can be calculated
in parallel at the same time. This attribute of graph helps to execute computational graphs in a distributed manner, which allows TensorFlow to be used for large-scale applications.

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

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-operations-within-a-computation-graph.png" width="500">


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

### Neural Networks
The neural network, or artificial neural network, was inspired by and modeled after the biological neural network. These networks, like the human brain, learn to perform specific tasks without being explicitly programmed. A neural network is composed of a series of neurons that are connected together to form a type of network, hence the name neural network. A neuron, or an artificial neuron, is the fundamental unit of a neural network. It is a mathematical function that replicates the neurons in the human brain, as you can see 


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-a-biological-neuron-and-an-artificial-neuron.png" width="500">

Comparison of biological and artificial neurons.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-comparison-of-a-biological-and-an-artificial-neuron.png" width="500">


#### Working of an Artificial Neuron (Perceptron)
The perceptron follows a particular flow of steps in order to achieve its desired output. Let’s go through these steps one by one to understand how
a perceptron works.

Step 1: Accepting Inputs
The perceptron accepts inputs from the user in the form of digital signals provided to it. These inputs are the “features” that will be used for training the model. They are represented by x(n), where n is the number of the feature. These inputs are then fed to the first layer of the neural network through a process called forward propagation.

Step 2: Setting the Weights and Bias
Weights: The weights are calculated and set while training the model. They are represented by w(n), where n is the number of the weight. For example,
the first weight will be w1, the second weight will be w2, and so on.
Bias: The bias is used to train a model with higher speed and accuracy. We generally represent it with w0.

Step 3: Calculating the Net Input Function
The equation for the net input function is as follows: `I = Sum(x(n).w(n) + w0)` Thus, each input feature is multiplied by its corresponding weight, and
the sum of all these products is taken. Then, the bias is added to this result.
The Perceptron Learning Rule: According to this rule, the algorithm automatically determines the optimum values for the weights. The input features are then multiplied by these weights in order to determine if the perceptron should forward the signal or not. The perceptron is fed with several signals, and if the resultant sum of these signals exceeds a particular threshold, it either returns an output signal or doesn’t.

Step 4: Passing the Values Through the Activation Function
The activation function helps with providing nonlinearity to the perceptron. There are three types of activation functions that can be used:
ReLU, Sigmoid, and Softmax.

-ReLU
The Rectified Linear Unit is used to eliminate negative values from our outputs. If the output is positive, it will leave it as it is. If the output is negative, it will display a zero.

-Sigmoid
It is a special mathematical function that produces an output with a probability of either 1 or 0.

-Softmax
It is generally used in the final layer of a neural network. It is generally used to convert the outputs to values that, when summed up, result in 1. Thus,
these values will lie between 0 and 1.

Note: The most common practice is to use a ReLU activation function in all the hidden layers, and then to use either a Softmax activation function (for multi-class classification) or Sigmoid activation function (for binary classification).

Step 5: Producing the Output
The final output is then passed from the last hidden layer to the output layer, which is then displayed to the user. Now that we know how a perceptron works, let’s go a little more in depth as to how a neural network performs a deep learning task.

#### Digging Deeper into Neural Networks
Deep learning goes a step further in machine learning. It allows the machine to begin thinking on its own in order to make decisions and carry out certain tasks. Neural networks are used to develop and train deep learning models. For example, consider a very simple neural network, which consists of an input layer, an output layer, and one layer of neurons, known as the hidden layer (as shown). 

The basic function of these three sections is as follows:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Basic-neural-network.png" width="400">

1. The input layer, as the name implies, is made of the input signals that will be further transmitted into the neural network.

2. The hidden layer is where all the important computations occur. The input that is fed to it is taken, calculations are performed on it, and then this input is sent to the next layer, which is the output layer. The hidden layer can have any number of neurons within it. There can also be more than one hidden layer, depending on our requirements and arrangement.

3. The output layer, as the name suggests, contains the output signals. These are nothing but the final results of all the calculations performed by the hidden layer/s.

So Artificial Neural Networks (ANNs) tries to mimic the brain at its most basic level, i.e., that of the neuron. An artificial neuron has a similar structure to that of a human neuron and comprises the following sections:

Input layer: This layer is similar to dendrites and takes input from other networks/neurons.

Summation layer: This layer functions like the soma of neurons. It aggregates the input signal received.

Activation layer: This layer is also similar to a soma, and it takes the aggregated information and fires a signal only if the aggregated input crosses a certain
threshold value. Otherwise, it does not fire.

Output layer: This layer is similar to axon terminals in that it might be connected to other neurons/networks or act as a final output layer (for predictions).

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Artificial-neural-network.png" width="400">

```
In the preceding figure, X 1 , X 2 , X 3 ,.........X n are the inputs fed to the neural network. W 1 , W 2 , W 3 ,............W n are the weights 
associated with the inputs, and Y is the final prediction. Many activation functions can be used in the activation layer, to convert all the linear 
details produced at the input and make the summation layer nonlinear. This helps users acquire more details about the input data that would not 
be possible if this were a linear function. Therefore, the activation layer plays an important role in predictions. Some of the most familiar 
types of activation functions are sigmoid, ReLU, and softmax as explained above.

```

Simple Neural Network Architecture : a typical neural network architecture is made up of an Input layer, Hidden layer, Output layer

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Simple-neural-network-architecture_regression.png" width="600">

```
Every input is connected to every neuron of the hidden layer and, in turn, connected to the output layer. If we are solving a 
regression problem, the architecture looks like the one shown in above picture, in which we have the output Y p, 
which is continuous if predicted at the output layer. If we are solving a classification (binary, in this case),
we will have the outputs Y class1 and Y class2 , which are the probability values for each of the binary 
classes 1 and 2 at the output layer, as shown bellow.
```

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Simple-neural-network-architecture_classifiacation.png" width="600">

Deep Neural Networks (DNNs): When a simple neural network has more than one hidden layer, it is known as a deep neural network (DNN). Architecture of a
typical DNN:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Deep-neural-network-with-three-hidden-layers.png" width="600">

It consists of an input layer with two input variables, three hidden layers with three neurons each, and an output layer (consisting either of a single output for regression or multiple outputs for classification). The more hidden layers, the more neurons. Hence, the neural network is able to learn the nonlinear (non-convex) relation between the inputs and output.


The Process

There are four main steps to the neural network process that allow it to come up with the most optimal solution for any problem that is given to it.

Step 1: The numerical input signals are passed into the neural network’s hidden layers.

Step 2: The net input function is calculated with the weights and the bias that are generated during the training.

Step 3: The activation function is applied to the net input function.

Step 4: The result is then produced as the output of the neural network.

Thus, deep learning, as a part of machine learning, stands out as an extremely useful technique in the area of artificial intelligence.

Types of Neural Networks

There are several types of neural networks, all based on their structure, composition, and flow. Let’s go ahead and discuss a few of the common
and most important ones that are used by deep learning developers.

-Single-Layer Neural Networks: A Perceptron
The perceptron is the oldest single-layer neural network. As you have seen before, it takes the input from the user, multiplies it by the corresponding weight, adds that to the bias to get the net input function, and then passes the result through the activation function to get the final output. Every perceptron produces only a single output. This type of neural network is not very efficient due to its extremely limited complexity. Thus, researchers came up with a model that contained more than one layer of perceptrons.

-Multi-Layer Neural Networks
This type of neural network is used mainly for natural language processing, speech recognition, image recognition, etc. It consists of two or more
layers of perceptrons, as follows:
• Input layer: This is all the available numerical data that is fed into the system and then transferred to the rest of the neural network.
• Hidden layers: This is where all the neurons are located. Every layer can have any amount of neurons. They are known as “hidden” layers because they remain hidden within the neural network as they perform the necessary computations.
• Output layer: This is the final result of all the calculations that happened in the hidden layers

-Convolutional Neural Networks
Convolutional neural networks follow the same principle as multi-layer neural networks, the only difference being that they include “convolutional
layers,” which make use of filters. A filter is a grid of size AxB that is moved across the image and gets multiplied several times by it to produce a new value. Each value represents a line or an edge in the image. Once the filters have been used on the image, its important characteristics can be extracted. This is done with the help of a pooling layer. These layers pool or collect the main features of each image. One popular technique of doing this is known as max pooling, which takes the largest number of each image and stores it in a separate grid. It thus compresses the main features into a single image and then passes it on to a regular multi-layer neural network for further processing. These neural networks are mainly used for image classification. They can also be used in search engines and recommender systems.

-Recurrent Neural Networks
Recurrent neural networks (RNNs) are used for temporal data; i.e., data that requires past experiences to predict future outcomes. State matrices remember previous states of data by storing the last output, and then use this data to calculate the new output.two states: long term and short term. RNNs begin in the layers after the first layer. Here, each node acts as a memory cell during the computation, which allows it to compare previous values with new values during
back propagation. These neural networks can be used for stock market predictions, natural language processing, and price determination.

-Sequence-to-Sequence Models
A sequence-to-sequence model is mainly used when the lengths of the input data and output data are unequal. It makes use of two recurrent neural networks, along with an encoder and a decoder. The encoder processes the input data, while the decoder processes the output data. These models are usually used for chatbots and machine translation.

-Modular Neural Networks
Modular neural networks have several different networks that each work independently to complete a part of the entire task. These networks are not
connected to each other, and so do not interact with each other during this process. This helps in reducing the amount of time taken to perform the computation by distributing the work done by each network. Each sub- task would require only a portion of the total time, power, and resources needed to complete the work.

Types of Data

The data that is collected and used can be either of the following:

• Labeled: Each class/type is labeled based on certain characteristics so that the machine can easily identify and separate the data into its respective groups. For example, if you have a collection of pictures that are separated and tagged as “cat” or “fish” accordingly.

• Unlabeled: Each class/type is not labeled, and so the machine needs to figure out how many classes are there and which item belongs where, and then it must
separate the data on its own. For example, if you have a set of pictures, but they are not separated and tagged as “cat” or “fish” accordingly. In this case, the machine would need to identify some particular features that differentiate one animal from the other (like a cat’s whiskers or a fish’s fins).

Based on the kind of data being used, there are two main types of machine learning methods:

• Supervised learning: This method uses labeled data.

• Unsupervised learning: This method uses unlabeled data

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Supervised_Unsupervised_Learning_diff.png" width="500">

A typical supervised machine learning architecture: 


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-supervised-ML-architecture.png" width="500">

```
Note: Within supervised learning, if we are to predict numeric values, this is called regression, whereas if we are to predict 
classes or categorical variables, we call that classification. For example, if the aim is to predict the sales (in dollars) a 
company is going to earn (numeric value), this comes under regression. If the aim is to determine whether a customer will 
buy a product from an online store or to check if an employee is going to churn or not (categorical yes or no), this is a 
classification problem. Classification can be further divided as binary and multi-class. Binary classification deals 
with classifying two outcomes, i.e., either yes or no. Multi-class classification yields multiple outcomes. For example, 
a customer is categorized as a hot prospect, warm prospect, or cold prospect, etc.
```

Each type of learning method has various types of algorithms that can be used to solve a machine learning problem. Let’s take a look at some important ones.

Supervised Learning Algorithms

The goal of every supervised learning algorithm is to map the input to the output, as shown in the following equation:
`y = f(x)`

There are several algorithms that can be used to solve a machine learning problem with the help of supervised learning. These algorithms can be segregated into the following categories:

1. Regression algorithms: These algorithms contain outputs that are real or countable. For example, height (4 feet, 5 feet, 6 feet), age (27, 31, 65), or price
(100 rupees, 20 pounds, 10 dollars)

2. Classification algorithms: These algorithms contain outputs that are abstract or categorical. For example, colors (orange, purple, turquoise), emotions (happy, sad, angry), or gender (girl, boy). To give you some idea of what these algorithms are, let’s go through
three common types of algorithms that are used:

- Linear regression
- Logistic regression
- K-Nearest neighbors

Applications of Supervised Learning Algorithms

1. Spam detection

2. Bioinformatics: This is the method of keeping a record of a person’s biological information for later use. One of the most common examples of this is the
security system on our cell phones, which can scan our fingerprint and grant us access accordingly.

Unsupervised Learning Algorithms

The goal of unsupervised learning algorithms is to discover possible patterns from the set of data that is provided. The algorithm has no prior information about the patterns and labels present in the data. There are several algorithms that can be used to solve a machine learning problem with the help of unsupervised learning. These algorithms can be segregated into the following categories:

1. Cluster analysis: This approach finds similarities among the data and then groups the common data together in clusters.

2. Dimensionality reduction: This approach attempts to reduce the complexity of data while still keeping the data relevant.

Two common algorithms that are used for unsupervised learning: K-means clustering and principal component analysis.

- KMeans Clustering

- Principal Component Analysis

Applications of Unsupervised Machine Learning Algorithms

Anomaly detection is the identification of certain anomalies or observations that are different from the rest of the observations. These anomalies are also
called outliers. For example, credit card fraud can be discovered by detecting unusual transactions made with the credit card. Association is the process of identifying associations between different observations with the help of provided data. For example, in e-commerce it is easy to figure out the type of products a customer might be interested in by analyzing previous purchases.

Note: Do a little more research on machine learning algorithms. You can even compare them with each other, as this will broaden your understanding of these algorithms to help you decide which one to use for any future projects you might have.

Apart from supervised and unsupervised machine learning, there are also two lesser-known methods of machine learning, as follows:

• Semi-supervised learning: This method uses some labeled data and a larger proportion of unlabeled data
for training.

• Reinforcement learning: This method is similar to training a pet. It sends positive signals to the machine when it gives the desired output, to let it know that it is right and to help it learn better. Similarly, it sends negative signals to a machine if it provides an incorrect output.


### Machine Learning Programming with Tensorflow 2.0

#### Structure of a Machine Learning Model
Machine learning, as mentioned earlier, requires part of the work to be done by us. The rest of it is all done behind the scenes by the computer. In other words, it all happens in the backend of the code. This, in all honesty, saves us, as programmers, a lot of trouble. There are, however, still plenty of tasks that we need to carry out while creating our model in order to make sure that we get the output we desire. A machine learning developer’s task is mainly to build the model and
then run it. There are several components to this model, depending on what exactly we are trying to accomplish, but the general architecture remains the same. Since we will be using neural networks to carry out our machine learning processes, we will study the structure of a deep learning model that uses a neural network. The overall idea for the structure of the model is as shown


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TenmsorFlow-Flowchart-of-a-ML-model.png" width="400">

As we can see in the flowchart, there are four main steps involved in developing a working machine learning model, as follows:

1. Data loading and pre-processing: This part accepts data, manipulates it, then prepares it for training and testing.

2. Building the model: This is the part where the developer specifies the various components of the model.

3. Training the model: This part takes the training data and begins performing calculations on it to get an optimum result.

4. Testing the model: This part validates or checks the accuracy of the model. The first two steps require the time, effort, and skills of a programmer,
since they involve the handling of data and the creation of a working model. For the last two steps, all the programmer has to do is set the model
running and then kick back and relax while the machine does all the hard work. Let’s go through this structure in a little more detail to get a better idea
of what it does, how it works, and what needs to be done.

##### Data Loading and Pre-Processing
We had a look at the different methods of collecting data. We also learned that this data requires some pre-processing before it can be used for any kind of analysis in order to ensure optimal results. This means that we might need to add, remove, or change some values. Now remember, this does not mean that we are completely changing our data, which can result in incorrect outputs. We are just making it more readable for our system to take and work with.

Data can be altered manually. Applications like spreadsheets or visualization software come in handy when working with structured data. However, when the dataset is huge, it becomes quite tiring and monotonous to work with. Thus, most developers use a Python library called Pandas, which provides users with several tools and utilities to work on their data. With the help of Pandas, users can import a .csv file (csv: comma separated values) from their local system into a Jupyter notebook. We will be using image datasets that are already integrated within the TensorFlow library. They can easily be called with the help of a TensorFlow function, as we will see later on.

The data that we use for training machine learning models is divided into two categories: labels and features.

-Labels: These are the components of the data that are to be predicted; i.e., the dependent variable or output. They are determined based on the features provided to the system.
-Features: These are the components of the data that are used for prediction; i.e., the independent variable or input. They determine the labels of the outputs. When choosing features, it is important to ensure that they are independent and distinct.

When training a deep learning model, we can choose either of the following methods based on how we intend to input our features and labels:

• Supervised learning: We feed the model with the features and the labels.

• Unsupervised learning: We feed the model with the features only.

• Semi-supervised learning: We feed the model with some labeled features and some unlabeled features.

Note The quality of the labels is proportional to that of the features. In other words, better features result in more accurate labels.

Once we have finished altering our data, we need to split it into two parts: the training data and the test data.

-Training data: Training data is what is fed into the model to be used while it is training. This will generally be a greater proportion of the data, since
the model requires a larger amount of data when training to get more accurate results.

-Test data: Test data is what is fed into the model after it has finished training and settled on optimal parameters. This will generally be a lesser
proportion because it is only meant to help the model determine how accurate or inaccurate its prediction is. After we are done pre-processing the data, the next step is to build the model.

##### Building the Model
We need to develop the architecture of our machine learning model. In this case, we will be using a neural network. Thus, we need to arrange the neural network by defining the following:
• The number of hidden layers

• The number of neurons in each layer

• The weights and biases

• The activation function

We learned how neural networks work, and we studied heir different types. For example, convolutional neural networks (CNNs) are best used for image classification and recognition, and recurrent neural networks (RNNs) are great for machine translation and speech recognition. We can choose our preferred neural network after
careful consideration of our data, resources, and desired outcome, and accordingly build the model that we require.

##### Training the Model
Once the model is built, it is ready to be trained. This is where the programmer steps aside and gives way to the machine, which proceeds to do some intense work. All we need to do here is call the training data into the model and then start the process. During training, the model begins trying out different values and
replacing the parameters, i.e., the weights and the biases, in order to come up with the most suitable equation that will give high accuracy and low error. It follows a trial-and-error manner, and keeps changing the values of the parameters until it gets a result that is satisfactory.

##### Testing the Model
Once we have our trained model, we need to feed the test data into it. We then allow the model to run this data through to see how accurate its predictions are. In this way, we validate the model. Depending on this accuracy, we can decide if we want to change certain aspects of the model and then retrain it, or leave it as it is. Of course, there are several other factors that can affect this decision as well, including time, computational power, and so on. For example, the programmer may not have enough resources to redesign and retrain the model. Or perhaps there isn’t enough time. So, before retraining the
model, the programmer must take all of these factors into consideration. The machine continues to repeat this cycle of training and testing the
model until it produces an acceptable outcome. The structure of a machine learning model can vary greatly with regard to more specific factors, depending on the type of problem that we are solving. Hence, as mentioned earlier, we need to correctly define our problem and the solution we hope to achieve, and then carefully plan out our model to minimize error.


#### Classification
We have two main types: binary classification and multi-class classification.

- Binary Classification

This is a very simple type of classification problem. Here, the variable to be predicted can take either one of two possible values. In other words, the data needs to be split into two groups. Let’s take a very simple example. Suppose we have a set of nine random numbers available to us: 2, 5, 700, 75654, 8273, 9, 23, 563, and 0. We can separate these numbers into two groups:
```
Odd Numbers (5, 8273, 9, 23, 563)
Even Numbers (2, 700, 75654, 0)
```
As you can see, we have two groups or “classes” here based on the type of number. Five of the given numbers are odd, and four of them are even. Let’s take another example. Suppose we have a set like this: “doe,” “ram,” stag,” “ewe,” “rooster,” “chicken.” This can be separated out into the following:
```
Male (“ram,” “stag,” “rooster”)
Female (“doe,” “ewe,” “chicken”)
```
Once again, here we have two categories based on their gender, male and female, each having three variables. Each variable within the set of
data is divided accordingly. Other more advanced applications of binary classification include cancer detection (cancer present/cancer absent), spam detection (spam/not spam), etc.

- Multi-class Classification

This is also called multinomial classification. Here, the variable to be predicted can take one of many possible values. In other words, the data needs to be separated into more than two groups. For example, suppose we have a set like this: “rose,” “cucumber,” “tulip,” “lily,” “apple,” “carrot,” “potato,” “orange,” “sunflower.” We can separate them into these groups:
```
Flowers (“rose,” “tulip,” “lily,” “sunflower”)
Fruits (“cucumber,” “apple,” “orange”)
Vegetables (“carrot,” “potato”)
```
As you can see, we have three groups into which the data is divided based on type: four of the variables are flowers, three of them are fruits, and two of them are vegetables. Let’s consider another example. Take a look at this set of eleven random numbers: 9, 55, 8, 22, 27, 16, 205, 93, 4, 49, 81. We can divide them into the following groups:
```
Multiples of 2 (8, 22, 16, 4)
Multiples of 3 (9, 27, 93, 81)
Multiples of 5 (55, 205)
Multiples of 7 (49)
```
We have four groups here based on the highest common factor (2, 3, 5, or 7): the multiples of 2 consisting of four variables, multiples of 3 consisting of four variables, multiples of 5 consisting of two variables, and multiples of 7 consisting of one variable.

Other more advanced applications of multi-class classification include eye-color recognition (blue, green, light brown, dark brown, grey), cat-
breed identification (Persian, Munchkin, Bengal, Siamese, Sphynx), etc. As we can see, in all these classification examples, the variables were grouped together depending on the characteristics that they shared. In this way, data can be classified or grouped based on similarities in particular
characteristics or features. 

### Programming with TensorFlow 2.0
The programs that we will be learning comprise image classification problems. Before we get into them, let’s have a quick look at how such problems need to be dealt with in order to solve them.

#### Image Classification: An Overview
Image classification is one of the most popular areas of deep learning due to its vast usability in practical purposes. It is the process of separating images within a dataset into groups, based on their similar features.For example, suppose we had images of a goldfish, a grasshopper, a sparrow, a rabbit, a penguin, a cat, a vulture, and a shark.We thus have the following four classes:

• Insect (grasshopper)

• Fish (goldfish, shark)

• Mammal (cat, rabbit)

• Bird (vulture, penguin, sparrow)

Having studied these subjects in school, we already know which of these creatures falls under which category. We can use our natural intelligence to distribute the images easily. But how would an artificially intelligent computer figure this out? We would have to train it to understand the ways in which some of the creatures relate to each other, while others don’t. The model can be trained by feeding it with labeled pictures of different kinds of creatures. The labels would inform the machine if the image is that of an animal, a bird, a fish, or an insect. The machine would then begin to observe all the images under a single class to gather
information on any kind of common features among them.

For example:

• The insects have six legs and antennae.

• The fish have streamlined bodies and fins.

• The mammals have four legs and furry bodies.

• The birds have wings and two legs each.

Once it has gathered its observations and made predictions that are verified to be accurate, it can be used for further problem solving. Now, if we give it the eight images, it would solve the problem effortlessly and classify the images according to their type by studying each picture, finding its closest possible label match, and placing it in that class. This is how image classification is done using a machine learning model.

In the programs that we will be going through, we will focus on instructing the computer to train and test similar image classification models with the help of neural networks. 


#### JupyterLab/Jupyter Notebooks environment


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


#### Programs/Notebooks (examples)

Program 1: Image Classification Using a Pre-Trained Model

Import TensorFlow and Keras utilities into the notebook.
```
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

```
Load the model into the notebook.
```
# ResNet50() or VGG16() 
model = ResNet50()
```
Load an image into the notebook.
```
image = load_img('Kitten1.jpg', target_size=(224, 224))
image
```
Note: Upload Kitten1.jpg into Jupyter Notebook it in the same folder as the Jupyter notebook that you are working in.

Prepare the image for the model.
```
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1],
image.shape[2]))
image = preprocess_input(image)
```
Make the prediction.
```
result = model.predict(image)
label = decode_predictions(result)
label = label[0][0]

```
Display the classification.
```
print('%s (%.2f%%)' % (label[1], label[2]*100))

```
The output will come like this:
Egyptian_cat (87.00%)

The ResNet50 model also gives the prediction: 87.00% percent sure of its answer.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TesorFlow-e1.png" width="500">


Program 2: Handwriting Recognition Using Keras in TensorFlow (Single Layer, Multi-class)

Import TensorFlow into your kernel.
```
import tensorflow as tf
```
Load the MNIST dataset.
```
data = tf.keras.datasets.mnist
(ip_train, op_train), (ip_test, op_test) = data.load_data()
```
Prepare the data.
```
ip_train, ip_test = ip_train / 255.0, ip_test / 255.0
```
Build the neural network.
```
model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape = (28,28)),
     tf.keras.layers.Dense(10, activation = 'softmax')
])
```

Compile the model.
```
model.compile(optimizer = 'adam',
                         loss = 'sparse_categorical_crossentropy',
                         metrics = ['accuracy'])
```
View the model.
```
model.summary()
```
Train the model.
```
model.fit(ip_train, op_train, epochs = 6)
```
Test the model.
```
model.evaluate(ip_test, op_test)
```
Carry out inference.
```
import matplotlib.pyplot as plt
%matplotlib inline
test_image=ip_test[9999]
plt.imshow(test_image.reshape(28,28))
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1,28,28)
result = model.predict(test_image)
result
np.around(result)
array([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32)
(np.around(result)).argmax()

```
This gives the output like this:

6

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-e2.png" width="500">


Program 3: Clothing Classification Using Keras in TensorFlow (Multi-layer, Multi-class)

In a new Jupyter notebook, import the TensorFlow library and
Keras utilities.
```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
```
Load the Fashion MNIST dataset.
```
data = datasets.fashion_mnist
(ip_train, op_train), (ip_test, op_test) = data.load_data()
```
Check the shape of the images.
```
print(ip_train.shape, ip_test.shape)
```
Step 4: Reshape the input values.
```
ip_train = ip_train.reshape((60000, 28, 28, 1))
ip_test = ip_test.reshape((10000, 28, 28, 1))
print(ip_train.shape, ip_train.shape)

```
Prepare the data.

```
ip_train, ip_test = ip_train / 255.0, ip_test / 255.0
```
Build the neural network.
```
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```
Compile the model.
```
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```
View the model.
```
model.summary()
```
Train the model.
```
model.fit(ip_train, op_train, epochs = 5)
```
Test the model.
```
model.evaluate(ip_test, op_test, verbose = 2)
```
Carry out inference.
```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
import matplotlib.pyplot as plt
%matplotlib inline
test_image=ip_test[5000]
plt.imshow(test_image.reshape(28,28))
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1, 28, 28, 1)
result = model.predict(test_image)
result
np.around(result)
n=(np.around(result)).argmax()
print(n)
```
This gives us the following output:

2

This output is very vague. All it tells us is the position of the predicted
class, but not what the actual item of clothing is. Thus, we add an extra line
of code:
```
print(class_names[n])
```
This will give us the following output:

Pullover

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-e3.png" width="500">


Program 4: Clothing Classification Using Convolutional Neural Networks (Multi-layer, Multi-class)
```
Import the TensorFlow library and Keras utilities.
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
```
Load the Fashion MNIST dataset.
```
data = datasets.fashion_mnist
(ip_train, op_train), (ip_test, op_test) = data.load_data()
```
Check the shape of the images.
```
print(ip_train.shape, ip_test.shape)
```
Reshape the input values.
```
ip_train = ip_train.reshape((60000, 28, 28, 1))
ip_test = ip_test.reshape((10000, 28, 28, 1))
print(ip_train.shape, ip_test.shape)
```
Prepare the data.
```
ip_train, ip_test = ip_train / 255.0, ip_test / 255.0
```
Build the convolutional neural network.
```
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation="relu", input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation="relu"))
```
Add the final dense layer and output layer.
```
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```
Compile the model.
```
model.compile(optimizer = 'adam',
                         loss = 'sparse_categorical_crossentropy',
                         metrics = ['accuracy'])
```                         
View the model.
```
model.summary()
```
Train the model.
```
model.fit(ip_train, op_train, epochs = 5)
```
Test the model.
```
model.evaluate(ip_test, op_test, verbose = 1)

```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-e4.png" width="500">

Program 5: Handwriting Recognition Using Convolutional Neural Networks (Multi-layer, Multi-class)  

Import the TensorFlow library and Keras utilities.
```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
```
Load the MNIST dataset.

```
data = datasets.mnist
(ip_train, op_train), (ip_test, op_test) = data.load_data()
```
Reshape the input values.
```
ip_train = ip_train.reshape((60000, 28, 28, 1))
ip_test = ip_test.reshape((10000, 28, 28, 1))
print(ip_train.shape, ip_test.shape)
```
Prepare the data.
```
ip_train, ip_test = ip_train / 255.0, ip_test / 255.0
````
Build the convolutional neural network.
```
model=models.Sequential()
model.add(layers.Conv2D(30,(3,3), activation="relu", input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(60,(3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(90,(3,3), activation="relu"))
```
Add the final dense layer, dropout layer, and output layer.
```
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
```
Compile the model.
```
model.compile(optimizer = 'adam',
                         loss = 'sparse_categorical_crossentropy',
                         metrics = ['accuracy'])
```
View the model.
```
model.summary()
```
Train the model.
```
model.fit(ip_train, op_train, epochs = 5)
```
Test the model.
```
model.evaluate(ip_test, op_test, verbose = 1)
````
Carry out inference.
```
import matplotlib.pyplot as plt
%matplotlib inline
test_image=ip_test[180]
plt.imshow(test_image.reshape(28,28))
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1,28,28,1)
result = model.predict(test_image)
result
np.around(result)
(np.around(result)).argmax()
```
We will get the output as 1. This shows that the model has correctly
predicted the class of the image.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-e5.png" width="500">


Program 6: Image Classification for CIFAR-10 Using Convolutional Neural Networks (Multi-layer, Multi-class)

```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.datasets import cifar10
(ip_train, op_train), (ip_test, op_test) = cifar10.load_data()
print(ip_train.shape, ip_test.shape)
ip_train = ip_train.reshape(ip_train.shape[0], 32, 32, 3)
ip_test = ip_test.reshape(ip_test.shape[0], 32, 32, 3)
ip_train, ip_test = ip_train / 255.0, ip_test / 255.0
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation="relu", input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer = 'adam',
                         loss = 'sparse_categorical_crossentropy',
                         metrics = ['accuracy'])
model.summary()
model.fit(ip_train, op_train, epochs = 10)
model.evaluate(ip_test, op_test, verbose = 2)
import matplotlib.pyplot as plt
%matplotlib inline
test_image=ip_test[20]
plt.imshow(test_image.reshape(32,32,3))
import numpy as np
from tensorflow.keras.preprocessing import image
classes = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1,32,32,3)
result = model.predict(test_image)
result
np.around(result)
n=(np.around(result)).argmax()
print(classes[n])
```
<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-e6-2.png" width="500">



Program 7: Dogs vs. Cats Classification Using Convolutional Neural Networks (Multi-layer, Binary)

Import all the required libraries and functions into Jupyter Notebook.
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

import os
```
Develop the CNN model and compile it. Here, we refer to our model as classifier. We can put any name we
want, provided it is easily understood by anyone who reads it.
```
classifier = Sequential()
classifier.add(Conv2D(64,(3,3),input_shape = (64,64,3),
activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =
['accuracy'])
```
Transform the imported data.
```
from tensorflow.keras.preprocessing.image import
ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
```
Download the dataset.
The dataset needs to be downloaded from a specific link. In this
section of code, we tell our program to download the data from the given
url, and then we store it on our system.
```
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_
dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
```
Set up the directories.
We need to set up different directories for the training and testing data,
and then separate out the cat and dog images accordingly.
```
trainingdir = os.path.join(data_path, 'train')
testingdir = os.path.join(data_path, 'validation')
# directory with the training cat pictures
cats_train = os.path.join(trainingdir, 'cats')
# directory with the training dog pictures
dogs_train = os.path.join(trainingdir, 'dogs')
# directory with the testing cat pictures
cats_test = os.path.join(testingdir, 'cats')
# directory with the testing dog pictures
dogs_test = os.path.join(testingdir, 'dogs')
```
Find the number of elements in each directory.
```
cats_train_num = len(os.listdir(cats_train))
dogs_train_num = len(os.listdir(dogs_train))
cats_test_num = len(os.listdir(cats_test))
dogs_test_num = len(os.listdir(dogs_test))
train_tot = cats_train_num + dogs_train_num
test_tot = cats_test_num + dogs_test_num
print(cats_train_num)
print(dogs_train_num)
print(cats_test_num)
print(dogs_test_num)
print(train_tot)
print(test_tot)
```
Load the training data and testing data, and display the label
map.
```
train_data = train_datagen.flow_from_directory(batch_size=128,
                                                           directory=trainingdir,
                                                           target_size=(64, 64),
                                                           class_mode='binary')
`
test_data = test_datagen.flow_from_directory(batch_size=128,
                                                              directory=testingdir,
                                                              target_size=(64, 64),
                                                              class_mode='binary')
                                                              
```
```
label_map = (train_data.class_indices)
print(label_map)
```
Train the model.
```
classifier.fit(
        train_data,
        epochs=30,
        validation_data=test_data)
```
Carry out inference.
```
import numpy as np
from tensorflow.keras.preprocessing import image
test_image_1= image.load_img('Dog.jpeg', target_size = (64,64))
test_image_2= image.load_img('Cat.jpeg', target_size = (64,64))
test_image_1
test_image_2
test_image_1 = image.img_to_array(test_image_1)
test_image_2 = image.img_to_array(test_image_2)
test_image_1 = test_image_1.reshape(1,64,64,3)
test_image_2 = test_image_2.reshape(1,64,64,3)
result1 = model.predict(test_image_1)
result2 = model.predict(test_image_2)
print(result1, result2)
if result1 == 1:
    prediction1 = 'dog'
else:
   prediction1 = 'cat'
print(prediction1)
if result2 == 1:
    prediction2 = 'dog'
else:
   n2 = 'cat'
print(prediction2)
```


### TensorFlow Models in Production
#### Python-Based Model Deployment: Deploying a Machine Learning Model As a REST Service (Flask + joblib + Pickle)

- Saving and Restoring a Machine Learning Model

Saving any model is also known as serialization. This can also be done in different ways, as Python has its own way of persisting a model, known as pickle. Pickle can be used to serialize machine language models, as well as any other transformer. The other approach has the built-in functionality of sklearn, which allows saving and restoring of Python-based machine learning models. In this section, we will focus on using the joblib function to save and persist sklearn models.
Once the model is saved on disk or at any other location, we can reload or restore it back, for making predictions on new data. In the example below, we consider the standard data set for building a linear regression model. The input data has five input columns and one output column. All the variables are numeric in nature, so little feature engineering is required. Nevertheless, the idea here is not to focus on building a perfect model but to build a baseline model, save it, and then restore it. In the first step, we load the data and create input and output feature variables (X,y).


Jupyter Notebook (Note: running not inside k8s in this example only to create/pickle/test the model)

Jupyter Notebooks are a browser-based (or web-based) IDE (integrated development environments)

Build custom JupyterLab docker image and pushing it into DockerHub container registry.
```
$ cd ./jupyterlab
$ docker build -t jupyterlab-eth .
$ docker tag jupyterlab-eth:latest davarski/jupyterlab-eth:latest
$ docker login 
$ docker push davarski/jupyterlab-eth:latest
```
Run Jupyter Notebook

```
$ sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/k3s-config-jupyter
$ sed -i "s/127.0.0.1/192.168.0.101/" ~/.kube/k3s-config-jupyter
$ docker run --rm --name jl -p 8888:8888 \
   -v "$(pwd)":"/home/jovyan/work" \
   -v "$HOME/.kube/k3s-config-jupyter":"/home/jovyan/.kube/config" \
   --user root \
   -e GRANT_SUDO=yes \
   -e JUPYTER_ENABLE_LAB=yes -e RESTARTABLE=yes \
   davarski/jupyterlab-eth:latest
```
Example:
```
$ docker run --rm --name jl -p 8888:8888 \
>    -v "$(pwd)":"/home/jovyan/work" \
>    -v "$HOME/.kube/k3s-config-jupyter":"/home/jovyan/.kube/config" \
>    --user root \
>    -e GRANT_SUDO=yes \
>    -e JUPYTER_ENABLE_LAB=yes -e RESTARTABLE=yes \
>    davarski/jupyterlab-eth:latest

Set username to: jovyan
usermod: no changes
Granting jovyan sudo access and appending /opt/conda/bin to sudo PATH
Executing the command: jupyter lab
[I 21:37:15.811 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 21:37:16.594 LabApp] Loading IPython parallel extension
[I 21:37:16.614 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 21:37:16.614 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[W 21:37:16.623 LabApp] JupyterLab server extension not enabled, manually loading...
[I 21:37:16.638 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 21:37:16.638 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 21:37:16.639 LabApp] Serving notebooks from local directory: /home/jovyan
[I 21:37:16.639 LabApp] The Jupyter Notebook is running at:
[I 21:37:16.639 LabApp] http://(e1696ffe20ab or 127.0.0.1):8888/?token=f0c6d63a7ffb4e67d132716e3ed49745e97b3e7fa78db28d
[I 21:37:16.639 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 21:37:16.648 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-17-open.html
    Or copy and paste one of these URLs:
        http://(e1696ffe20ab or 127.0.0.1):8888/?token=f0c6d63a7ffb4e67d132716e3ed49745e97b3e7fa78db28d
```
Open IDE in browser: http://127.0.0.1:8888/?token=f0c6d63a7ffb4e67d132716e3ed49745e97b3e7fa78db28d

Create new Python 3 notebook and create/picle/test model

```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
```
Note: Upload "Linear_regression_dataset.csv" into Jupyter Notebook it in the same folder as the Jupyter notebook that you are working in.

```
df=pd.read_csv('Linear_regression_dataset.csv',header='infer')
X=df.loc[:,df.columns !='output']
y=df['output']
```
The next step is to split the data into train and test sets. Then we build
the linear regression model on the training data and access the coefficient
values for all the input variables.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
lr = LinearRegression().fit(X_train, y_train)
lr.coef_
```
Example output:

array([ 3.33773356e-04,  6.12485115e-05,  2.57628117e-04, -6.69740154e-01,
        5.04645773e-01])


```
lr.score(X_train,y_train)
```
Exampe Output: 

0.8700856735266858
```
lr.score(X_test,y_test)
```
Exampe Output: 

0.8666629982427125

The performance of this baseline model seems reasonable, with an R-squared value of 87% on the training set and 86% on the test set.

Now that we have the trained model available, we can save it at any location or disk, using joblib or pickle. We name the exported model linear_regression_model.pkl.

```
import joblib
joblib.dump(lr,'linear_regression_model.pkl')
````

['linear_regression_model.pkl']

Now, we create a random input feature set and predict the output, using the trained model that we just saved.

```
test_data=[600,588,90,0.358,0.333]
pred_arr=np.array(test_data)
print(pred_arr)
```
Example Output:

[[6.00e+02 5.88e+02 9.00e+01 3.58e-01 3.33e-01]]

In order to predict the output with the same model, we first must
import or load the saved model, using joblib.load. Once the model is
loaded, we can simply use the predict function, to make the prediction on
a new data point.

```
model=open("linear_regression_model.pkl","rb")
lr_model=joblib.load(model)
model_prediction=lr_model.predict(preds)
print(model_prediction)
```
Example Output:

[0.36941795]

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-pickle-the-model.png" width="500">


- Deploying a Machine Learning Model As a REST Service (in docker container)

we can deploy the model as a REST (representational state transfer) service, in order to expose it to external users. This allows them to use the model output or prediction without having to access the underlying model. In this section, we will make use of Flask to deploy the model as a REST service. Flask is a lightweight web framework, built in Python, to deploy applications on a server.

Flask app.py : First, we import all the required libraries from Python. Next, we create our first function, which is the home page that renders the HTML template to allow users to fill input values. The next function is to publish the predictions by the model on those input values provided by the user. We save the input values into five different variables coming from the user and create a list (pred_args). We then convert that into a numpy array. We reshape it into the desired form, to be able to make predictions in the same way. The next step is to load the trained model (linear_regression_ model.pkl) and make the predictions. We save the final output into a variable (model_prediction). We then publish these results via another HTML template (predict.html). Tempaltes: There are two web pages that we have to design, in order to post requests to the server and receive in return the response message, which is the prediction by the machine learning model for that particular request. we are creating a form to request five values in five different variables. We are using a standard CSS template with very basic fields. The next template is to publish the model prediction back to the user. It is less complicated, compared to the first template, as there is just one value that we have to post back to the user.

We are going to use the same model that we built in the preceding section and deploy it, using the Flask server.. We can either move the model.pkl file manually to the web_app folder:

```
cd ./docker/web_app
$ docker ps -a 
CONTAINER ID        IMAGE                            COMMAND                  CREATED             STATUS              PORTS                    NAMES
b9571113d5a1        davarski/jupyterlab-eth:latest   "tini -g -- start-no…"   21 minutes ago      Up 21 minutes       0.0.0.0:8888->8888/tcp   jl
$ docker exec -it jl bash -c "ls"
Linear_regression_dataset.csv  linear_regression_model.pkl  Untitled.ipynb  work
$ docker cp jl:/home/jovyan/linear_regression_model.pkl .
$ docker build -t davarski/tf-linear-regression-rest:1.0.0 .
$ docker login 
$ docker push davarski/tf-linear-regression-rest:1.0.0
$ docker run -d -p 5000:5000 davarski/tf-linear-regression-rest:1.0.0 
936ef10f86797c1a7f1987ca168deb428f700fee8434e176805d1f5123cc95d2
$ docker logs 936ef10f8679
 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

Access  http://localhost:5000/

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-model-UI-1.png" width="500">

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-model-UI-2.png" width="500">

Building a Keras TensorFlow-Based Model

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-sample-from-the-Fashion-MNIST-data-set.png" width="500">

Execute Program3 in a new Jupyter notebook
```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
data = datasets.fashion_mnist
(ip_train, op_train), (ip_test, op_test) = data.load_data()
print(ip_train.shape, ip_test.shape)
ip_train = ip_train.reshape((60000, 28, 28, 1))
ip_test = ip_test.reshape((10000, 28, 28, 1))
print(ip_train.shape, ip_train.shape)
ip_train, ip_test = ip_train / 255.0, ip_test / 255.0
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model.summary()
model.fit(ip_train, op_train, epochs = 5)
model.evaluate(ip_test, op_test, verbose = 2)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
import matplotlib.pyplot as plt
%matplotlib inline
test_image=ip_test[5000]
plt.imshow(test_image.reshape(28,28))
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1, 28, 28, 1)
result = model.predict(test_image)
result
np.around(result)
n=(np.around(result)).argmax()
print(n)
print(class_names[n])
```

Now, we save the model as a Keras model and load it back, using load_model for prediction.

```
model.save("keras_model.h5")
loaded_model = models.load_model("keras_model.h5")
```
In the following example, we load a test image (300), which is a dress, and then we will use our saved model to make a prediction about this image : result = loaded_model.predict(test_image)


```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
import matplotlib.pyplot as plt
%matplotlib inline
test_image=ip_test[300]
plt.imshow(test_image.reshape(28,28))
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1, 28, 28, 1)
result = loaded_model.predict(test_image)
result
np.around(result)
n=(np.around(result)).argmax()
print(n)
print(class_names[n])
```
Example Output:

Dress


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-Keras-save-load-models.png" width="500">

TODO: TF ind deployment: Productionizing the machine learning model (deploy model) via Kubeflow/Seldon core (Note: package model into docker container if needed) 


Kubeflow is a native tool for managing and deploying machine learning models on Kubernetes. Kubernetes can be defined as a container orchestration platform that allows for the running, deployment, and management of containerized applications (machine learning models, in our case). We will replicate the same model that we built previously and run it in the cloud (via Google Cloud Platform), using Kubeflow. We will also use the Kubeflow UI, to navigate and run Jupyter Notebook in the cloud. Because we are going to use Google Cloud Platform (GCP), we must have a Google account, so that we can avail ourselves of the free credits provided by Google for the use of GCP components. Go to https://console.cloud.google.com/ and create a Google user account, if you do not have one already. You will be required to provide a few additional details, along with credit card information. Once we log in to the Google console, there are many options to explore, but first, we must enable the free credits provided by Google, in order to access the cloud services for free (up to $300). Next, we must create a new project or select one of the existing projects, for users already in possession of a Google account. To use Kubeflow, the final step is to enable Kubernetes Engine APIs. In
order to enable Kubernetes Engine APIs, we must go to the APIs & Services dashboard and search for Kubernetes Engine API. Once this shows up in the library, we must enable it.The next step is to deploy the Kubernetes cluster on GCP, using Kubeflow. Ref:  https://www.kubeflow.org/docs/gke/deploy/ Once we log in to the Kubeflow UI, we can see the Kubeflow dashboard, with its multiple options, such as Pipelines, Notebook Servers, etc. We must select Notebook Servers, to start a new notebook server. For a new notebook server, we must provide a few details regarding the desired configuration. Now we must provide a few configuration details to spin up the server, such as base image (with pre-installed libraries and dependencies), the size of CPU/GPUs, and total memory (5 CPUs and 5GB memory suffices for our model). We can select the image with TensorFlow version (2.0, if we are building the model with that version). We must also add GCP credentials, in case we want to save the model to GCP’s storage bucket and use it for serving purposes. After a while, the notebook server will be up and running, and we can click Connect, to open the Jupyter Notebook running on the Kubeflow server. Once Jupyter Notebook is up, we can select the option to create a new Python 3 notebook or simply go to its terminal and clone the required repo from Git, to download all the model files to this notebook. In our case, because we are building the model from scratch, we will create a new Python 3 notebook and replicate the same model built earlier. It should work exactly as before, the only difference being that we are now using Kubeflow to build and serve the model. In case any library is not available, we can simply pip3 install the library and use it in this notebook. Once the model is built and we have used the services of Kubeflow, we must terminate and delete all the resources, in order to avoid any extra cost. We must go back to the Google console and, under the Kubernetes clusters list, delete the Kubeflow server.

```
$ docker cp jl:/home/jovyan/keras_model.h5 .
$ mv keras_model.h5 model.pkl
$ mc mb minio-cluster/tensorflow
$ mc cp model.pkl minio-cluster/tensorflow/artifacts/model/model.pkl
```

Databrick

Use TensorFlow is through the Databricks platform.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/Databrick-Unified-Analytics-Platform.png" width="500">

Log in to the Databricks account and spin up a cluster of desired size:

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/Databrick-environment.png" width="500">


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/Databrick-create-cluster.png" width="500">

Once the cluster is up and running, we go to the Libraries options of the cluster, via Actions. Within the Libraries tab, if the cluster already has a set of pre-installed libraries, they will be listed, or, in the case of a new cluster, no packages will be installed. We then click the Install New button. This will open a new window with multiple options to import or install a new library in Databricks. We select PyPI, and in the Package option, we mention the version of TensorFlow required to be installed. It will take some time, and we can then see TensorFlow successfully installed in Databricks, under Libraries. We can now open a new or existing notebook using the same cluster. 


TODO1: DeepML with IoT

Smart Homes
Smart homes refer to the phenomenon of home automation, where the home does tasks on its own, without the need for anyone to control it. So far, smart homes have been able to do the following:

1. Switch lights on or off.
2. Keep a check on the overall temperature of the home.
3. Make sure all electronic devices are turned off when not in use.
4. Monitor the health of the inhabitants of the home.

Wearables
Wearables, as the name suggests, are devices that can be worn and that collect data about the wearer for further processing. Some common wearables are as follows:

1. Heart-rate monitors
2. Fitness monitors
3. Glucose monitors

Smart Greenhouse
The greenhouse farming technique aims to enhance crop yield by moderating temperature parameters. The problem is, this becomes difficult when it is required to be done by people. Smart greenhouses, therefore, can be used to fix this.

1. Its sensors can measure the various parameters.
2. It sends this data to the cloud, which processes this information.
3. An appropriate action is taken with regards to the plant/s involved.

