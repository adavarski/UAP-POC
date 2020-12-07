
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


### TensorFlow 2.0 DeepML library

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

#### Neural Networks
The neural network, or artificial neural network, was inspired by and modeled after the biological neural network. These networks, like the human brain, learn to perform specific tasks without being explicitly programmed. A neural network is composed of a series of neurons that are connected together to form a type of network, hence the name neural network. A neuron, or an artificial neuron, is the fundamental unit of a neural network. It is a mathematical function that replicates the neurons in the human brain, as you can see 


<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-a-biological-neuron-and-an-artificial-neuron.png" width="500">

Comparison of biological and artificial neurons.

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo4-DeepML-TensorFlow/pictures/TensorFlow-comparison-of-a-biological-and-an-artificial-neuron.png" width="500">


### Working of an Artificial Neuron (Perceptron)
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
layers,” which make use of filters. A filter is a grid of size AxB that is moved across the image and gets multiplied several times by it to produce a new value. Each value represents a line or an edge in the image. Once the filters have been used on the image, its important characteristics can be extracted. This is done with the help of a pooling layer. These layers pool or collect the main features of each image. One popular technique of doing this is known as max pooling, which takes the largest number of each image and stores it in a separate grid. It thus compresses the main features into a single image and then passes it on to a
regular multi-layer neural network for further processing. These neural networks are mainly used for image classification. They
can also be used in search engines and recommender systems.

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


Each type of learning method has various types of algorithms that can be used to solve a machine learning problem. Let’s take a look at some important ones.

Supervised Learning Algorithms
The goal of every supervised learning algorithm is to map the input to the output, as shown in the following equation:
`y = f(x)`

There are several algorithms that can be used to solve a machine learning problem with the help of supervised learning. These algorithms can be segregated into the following categories:

1. Regression algorithms: These algorithms contain outputs that are real or countable. For example, height (4 feet, 5 feet, 6 feet), age (27, 31, 65), or price
(100 rupees, 20 pounds, 10 dollars)

2. Classification algorithms: These algorithms contain outputs that are abstract or categorical. For example, colors (orange, purple, turquoise), emotions (happy, sad, angry), or gender (girl, boy). To give you some idea of what these algorithms are, let’s go through
three common types of algorithms that are used:

• Linear regression
• Logistic regression
• K-Nearest neighbors

Applications of Supervised Learning Algorithms

1. Spam detection

2. Bioinformatics: This is the method of keeping a record of a person’s biological information for later use. One of the most common examples of this is the
security system on our cell phones, which can scan our fingerprint and grant us access accordingly.

Unsupervised Learning Algorithms
The goal of unsupervised learning algorithms is to discover possible patterns from the set of data that is provided. The algorithm has no prior information about the patterns and labels present in the data. There are several algorithms that can be used to solve a machine learning problem with the help of unsupervised learning. These algorithms can be segregated into the following categories:

• Cluster analysis: This approach finds similarities
among the data and then groups the common data
together in clusters.

• Dimensionality reduction: This approach attempts
to reduce the complexity of data while still keeping the
data relevant.

Two common algorithms that are used for unsupervised learning: K-means clustering and principal component analysis.

-KMeans Clustering

-Principal Component Analysis

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



