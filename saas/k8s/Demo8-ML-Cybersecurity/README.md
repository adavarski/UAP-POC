## Jupyter environment

Jupyter Notebooks are a browser-based (or web-based) IDE (integrated development environments)

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
$ cd ./k8s/
$ sudo k3s crictl pull davarski/jupyterlab-eth:latest
$ kubectl apply -f jupyter-notebook.pod.yaml -f jupyter-notebook.svc.yaml -f jupyter-notebook.ingress.yaml

```
Once the Pod is running, copy the generated token from the pod output logs.
```
$ kubectl logs jupyter-notebook
[I 06:44:51.680 LabApp] Writing notebook server cookie secret to /home/jovyan/.local/share/jupyter/runtime/notebook_cookie_secret
[I 06:44:51.904 LabApp] Loading IPython parallel extension
[I 06:44:51.916 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 06:44:51.916 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[W 06:44:51.920 LabApp] JupyterLab server extension not enabled, manually loading...
[I 06:44:51.929 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.7/site-packages/jupyterlab
[I 06:44:51.929 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 06:44:51.930 LabApp] Serving notebooks from local directory: /home/jovyan
[I 06:44:51.930 LabApp] The Jupyter Notebook is running at:
[I 06:44:51.930 LabApp] http://(jupyter-notebook or 127.0.0.1):8888/?token=1efac938a73ef297729290af9b301e92755f5ffd7c72bbf8
[I 06:44:51.930 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 06:44:51.933 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://(jupyter-notebook or 127.0.0.1):8888/?token=1efac938a73ef297729290af9b301e92755f5ffd7c72bbf8
```
Browse to http://jupyter.data.davar.com/lab


Note: Examples from book "Hands-On Artificial Intelligence for Cybersecurity"

## Machine Learning Map

<img src="https://github.com/adavarski/PaaS-and-SaaS-POC/blob/main/saas/k8s/Demo8-ML-Cybersecurity/pictures/machine-learning-map.png" width="900">


## Getting to know Python's libraries

Python libraries that are among the most well known and widespread in the field of ML:

   - NumPy 
   - pandas 
   - Matplotlib 
   - scikit-learn
   - Seaborn



### NumPy as an AI building block

Of all the Python libraries dedicated to data science and AI, there is no doubt that NumPy holds a privileged place. Using the functionalities and APIs implemented by NumPy, it is possible to build algorithms and tools for ML from scratch.

Of course, having specialized libraries available for AI (such as the scikit-learn library) accelerates the process of the development of AI and ML tools, but to fully appreciate the advantages deriving from the use of such higher-level libraries, it is useful to understand the building blocks on which they are built. This is why knowledge of the basic concepts of NumPy is helpful in this regard.


NumPy multidimensional arrays

NumPy was created to solve important scientific problems, which include linear algebra and matrix calculations. It offers a particularly optimized version, compared to the corresponding native versions of data structures offered by the Python language, such as lists of arrays and making multidimensional array objects, known as ndarrays, available. In fact, an object of the ndarray type allows the acceleration of operations to reach speeds of up to 25 times faster compared to traditional for loops, which is necessary to manage access to data stored in a traditional Python list.

Moreover, NumPy allows the management of operations on matrices, which is particularly useful for the implementation of ML algorithms. Unlike ndarray objects, matrices are objects that can take only two dimensions and represent the main data structures used in linear algebra.

Here are some examples of defining NumPy objects:
```
import numpy as np
np_array = np.array( [0, 1, 2, 3] )

# Creating an array with ten elements initialized as zero
np_zero_array = np.zeros(10)
```

Matrix operations with NumPy

As anticipated, matrices and the operations executed on them are of particular importance in the field of ML, and, more generally, they are used to conveniently represent the data to be fed to AI ​​algorithms.

Matrices are particularly useful in the management and representation of large amounts of data.

The notation itself is commonly used to identify the elements of a matrix, making use of positional indexes that allow the execution of consistent, rapid fashion operations, and calculations that concern either the whole matrix or just specific subsets. For example, the  element is easily identified within the matrix, crossing row  and column .

A special matrix, consisting of only one row (and several columns) is identified as a vector. Vectors can be represented in Python as objects of a list type.

However, the particular rules established by linear algebra should be taken into account when performing operations between matrices and vectors.

The basic operations that can be performed on matrices are as follows:

    Addition
    Subtraction
    Scalar multiplication (resulting in a constant value multiplied for each matrix element)

If such operations on matrices are relatively simple to accomplish, and are required only as a necessary precondition that the matrices that add or subtract from each other are of the same size, then the result of the addition or subtraction of two matrices is a new matrix whose elements are the result of the sum of corresponding elements in row and column order.

When dealing with the product operation between matrices or between vectors and matrices, the rules of linear algebra are partly different, since, for example, the commutative property is not applicable as it is in the case of the product of two scalars.

In fact, while in the case of the product of two numbers among them, the order of factors does not change the result of multiplication (that is, 2 x 3 = 3 x 2), in the case of the product of two matrices, the order is important:
```
aX != Xa
```
Here, X represents a matrix and a represents a vector of coefficients. Moreover, it is not always possible to multiply two matrices, as in the case of two matrices with incompatible dimensions.

For this reason, the numpy library provides the dot() function to calculate the product of two matrices between them (usable whenever this operation is possible):
```
import numpy as np
a = np.array([-8, 15])
X = np.array([[1, 5], 
              
              [3, 4],  
              
              [2, 3]])
y = np.dot(X, a)
```
In the preceding example, we calculate the product between matrix X and vector a using the np.dot() function.

This product is the expression of the model:
```
y = Xa
```
It represents one of the most basic models used in ML to associate a set of weights (a) to an input data matrix (X) in order to obtain the estimated values (y) as output.


Implementing a simple predictor with NumPy

To fully understand the use of the dot() method of NumPy in matrix multiplication operations, we can try to implement a simple predictor from scratch, to predict future values starting from a set of multiple inputs and on the basis of relative weights, using the product between matrices and vectors:

```
import numpy as np
def predict(data, w):  
     return data.dot(w)

# w is the vector of weights
w = np.array([0.1, 0.2, 0.3]) 

# matrices as input datasets
data1 = np.array([0.3, 1.5, 2.8]) 
data2 = np.array([0.5, 0.4, 0.9]) 
data3 = np.array([2.3, 3.1, 0.5])
data_in = np.array([data1[0],data2[0],data3[0]]) 
print('Predicted value: $%.2f' %  predict(data_in, w) )
```



### Scikit-learn

One of the best and most used ML libraries is definitely the scikit-learn library. First developed in 2007, the scikit-learn library provides a series of models and algorithms that are easily reusable in the development of customized solutions, which makes use of the main predictive methods and strategies, including the following:

    Classification
    Regression
    Dimensionality reduction
    Clustering

The list does not end here; in fact, scikit-learn also provides ready-to-use modules that allow the following tasks:

    Data preprocessing
    Feature extraction
    Hyperparameter optimization
    Model evaluation

The particularity of scikit-learn is that it uses the numpy library in addition to the SciPy library for scientific computing. As we have seen, NumPy allows the optimization of calculation operations performed on large datasets, using multidimensional arrays and matrices.

Among the advantages of scikit-learn, we must not forget that it provides developers with a very clean application programming interface (API), which makes the development of customized tools from the classes of the library relatively simple.

As an example of using the predictive analytics templates available in scikit-learn, we will show how to perform a prediction on training data (stored in the X matrix) using the linear regression model, based on a y weight vector.

Our goal will be to use the fit() and predict() methods implemented in the LinearRegression class:
```
import numpy as np
from sklearn.linear_model import LinearRegression

# X is a matrix that represents the training dataset

# y is a vector of weights, to be associated with input dataset

X = np.array([[3], [5], [7], [9], [11]]).reshape(-1, 1) 
y = [8.0, 9.1, 10.3, 11.4, 12.6]  
lreg_model = LinearRegression()  
lreg_model.fit(X, y) 

# New data (unseen before)
new_data = np.array([[13]]) 
print('Model Prediction for new data: $%.2f' 
       %  lreg_model.predict(new_data)[0]  )
```
Upon execution, the script produces the following output:
```
Model Prediction for new data: $13.73
```

### Matplotlib and Seaborn

One of the analytical tools used the most by analysts in AI and data science consists of the graphical representation of data. This allows a preliminary activity of data analysis known as exploratory data analysis (EDA). By means of EDA, it is possible to identify, from a simple visual survey of the data, the possibility of associating them with regularities or better predictive models than others.

Among graphical libraries, without a doubt, the best known and most used is the matplotlib library, through which it is possible to create graphs and images of the data being analyzed in a very simple and intuitive way.

Matplotlib is basically a data plotting tool inspired by MATLAB, and is similar to the ggplot tool used in R.

In the following code, we show a simple example of using the matplotlib library, using the plot() method to plot input data obtained by the arange() method (array range) of the numpy library:

```
import numpy as np 
import matplotlib.pyplot as plt  
plt.plot(np.arange(15), np.arange(15))
plt.show() 
```

In addition to the matplotlib library in Python, there is another well-known visualization tool among data scientists called Seaborn.

Seaborn is an extension of Matplotlib, which makes various visualization tools available for data science, simplifying the analyst's task and relieving them of the task of having to program the graphical data representation tools from scratch, using the basic features offered by matplotlib and scikit-learn.

### Pandas

The last (but not least) among Python's most used libraries that we'll look at here, is the pandas package, which helps to simplify the ordinary activity of data cleaning (an activity that absorbs most of the analyst's time) in order to proceed with the subsequent data analysis phase.

The implementation of pandas is very similar to that of the DataFrame package in R; DataFrame is nothing but a tabular structure used to store data in the form of a table, on which the columns represent the variables, while the rows represent the data itself.

In the following example, we will show a typical use of a DataFrame, obtained as a result of the instantiation of the DataFrame class of pandas, which receives, as an input parameter, one of the datasets (the iris dataset) available in scikit-learn.

After having instantiated the iris_df object of the  DataFrame type, the head() and describe() methods of the pandas library are invoked, which shows us the first five records of the dataset, respectively, and some of the main statistical measures calculated in the dataset:

```
import pandas as pd  
from sklearn import datasets

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head()
iris_df.describe()
```

Example python libraries notebook: https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Python-libs-examples.ipynb



## Python libraries for cybersecurity

Python is not only one of the best languages for data science and AI, but also the language preferred by penetration testers and malware analysts (along with low-level languages, such as C and Assembly).

In Python, there are an infinite number of libraries ready for use, which simplify the daily activities of researchers.

Next, we will analyze some of the most common and the most used of them.


### Pefile

The Pefile library is very useful for analyzing Windows executable files, especially during the phases of static malware analysis, looking for possible indications of compromise or the presence of malicious code in executables. In fact, Pefile makes it very easy to analyze the Portable Executable (PE) file format, which represents the standard for the object files (contained or retrievable as libraries of external executable functions) on the Microsoft platform.

So, not only the classic .exe files, but also the .dll libraries and .sys device drivers, follow the PE file format specification. The installation of the Pefile library is very simple; it is sufficient to use the pip command as used in the following example:

```
! pip install pefile
```
Once the installation is complete, we can test the library with a simple script such as the following, which loads the executable notepad.exe into runtime memory, and then extracts from its executable image some of the most relevant information saved in the relative PE file format fields:

```
import os
import pefile
notepad = pefile.PE("notepad.exe", fast_load=True)
dbgRVA = notepad.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress
imgver = notepad.OPTIONAL_HEADER.MajorImageVersion
expRVA = notepad.OPTIONAL_HEADER.DATA_DIRECTORY[0].VirtualAddress
iat = notepad.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress
sections = notepad.FILE_HEADER.NumberOfSections
dll = notepad.OPTIONAL_HEADER.DllCharacteristics
print("Notepad PE info: \n")
print ("Debug RVA: " + dbgRVA)
print ("\nImage Version: " + imgver)
print ("\nExport RVA: " + expRVA)
print ("\nImport Address Table: " + iat)
print ("\nNumber of Sections: " + sections)
print ("\nDynamic linking libraries: " + dll)
```

### Volatility

Another tool widely used by malware analysts is volatility, which allows the analysis of the runtime memory of an executable process, highlighting the presence of possible malware code.

Volatility is a Python-programmable utility, which is often installed by default in distributions for malware analysis and pentesting, such as Kali Linux. Volatility allows the extraction of important information about processes (such as API hooks, network connections and kernel modules) directly from memory dumps, providing the analyst with a suite of programmable tools using Python.

These tools allow the extraction from the memory dumps of all the processes running on the system and any relevant information about injected Dynamic-Link Libraries (DLLs), along with the presence of rootkits, or more generally, the presence of hidden processes within the runtime memory, which easily escapes the detection of common antivirus softwares.


## Python DL libraries
Python libraries for AI, in particular, to exploit the potential of deep learning. The libraries that are as follows:

   - TensorFlow
   - Keras
   - PyTorch


### Deep learning pros and cons for cybersecurity

One of the distinctive features of deep learning, compared to other branches of AI, is the ability to exploit general-purpose algorithms, by leveraging neural networks. In this way, it is possible to face similar problems that entail several different application domains, by reusing common algorithms elaborated in different contexts.

The deep learning approach exploits the possibility of neural networks (NNs) to add multiple processing layers, each layer having the task of executing different types of processing, sharing the results of the processing with the other layers.

Within a neural network, at least one layer is hidden, thus simulating the behavior of human brain neurons.

Among the most common uses of deep learning, are the following:

    Speech recognition
    Video anomaly detection
    Natural language processing (NLP)

These use cases are also of particular importance in the field of cybersecurity.

For example, for biometric authentication procedures, which are increasingly carried out by resorting to deep learning algorithms, deep learning can also be used successfully in the detection of anomalous user behaviors, or in the abnormal use of payment instruments, such as credit cards, as part of fraud detection procedures.

Another important use of deep learning is in the detection of possible malware or networking threats. Given the vast potential for using deep learning, it should not be surprising that even bad guys have begun to use it.

In particular, the recent spread of evolved neural networks such as generative adversarial networks (GANs) is posing a serious challenge to traditional biometric authentication procedures, which resort to facial recognition or voice recognition. By using a GAN, it is, in fact, possible to generate artificial samples of biometric evidence, which are practically indistinguishable from the original ones.

### TensorFlow

The first deep learning library we will deal with is TensorFlow; in fact, it plays a special role, having been specifically developed to program deep neural network (DNN) models.

Running a sample TensorFlow program as follows:
```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
For further documentation, visit the TensorFlow website at https://www.tensorflow.org/.

### Keras

The other deep learning library we will use is keras.

A characteristic of Keras is that it can be installed on top of TensorFlow, thus constituting a high-level interface (with respect to TensorFlow) for NN development. 

For further documentation, visit the Keras website at https://keras.io/.

### PyTorch

The last example of a deep learning library we will examine here is pytorch.

PyTorch is a project developed by Facebook, specially designed to perform large-scale image analysis. 


Note1:PyTorch versus TensorFlow

To compare both of the learning libraries, it should be noted that PyTorch is the most optimized solution for performing tensor calculus tasks on GPUs, as it has been specifically designed to improve performance in large-scale contexts.

Some of the most common use cases for using PyTorch are as follows:

    NLP
    Large-scale image processing
    Social media analysis

However, when compared only on the basis of performance, both PyTorch and TensorFlow are excellent choices; there are other characteristics that could make you lean toward one solution or the other.

For example, in TensorFlow, the debugging of programs is more complex than in PyTorch. This is because, in TensorFlow, development is more cumbersome (having to define tensors, initialize a session, keep track of tensors during the session, and so on), while the deployment of the TensorFlow model is certainly preferred.

Note2: Python on steroids with parallel GPU

To fully exploit the potential of some ML libraries, and especially DL, it is necessary to deploy dedicated hardware that includes the use of graphics processing units (GPUs) in addition to traditional CPUs. As current GPUs are, in fact, optimized to perform parallel calculations, this feature is very useful for the effective execution of many DL algorithms.

Reference hardware equipment could be the following:

    CPU Intel Core i5 6th Generation or higher (or AMD equivalent)
    8 GB RAM as a minimum (16 GB or higher is recommended)
    GPU  NVIDIA GeForce GTX 960 or higher (visit https://developer.nvidia.com/cuda-gpus for more info)
    Linux operating system (for example Ubuntu)

By leveraging the Numba compiler for example (provided by Anaconda), you can compile the Python code and run it on CUDA-capable GPUs.

For further information, please refer to the website of your GPU manufacturer and the Numba documentation (https://numba.pydata.org/numba-doc/latest/user/index.html).



## Types of machine learning used with Cybersecurity           
The process of mechanical learning from data can take different forms, with different characteristics and predictive abilities.In the case of ML (which, as we have seen, is a branch of research belonging to AI), it is common to distinguish between the following types of ML:
- Supervised learning
- Unsupervised learning
- Reinforcement learning
The differences between these learning modalities are attributable to the type of result (output) that we intend to achieve, based on the nature of the input required to produce it.


### Supervised learning

In the case of supervised learning, algorithm training is conducted using an input dataset, from which the type of output that we have to obtain is already known.

In practice, the algorithms must be trained to identify the relationships between the variables being trained, trying to optimize the learning parameters on the basis of the target variables (also called labels) that, as mentioned, are already known.

An example of a supervised learning algorithm is classification algorithms, which are particularly used in the field of cybersecurity for spam classification.

A spam filter is in fact trained by submitting an input dataset to the algorithm containing many examples of emails that have already been previously classified as spam (the emails were malicious or unwanted) or ham (the emails were genuine and harmless).

The classification algorithm of the spam filter must therefore learn to classify the new emails it will receive in the future, referring to the spam or ham classes based on the training previously performed on the input dataset of the already classified emails.

Another example of supervised algorithms is regression algorithms. Ultimately, there are the following main supervised algorithms:

   - Regression (linear and logistic)
   - k-Nearest Neighbors (k-NNs)
   - Support vector machines (SVMs)
   - Decision trees and random forests
   - Neural networks (NNs)
    
Example notebooks: 

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Supervised-learning-example%20_%20linear_regression.ipynb

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Simple-Neural%20Network-example_perceptron.ipynb

### Unsupervised learning

In the case of unsupervised learning, the algorithms must try to classify the data independently, without the aid of a previous classification provided by the analyst. In the context of cybersecurity, unsupervised learning algorithms are important for identifying new (not previously detected) forms of malware attacks, frauds, and email spamming campaigns.

Here are some examples of unsupervised algorithms:

   - Dimensionality reduction:
       - Principal component analysis (PCA)
       - PCA Kernel
   - Clustering:
       - k-means
       - Hierarchical cluster analysis (HCA)

Example: https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Unsupervised-learning-example_clustering.ipynb

### Reinforcement learning

In the case of reinforcement learning (RL), a different learning strategy is followed, which emulates the trial and error approach. Thus, drawing information from the feedback obtained during the learning path, with the aim of maximizing the reward finally obtained based on the number of correct decisions that the algorithm has selected.

In practice, the learning process takes place in an unsupervised manner, with the particularity that a positive reward is assigned to each correct decision (and a negative reward for incorrect decisions) taken at each step of the learning path. At the end of the learning process, the decisions of the algorithm are reassessed based on the final reward achieved.

Given its dynamic nature, it is no coincidence that RL is more similar to the general approach adopted by AI than to the common algorithms developed in ML.

The following are some examples of RL algorithms:

   -  Markov process
   -  Q-learning
   -  Temporal difference (TD) methods
   -  Monte Carlo methods

In particular, Hidden Markov Models (HMM) (which make use of the Markov process) are extremely important in the detection of polymorphic malware threats.

### Algorithm training and optimization

When preparing automated learning procedures, we will often face a series of challenges. We need to overcome these challenges in order to recognize and avoid compromising the reliability of the procedures themselves, thus preventing the possibility of drawing erroneous or hasty conclusions that, in the context of cybersecurity, can have devastating consequences.

One of the main problems that we often face, especially in the case of the configuration of threat detection procedures, is the management of false positives; that is, cases detected by the algorithm and classified as potential threats, which in reality are not (example Fraud Prevention)

The management of false positives is particularly burdensome in the case of detection systems aimed at contrasting networking threats, given that the number of events detected are often so high that they absorb and saturate all the human resources dedicated to threat detection activities.

On the other hand, even correct (true positive) reports, if in excessive numbers, contribute to functionally overloading the analysts, distracting them from priority tasks. The need to optimize the learning procedures therefore emerges in order to reduce the number of cases that need to be analyzed in depth by the analysts.

This optimization activity often starts with the selection and cleaning of the data submitted to the algorithms.

### How to find useful sources of data

In the case of anomaly detection, for example, particular attention must be paid to the data being analyzed. An effective anomaly detection activity presupposes that the training data does not contain the anomalies sought, but that on the contrary, they reflect the normal situation of reference.

If, on the other hand, the training data was biased with the anomalies being investigated, the anomaly detection activity would lose much of its reliability and utility in accordance with the principle commonly known as GIGO, which stands for garbage in, garbage out.

Given the increasing availability of raw data in real time, often the preliminary cleaning of data is considered a challenge in itself. In fact, it's often necessary to conduct a preliminary skim of the data, eliminating irrelevant or redundant information. We can then present the data to the algorithms in a correct form, which can improve their ability to learn, adapting to the form of data on the basis of the type of algorithm used.

For example, a classification algorithm will be able to identify a more representative and more effective model in cases in which the input data will be presented in a grouped form, or is capable of being linearly separable. In the same way, the presence of variables (also known as dimensions) containing empty fields weighs down the computational effort of the algorithm and produces less reliable predictive models due to the phenomenon known as the curse of dimensionality.

This occurs when the number of features, that is, dimensions, increases without improving the relevant information, simply resulting in data being dispersed in the increased space of research.

Also, the sources from which we draw our test cases (samples) are important. Think, for example, of a case in which we have to predict the mischievous behavior of an unknown executable. The problem in question is reduced to the definition of a model of classification of the executable, which must be traced back to one of two categories: genuine and malicious.

To achieve such a result, we need to train our classification algorithm by providing it with a number of examples of executables that are considered malicious as an input dataset.


### Quantity versus quality

When it all boils down to quantity versus quality, we are immediately faced with the following two problems:

    What types of malware can we consider most representative of the most probable risks and threats to our company?
    How many example cases (samples) should we collect and administer to the algorithms in order to obtain a reliable result in terms of both effectiveness and predictive efficiency of future threats?

The answers to the two questions are closely related to the knowledge that the analyst has of the specific organizational realm in which they must operate. 

All this could lead the analyst to believe that the creation of a honey-pot, which is useful for gathering malicious samples in the wild that will be fed to the algorithms as training samples, would be more representative of the level of risk to which the organization is exposed than the use of datasets as examples of generic threats. At the same time, the number of test examples to be submitted to the algorithm is determined by the characteristics of the data themselves. These can, in fact, present a prevalence of cases (skewness) of a certain type, to the detriment of other types, leading to a distortion in the predictions of the algorithm toward the classes that are most numerous, when in reality, the most relevant information for our investigation is represented by a class with a smaller number of cases.

In conclusion, it will not be a matter of being able to simply choose the best algorithm for our goals (which often does not exist), but mainly to select the most representative cases (samples) to be submitted to a set of algorithms, which we will try to optimize based on the results obtained.

### AI in the context of cybersecurity

With the exponential increase in the spread of threats associated with the daily diffusion of new malware, it is practically impossible to think of dealing effectively with these threats using only analysis conducted by human operators. It is necessary to introduce algorithms that allow us to automate that introductory phase of analysis known as triage, that is to say, to conduct a preliminary screening of the threats to be submitted to the attention of the cybersecurity professionals, allowing us to respond in a timely and effective manner to ongoing attacks.

We need to be able to respond in a dynamic fashion, adapting to the changes in the context related to the presence of unprecedented threats. This implies not only that the analysts manage the tools and methods of cybersecurity, but that they can also correctly interpret and evaluate the results offered by AI and ML algorithms.

Cybersecurity professionals are therefore called to understand the logic of the algorithms, thus proceeding to the fine tuning of their learning phases, based on the results and objectives to be achieved.

Some of the tasks related to the use of AI are as follows:

    Classification: This is one of the main tasks in the framework of cybersecurity. It's used to properly identify types of similar attacks, such as different pieces of malware belonging to the same family, that is, having common characteristics and behavior, even if their signatures are distinct (just think of polymorphic malware). In the same way, it is important to be able to adequately classify emails, distinguishing spam from legitimate emails.
    Clustering: Clustering is distinguished from classification by the ability to automatically identify the classes to which the samples belong when information about classes is not available in advance (this is a typical goal, as we have seen, of unsupervised learning). This task is of fundamental importance in malware analysis and forensic analysis.
    Predictive analysis: By exploiting NNs and DL, it is possible to identify threats as they occur. To this end, a highly dynamic approach must be adopted, which allows algorithms to optimize their learning capabilities automatically.

Possible uses of AI in cybersecurity are as follows:

    Network protection: The use of ML allows the implementation of highly sophisticated intrusion detection systems (IDS), which are to be used in the network perimeter protection area.
    Endpoint protection: Threats such as ransomware can be adequately detected by adopting algorithms that learn the behaviors that are typical of these types of malware, thus overcoming the limitations of traditional antivirus software.
    Application security: Some of the most insidious types of attacks on web applications include Server Side Request Forgery (SSRF) attacks, SQL injection, Cross-Site Scripting (XSS), and Distributed Denial of Service (DDoS) attacks. These are all types of threats that can be adequately countered by using AI and ML tools and algorithms.
    Suspect user behavior: Identifying attempts at fraud or compromising applications by malicious users at the very moment they occur is one of the emerging areas of application of DL.


## Detecting Cybersecurity Threats with AI

This section is dedicated to security threat detection techniques, using different strategies and algorithms of machine learning and deep learning, and comparing the results obtained.

This section contains the following examples:

- Ham or Spam? Detecting Email Cybersecurity Threats with AI
- Malware Threat Detection
- Network Anomaly Detection with AI

### Ham or Spam? Detecting Email Cybersecurity Threats with AI

Most security threats use email as an attack vector. Since the amount of traffic conveyed in this way is particularly large, it is necessary to use automated detection procedures that exploit machine learning (ML) algorithms. In this chapter, different detection strategies ranging from linear classifiers and Bayesian filters to more sophisticated solutions such as decision trees, logistic regression, and natural language processing (NLP) will be illustrated.

This example will cover the following topics:

    - How to detect spam with Perceptrons
    - Image spam detection with support vector machines (SVMs)
    - Phishing detection with logistic regression and decision trees
    - Spam detection with Naive Bayes adopting NLP


#### Detecting spam with linear classifiers

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Perceptron.ipynb

#### Spam detection with SVMs && Image spam detection with SVMs

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/SVM.ipynb


####  Phishing detection with logistic regression and decision trees

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Logistic_Regression_Phishing_Detector.ipynb
https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Decision_Tree_Phishing_Detector.ipynb


####  A Bayesian spam detector with NLTK
https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Bayesian_Spam_Detector_with_Nltk.ipynb


### Malware Threat Detection

This section will cover the following topics:

    - How to tell different malware families apart
    - Introducing the malware analysis methodology
    - Decision tree malware detectors
    - Detecting metamorphic malware with Hidden Markov Models (HMMs)
    - Advanced malware detection with deep learning


The high diffusion of malware and ransomware codes, together with the rapid polymorphic mutation in the different variants (polymorphic and metamorphic malware) of the same threats, has made traditional detection solutions based on signatures and hashing of image files obsolete, on which most common antivirus software is based.

It is therefore increasingly necessary to resort to machine learning (ML) solutions that allow a rapid screening (triage) of threats, focusing attention on not wasting scarce resources such as a malware analyst's skills and efforts.


#### Malware goes by many names

There are many types of malware, and every day new forms of threat arise that creatively reutilize previous forms of attack, or adopt radically new compromising strategies that exploit specific characteristics of the target organization (in the case of Advanced Persistent Threats (APTs), these are tailored forms of attack that perfectly adapt themselves to the target victim). This is only limited to the imagination of the attacker.

However, it is possible to compile a classification of the most common types of malware, in order to understand which are the most effective measures of prevention, and contrast their effectiveness for dealing with each malware species:

    - Trojans: Executables that appear as legitimate and harmless, but once they are launched, they execute malicious instructions in the background
    - Botnets: Malware that has the goal of compromising as many possible hosts of a network, in order to put their computational capacity at the service of the attacker
    - Downloaders: Malware that downloads malicious libraries or portions of code from the network and executes them on victim hosts
    - Rootkits: Malware that compromises the hosts at the operating system level and, therefore, often come in the form of device drivers, making the various countermeasures (such as antiviruses installed on the endpoints) ineffective
    - Ransomwares: Malware that proceeds to encrypt files stored inside the host machines, asking for a ransom from the victim (often to be paid in Bitcoin) to obtain the decryption key which is used for recovering the original files
    - APTs: APTs are forms of tailored attacks that exploit specific vulnerabilities on the victimized hosts
    - Zero days (0 days): Malware that exploits vulnerabilities not yet disclosed to the community of researchers and analysts, whose characteristics and impacts in terms of security are not yet known, and therefore go undetected by antivirus software

Obviously, these different types of threats can be amplified by the fact that they can mix together in the same malicious file (for example, a seemingly harmless Trojan becomes a real threat, as it behaves like a downloader once executed, connecting to the network and downloading malicious software, such as rootkits, which compromises the local network and turns it into a botnet).


#### Malware analysis tools of the trade

Many of the tools commonly used for conducting malware analysis can be categorized as follows:

    - Disassemblers (such as Disasm and IDA)
    - Debuggers (such as OllyDbg, WinDbg, and IDA)
    - System monitors (such as Process Monitor and Process Explorer)
    - Network monitors (such as TCP View, Wireshark, and tcpdump)
    - Unpacking tools and Packer Identifiers (such as PEiD)
    - Binary and code analysis tools (such as PEView, PE Explorer, LordPE, and ImpREC)

####  Malware detection strategies

 most common malware detection activities, we can include the following malware detection activities:

   - Hashes file calculation: To identify known threats already present in the knowledge base
   - System monitoring: To identify anomalous behavior of both the hardware and the operating system (such as an unusual increase in CPU cycles, a particularly heavy disk writing activity, changes to the registry keys, and the creation of new and unsolicited processes in the system)
   - Network monitoring: To identify anomalous connections established by host machines to remote destinations

These detection activities can be easily automated by using specific algorithms, as we will see shortly.


#### Random Forest Malware Classifier

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Random_Forest_Malware_Classifier.ipynb

#### Decision Tree Malware Detector

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Decision_Tree_Malware_Detector.ipynb

#### K-means malware clustering

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/K-means_malware_clustering.ipynb

#### Detecting metamorphic malware with Hidden Markov Models (HMMs)

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/HMM.ipynb

#### Advanced malware detection with deep learning (Convolutional Neural Networks :CNNs)


Detecting malware from images with CNNs

A tool was developed that leverages CNNs to recognize and classify the images that represent malware codes.

The tool can be downloaded from the GitHub repository, by executing the following command:
```
git clone https://github.com/AFAgarap/malware-classification.git/
```
Inside the archive, there is also a dataset of images of malware codes (malimg.npz). To convert your malware codes to grayscale images, you can also use the Python script developed by Chiheb Chebbi, which is available at https://github.com/PacktPublishing/Mastering-Machine-Learning-for-Penetration-Testing/blob/master/Chapter04/MalwareConvert.py.

We show some examples of the tool's usage as follows:
```
Usage: main.py [-h] -m MODEL -d DATASET -n NUM_EPOCHS -c PENALTY_PARAMETER -k CHECKPOINT_PATH -l LOG_PATH -r RESULT_PATH
```
To use the CNN-SVM model, set the -model parameter to 1, as in the following example:
```
main.py –model 1 –dataset ./dataset/malimg.npz –num_epochs 100 –penalty_parameter 10 -c ./checkpoint/ -l ./logs/ -r ./results/
```

### Network Anomaly Detection with AI


The current level of interconnection that can be established between different devices (for example, think of the Internet of Things (IoT)) has reached such a complexity that it seriously questions the effectiveness of traditional concepts such as perimeter security. As a matter of fact, cyberspace's attack surface grows exponentially, and it is therefore essential to resort to automated tools for the effective detection of network anomalies associated with unprecedented cybersecurity threats.

This chapter will cover the following topics:

    - Network anomaly detection techniques
    - How to classify network attacks
    - Detecting botnet topology
    - Different machine learning (ML) algorithms for botnet detection

 

In this section, we will focus on anomaly detection related to network security, postponing the discussion of the aspects of fraud detection and user anomaly behavior detection until the following sections.

#### Network anomaly detection techniques

The techniques we have seen so far can also be adopted to manage anomaly detection and related attempts to gain unauthorized access to the corporate network.
In fact, anomaly detection has always been a research area of cybersecurity, particularly in the field of network security protection. However, anomaly detection is not limited to identifying and preventing network attacks, but can also be adopted in other areas, such as fraud detection and in the identification of possible compromises of user profiles.


#### Intrusion Detection Systems 

Traditionally, intrusion detection activity has been managed through the introduction of specialized devices, known as Intrusion Detection Systems (IDS). These devices were usually divided into the following two categories:

    - Host-based IDS
    - Network-based IDS

With the introduction of Artificial Intelligence (AI) techniques in the field of cybersecurity, a third type of IDS has also been added to the aforementioned two traditional types: anomaly-driven IDS.


#### Anomaly-driven IDS

With the introduction of AI techniques to the field of NIDS, it is now possible to evolve traditional IDS toward more advanced detection solutions, exploiting supervised and unsupervised learning algorithms, as well as reinforcement learning and deep learning.

Similarly, the clustering techniques analyzed in the previous chapters, which exploit the concepts of similarity between the categories of data, can validly be used for the implementation of anomaly-based IDS.

In choosing the algorithms for the anomaly detection network, however, some characteristic aspects of the network environment must be taken into consideration:

    - In the case of solutions based on supervised learning algorithms, we will have to categorize (label) all the data since, by definition, in supervised learning the classes to which sample data belongs are known in advance.
    - The categorization of all the data inevitably involves a computational overload and a possible slowdown in the network performance, since the network traffic must be analyzed before being sent to the destination. In this sense, we could decide to resort to unsupervised learning algorithms, not only to let the algorithm identify unknown classes, but also to reduce the computational overhead.

Similarly, the algorithms that exploit the concept of similarity (such as clustering algorithms) are well suited for the implementation of anomaly detection solutions. However, it is also necessary in this case to pay particular attention to the type of metrics used to define the concept of similarity that distinguishes the traffic considered as normal from that being considered as anomalous.

Commonly, in the implementation of anomaly detection solutions, scoring systems are used for the assessment of traffic: identifying thresholds of values ​​that separate the different types of traffic from each other (normal versus anomalous). For this purpose, when choosing the most appropriate metrics, we must take into consideration the ordering and distribution of the data.

In other words, an anomaly detection system can use—as a scoring metric—the distance existing between the values ​​of the dataset (considered as points of the n-dimensional space representing the different features), or evaluate the regularity in the distribution of data, with respect to a distribution considered representative of the phenomenon under investigation.

#### Turning service logs into datasets

One of the problems related to network anomaly detection is how to collect sufficient and reliable data to perform algorithm analysis and training. There are hundreds of datasets freely available on the internet, and there are different datasets on which to carry out our analysis; however, it is also possible to use our own network devices to accumulate data that is more representative of our specific reality.

To this end, we can use the following:

    - Network devices, such as routers or network sensors, using tools such as tcpdump for data collection
    - Services logs and system logs

Within the operating system, services logs and system logs can be stored in various places. In the case of a Unix-like system, the services logs are usually stored as text files inside the /var/log directory and its relative subdirectories. In the case of Windows OSes, logs are distinguished between Windows logs (including security logs and system logs) and application logs. They are accessible through the Windows Event Viewer application, or by accessing file system locations such as %SystemRoot%\System32\Config.

In both the cases of Unix-like systems and Windows OSes, the log files are of a text-based format according to predefined templates, with the difference being that, in the case of Windows, an event ID is also associated with each event recorded in the corresponding log file. The textual nature of the logs files is well suited for the integration of the information stored in the logs.


#### Advantages of integrating network data with service logs

Both data sources, that is, network data and services logs, entail advantages and disadvantages for the purposes of network anomaly detection.

However, their integration makes it possible to limit the disadvantages in favor of the advantages.

It is no coincidence that in recent years, several software solutions (both proprietary and open source) have been released to solve the task of integrating different data sources, allowing users to utilize methods of analysis from data science and big data analytics.

Among the most widespread solutions, we can mention the ElasticSearch, Logstash, Kibana (ELK) suite, which allows the indexing of events extracted from log files and can be represented in an intuitive visual form.

Other widespread proprietary networking solutions are based on Cisco's NetFlow protocol, which allows for a compact representation of network traffic.

Reconstructing the events of interest starting from the raw data is anything but easy. This moreover lends itself—if carried out in an automated manner—to the generation of unreliable signals (false positives) that represent a problem in the management of security.

Moreover, in the case of network data, they are representative of the individual services to which they refer, while in the case of service logs, they are directly related to the processes that generated them.

The integration of both data sources (network data and service logs) therefore allows for a contextualization of the events being analyzed, consequently increasing contextual awareness, and reducing the effort required for interpreting events when starting from raw data.

####  Most common network attacks

Given the enormous variety of combinations that we can identify by putting together different features, it is inevitable that we have to resort to a threat model that reflects the level of risk to which a given organization is subjected, and on the basis of this model, to identify the most representative feature of combinations for possible attacks.

In this sense, it can be useful to analyze which are the most frequent types of network attacks:

    - Malware-based
    - Zero-day exploits
    - Data exfiltration via network sniffing
    - Saturation of network resources (DoS)
    - Session hijacking
    - Connection spoofing
    - Port scanning

On the basis of a similar classification (to be adapted to the specific context and constantly updated), we can identify which features to consider, feeding our algorithms with more representative datasets.

#### Anomaly detection strategies

We have therefore seen that the very concept of anomaly detection refers to a behavior that is different from what was expected; this difference, in technical terms, translates into outlier detection.

To identify the outliers, it is possible to follow different strategies:

   - Analyzing a sequence of events within a time series: The data is collected at regular intervals, evaluating the changes that occur in the series over time. This is a technique widely used in the analysis of financial markets, but it can be also validly used in the cybersecurity context to detect the frequency of characters (or commands) entered by the user in a remote session. Even the simple unnatural increase in the frequency of data entered per unit of time is indicative of an anomaly that can be traced back to the presence of an automated agent (instead of a human user) in the remote endpoint.

   - Using supervised learning algorithms: This approach makes sense when normal and anomalous behaviors can be reliably distinguished from each other, as in the case of credit card fraud, in which it is possible to detect predefined patterns of suspicious behavior, relying on the fact that future fraud attempts are attributable to a predefined scheme.

   - Using unsupervised learning algorithms: In this case, it is not possible to trace the anomalies back to predefined behaviors, as it is not possible to identify a reliable and representative training dataset for supervised learning. This scenario is the one that most commonly describes the reality of cybersecurity, characterized by new forms of attack or exploits of new vulnerabilities (zero-day attacks). Similarly, it is often difficult to trace all the theoretically possible intrusions back to a single predefined scheme.

####  Detecting botnet topology

One of the most common pitfalls in network anomaly detection has to do with the detection of botnets within the corporate network. Given the danger of such hidden networks, the detection of botnets is particularly relevant, not only for preventing the exhaustion of the organization's computational and network resources by external attackers, but also for preventing the dissemination of sensitive information (data leakage) outward.

However, identifying the presence of a botnet in time is often an operation that is anything but simple. This is why it is important to understand the very nature of botnets.

In the case of botnets, the attacker's intent is to transform the victim host (by installing malware) into an automated agent that fulfills the orders received by the attacker, through a C2 console that is usually managed by a centralized server.

The victim machine thus becomes part of a vast network of compromised machines (the botnet), contributing toward a common goal with its own computational and network resources:

    - Taking part in email spamming campaigns
    - Performing Distributed Denial of Services (DDoS) toward institutional or private third-party sites
    - Bitcoin and cryptocurrency mining
    - Password cracking
    - Credit card cracking
    - Data leakages and data breaches

For an organization, dealing with a botnet (even unconsciously) represents a serious risk in terms of legal responsibility toward third parties; it is not just a waste of company resources.

This is why it is important to monitor the company network by trying to promptly identify the presence of hosts that might be part of a botnet.

####  Different ML algorithms for botnet detection


From what we have described so far, it is clear that it is not advisable to exclusively rely on automated tools for network anomaly detection, but it may be more productive to adopt AI algorithms that are able to dynamically learn how to recognize the presence of any anomalies within the network traffic, thus allowing the analyst to perform an in-depth analysis of only really suspicious cases. Now, we will demonstrate the use of different ML algorithms for network anomaly detection, which can also be used to identify a botnet.

The section features in our example consist of the values of network latency and network throughput. In our threat model, anomalous values associated with these features can be considered as representative of the presence of a botnet.

For each example, the accuracy of the algorithm is calculated, in order to be able to make comparisons between the results obtained:

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Network_Anomaly_Detection.ipynb

####  Anomaly detection using the Gaussian distribution


Gaussian anomaly detection

One of the most widespread approaches for detecting regularity within data distribution makes use of the Gaussian distribution of probabilities.

As we shall see, this statistical distribution presents a series of interesting characteristics that help to adequately model many natural, social, and economic phenomena.

Obviously, not all the phenomena under investigation can be represented by the Gaussian distribution (very often, as we have seen, the underlying distribution of the analyzed phenomena is unknown); however, it constitutes a reliable reference point in many cases of anomaly detection.

Therefore, we must see the characteristics of the Gaussian distribution in order to understand why it is frequently used.

https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Gaussian_Network_Anomaly_Detection.ipynb


## Protecting Sensitive Information and Assets

This section covers authentication abuse and fraud prevention through biometric authentication, login attempt classification, account velocity features, and reputation scores.

This section contains the following chapters:

    - Securing User Authentication
    - Fraud Prevention with Cloud AI Solutions
    - GANS – Attacks and Defenses



### Securing User Authentication

In the field of cybersecurity, Artificial Intelligence (AI) is assuming an increasingly important role in the protection of users' sensitive information, including the credentials they use to access their network accounts and applications in order to prevent abuse, such as identity theft.

This chapter will cover the following topics:

    - Authentication abuse prevention
    - Account reputation scoring
    - User authentication with keystroke recognition
    - Biometric authentication with facial recognition





#### Keystroke detection example 


Keystroke Dynamics - Benchmark Data Set:

```
$ wget https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv (and put into Jupyter env directory: datasets)
```
https://github.com/adavarski/PaaS-and-SaaS-POC/tree/main/saas/k8s/Demo8-ML-Cybersecurity/notebooks/Keystroke_Detection.ipynb


### Fraud Prevention with Cloud AI Solutions


Fraud Prevention with Cloud AI Solutions

The objective of many security attacks and data breaches that corporations suffer from is the violation of sensitive information, such as customers' credit card details. Such attacks are often conducted in stealth mode, and so it is difficult to detect such threats using traditional methods. In addition, the amount of data to be monitored often assumes dimensions that cannot be effectively analyzed with just traditional extract, transform, and load (ETL) procedures that are executed on relational databases, which is why it is important to adopt artificial intelligence (AI) solutions that are scalable. By doing this, companies can take advantage of cloud architectures in order to manage big data and leverage predictive analytics methodology.

Credit card fraud represents an important test for the application of AI solutions in the field of cybersecurity since it requires the development of predictive analytics models that exploit big data analytics through the use of cloud computing platforms.

In this section, we will learn about the following topics:

    - How to leverage machine learning (ML) algorithms for fraud detection
    - How bagging and boosting techniques can improve an algorithm's effectiveness
    - How to analyze data with IBM Watson and Jupyter Notebook
    - How to resort to statistical metrics for results evaluation
    
    
    

Download the 'creditcard' dataset in .csv format from the following link:
https://www.openml.org/data/get_csv/1673544/phpKo8OWT

Dataset License: https://www.openml.org/d/1597
(PUBLIC DOMAIN: https://creativecommons.org/publicdomain/mark/1.0/)

Dataset Credits:
Author: Andrea Dal Pozzolo, Olivier Caelen and Gianluca Bontempi
Source: Credit card fraud detection - Date 25th of June 2015
Please cite: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. 
Calibrating Probability with Undersampling for Unbalanced Classification. 
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

```
$ wget https://www.openml.org/data/get_csv/1673544/phpKo8OWT
# Put file into jupyter dataset folder
```

Rule-based IDPS:
```
IF amount > $1,000 AND buying_frequency > historical_buying_frequency THEN fraud_likelihood = 90%

IF distance(new_transaction, last_transaction) > 1000 km AND time_range < 30 min THEN block_transaction
```

-------------

Bagging Classifier example:

```
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(
            DecisionTreeClassifier(), 
            n_estimators=300,
            max_samples=100, 
            bootstrap=True
          )
```

--------------

Boosting with AdaBoost example:

```
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier


adaboost = AdaBoostClassifier(
              DecisionTreeClassifier(),
              n_estimators=300
           )

```
----------------

Gradient Boosting Classifier example:

```
from sklearn.ensemble import GradientBoostingClassifier

gradient_boost = GradientBoostingClassifier(
                   max_depth=2, 
                   n_estimators=100, 
                   learning_rate=1.0,
                   warm_start=True
                 )
```

-----------------

eXtreme Gradient Boosting (Xgboost) Classifier example:

```
from xgboost.sklearn import XGBClassifier

xgb_model = XGBClassifier()
```

-----------------

Under-sampling with RandomUnderSampler:
```
# From the Imbalanced-Learn library documentation:
# https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.RandomUnderSampler.html

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler 

X, y = make_classification(n_classes=2, class_sep=2,
 weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
```

Over-sampling with SMOTE:
```
# From the Imbalanced-Learn library documentation:
# https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 

X, y = make_classification(n_classes=2, class_sep=2,
   weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
   n_features=20, n_clusters_per_class=1, n_samples=1000,    
   random_state=10)

print('Original dataset shape %s' % Counter(y))
Original dataset shape Counter({1: 900, 0: 100})

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
Resampled dataset shape Counter({0: 900, 1: 900})
```


-----------------

Note: Develop a predictive model for credit card fraud detection, exploiting the IBM Cloud platform with IBM Watson Studio.

IBM Fraud Detection notebook available at:   
https://github.com/IBM/xgboost-smote-detect-fraud/blob/master/notebook/Fraud_Detection.ipynb   
(Source code Released under apache version 2 license: http://www.apache.org/licenses/LICENSE-2.0.txt) 

### GANs - Attacks and Defenses

Generative adversarial networks (GANs) represent the most advanced example of neural networks that deep learning makes available to us in the context of cybersecurity. GANs can be used for legitimate purposes, such as authentication procedures, but they can also be exploited to violate these procedures. 

In this chapter, we will look at the following topics:

    The fundamental concepts of GANs and their use in attack and defense scenarios
    The main libraries and tools for developing adversarial examples
    Attacks against deep neural networks (DNNs) via model substitution
    Attacks against intrusion detection systems (IDS) via GANs
    Attacks against facial recognition procedures using adversarial examples


GANs in a nutshell

GANs were theorized in a famous paper that dates back to 2014 (https://arxiv.org/abs/1406.2661), written by a team of researchers including Ian Goodfellow and Yoshua Bengio, which described the potential and characteristics of a special category of adversarial processes, called GANs.

The basic idea behind GANs is simple, as they consist of putting two neural networks in competition with one another, until a balanced condition of results is achieved; however at the same time, the possibilities of using these intuitions are almost unlimited, since GANs are able to learn how to imitate and artificially reproduce any data distribution, whether it represents faces, voices, texts, or even works of art.

In this section, we will extend the use of GANs in the field of cybersecurity, learning how it is possible to use them to both carry out attacks (such as attacks against security procedures based on the recognition of biometric evidences) and to defend neural networks from attacks conducted through GANs. In order to fully understand the characteristics and potential of GANs, we need to introduce a number of fundamental concepts concerning neural networks (NNs) and deep learning (DL).


Getting to know GANs

We have said that the intuition on which GANs are based entails putting two NNs in competition with one another in order to improve the overall results. The term adversarial refers specifically to the fact that the two NNs compete between themselves in completing their respective tasks. The outcome of this competition is an overall result that cannot be further improved, thereby attaining an equilibrium condition.

A typical example of using GANs is the implementation of a particular NN, called a generative network, with the task of creating an artificial image that simulates the characteristics of a real image. A second NN, called the discriminator network, is placed in competition with the first one (the generator) in order to distinguish the artificially simulated image from the real one.

An interesting aspect is the fact that the two networks collaborate in achieving a situation of equilibrium (condition of indifference), putting in competition with one another the optimization of their respective objective functions. The generator network bases its optimization process on its ability to deceive the discriminator network.

The discriminator network, in turn, carries out its optimization process, based on the accuracy achieved in distinguishing the real image from the artificially generated image from the generator network. 

Examples:
```
cd ./GANs/
```

### Evaluating and Testing Your AI Arsenal

Learning to evaluate, and continuously test the effectiveness of, your AI-based cybersecurity algorithms and tools is just as important as knowing how to develop and deploy them. In this section, we'll learn how to evaluate and test our work.

This section contains the following chapters:

    - Evaluating Algorithms
    - Assessing Your AI Arsenal

####  Evaluating Algorithms


As we have seen in the previous sections, several AI solutions are available to achieve certain cybersecurity goals, so it is important to learn how to evaluate the effectiveness of various alternative solutions, using appropriate analysis metrics. At the same time, it is important to prevent phenomena such as overfitting, which can compromise the reliability of forecasts when switching from training data to test data.

In this section, we will learn about the following topics:

    - Feature engineering best practices in dealing with raw data
    - How to evaluate a detector's performance using the ROC curve
    - How to appropriately split sample data into training and test sets
    - How to manage algorithms' overfitting and bias–variance trade-offs with cross validation


Feature engineering examples with sklearn

Let's look at some examples of feature engineering implementation using the NumPy library and the preprocessing package of the scikit-learn library.

In this section, we will look at the different techniques commonly adopted to evaluate the predictive performances of different algorithms. We will look at how to transform raw data into features, following feature engineering best practices, thereby allowing algorithms to use data that does not have a numeric form, such as categorical variables. We will focused on the techniques needed to correctly evaluate the various components (such as bias and variance) that constitute the generalization error associated with the algorithms, and finally, we will learn how to perform the cross validation of the algorithms to improve the training process.

```
# Min-Max Scaler

from sklearn import preprocessing 

import numpy as np

raw_data = np.array([

  [ 2., -3., 4.],

  [ 5., 0., 1.],

  [ 4., 0., -2.]]) 

min_max_scaler = preprocessing.MinMaxScaler()

scaled_data = min_max_scaler.fit_transform(raw_data)  


# Standard Scaler

from sklearn import preprocessing 

import numpy as np

raw_data = np.array([

  [ 2., -3., 4.],

  [ 5., 0., 1.],

  [ 4., 0., -2.]]) 

std_scaler = preprocessing.StandardScaler().fit(raw_data) 

std_scaler.transform(raw_data)

test_data = [[-3., 1., 2.]]

std_scaler.transform(test_data) 


# Power Transformation

from sklearn import preprocessing 

import numpy as np

pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)  

X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))

pt.fit_transform(X_lognormal) 


# Ordinal Encoding

from sklearn import preprocessing 

ord_enc = preprocessing.OrdinalEncoder()

cat_data = [['Developer', 'Remote Working', 'Windows'], ['Sysadmin', 'Onsite Working', 'Linux']] 

ord_enc.fit(cat_data)

ord_enc.transform([['Developer', 'Onsite Working', 'Linux']])


# One-Hot Encoding

from sklearn import preprocessing 

one_hot_enc = preprocessing.OneHotEncoder()

cat_data = [['Developer', 'Remote Working', 'Windows'], ['Sysadmin', 'Onsite Working', 'Linux']] 

one_hot_enc.fit(cat_data)

one_hot_enc.transform([['Developer', 'Onsite Working', 'Linux']])


# ROC Metrics Examples

import numpy as np

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

y_true = np.array([0, 1, 1, 1])
y_pred = np.array([0.2, 0.7, 0.65, 0.9])

prec, rec, thres = precision_recall_curve(y_true, y_pred) 

average_precision_score(y_true, y_pred)

metrics.precision_score(y_true, y_pred) 

metrics.recall_score(y_true, y_pred) 

metrics.f1_score(y_true, y_pred) 


# ROC curve Example

import numpy as np
from sklearn.metrics import roc_curve

y_true = np.array([0, 1, 1, 1])
y_pred = np.array([0.2, 0.7, 0.65, 0.9])

FPR, TPR, THR = roc_curve(y_true, y_pred) 


# AUC score Example

import numpy as np
from sklearn.metrics import roc_auc_score 

y_true = np.array([0, 1, 1, 1])
y_pred = np.array([0.2, 0.7, 0.65, 0.9])

roc_auc_score(y_true, y_pred)


# Brier score Example

import numpy as np
from sklearn.metrics import brier_score_loss

y_true = np.array([0, 1, 1, 1])
y_cats = np.array(["fraud", "legit", "legit", "legit"]) 

y_prob = np.array([0.2, 0.7, 0.9, 0.3])
y_pred = np.array([1, 1, 1, 0])

brier_score_loss(y_true, y_prob)

brier_score_loss(y_cats, y_prob, pos_label="legit") 

brier_score_loss(y_true, y_prob > 0.5)


# Splitting samples into training and testing subsets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Learning curve example

from sklearn.model_selection import learning_curve 

from sklearn.svm import SVC

_sizes = [ 60, 80, 100]

train_sizes, train_scores, valid_scores = learning_curve(SVC(), X, y, train_sizes=_sizes) 


# K-folds Cross Validation example

import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1., 0.], [2., 1.], [-2., -1.], [3., 2.]])
y = np.array([0, 1, 0, 1])

k_folds = KFold(n_splits=2)

for train, test in k_folds.split(X): 
    print("%s %s" % (train, test))

[2 0] [3 1]
[3 1] [2 0] 

X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
```



#### Assessing Your AI Arsenal


In addition to evaluating the effectiveness of their algorithms, it is also important to know the techniques that attackers exploit to evade Our AI-empowered tools. Only in this way is it possible to gain a realistic idea of the effectiveness and reliability of the solutions adopted. Also, the aspects related to the scalability of the solutions must be taken into consideration, along with their continuous monitoring, in order to guarantee reliability.

In this section, we will learn about the following:

    - How attackers leverage Artificial Intelligence (AI) to evade Machine Learning (ML) anomaly detectors
    - The challenges we face when implementing ML anomaly detection
    - How to test our solutions for data and model quality
    - How to ensure security and reliability of our AI solutions for cybersecurity



Note: We dealt with the assessment of our AI-based cybersecurity solutions, analyzing the aspects of security, data, and model quality, in order to guarantee the reliability and high availability of solutions deployed in production environments, without neglecting the requirements of privacy and confidentiality of the sensitive data used by algorithms.

Examples: 

```
cd ./additional
```

