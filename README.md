# Project Overview
Project main goal was to understand and, hopefully implement Convolutional Neural Network. <br>
The results of the implementation, in my opinion, are poor. <br>
First, network using convolutional layer is extremely slow.
Main reason for that is self-implemented convolution which is highly unoptimised. <br>
Second, network is unable to perform back-propagation in convolutional layers.
It is caused by the fact that another subroutine used while computing 
necessary values is also, really slow. <br>
Nevertheless I believe that this project was worth completing,
and it produced some interesting conclusions, which will be pointed out 
later.

---

## Convolutional Layer

Implementation of the convolutional layer was the main goal of the project.
The created layer performs convolution, max pooling and reshaping operations.
It also updates filters and biases weights accordingly to 
sgd algorithm and backpropagation. <br>
Unfortunately this layer does not perform further backpropagation. 
It is caused by several reasons: 

- Huge time overhead due to unoptimised subroutine.
- Immense growth of the back-propagated gradients.
- Several difficulties with flattening layers caused by the architecture of the layer.

Nevertheless, convolutional layer is performing fine as the input layer for the network.

---

## Network Architecture

Network architecture is focused on layers positioning. It means that 
network is constructed as the chain of layers which implement necessary methods.
Network performs backpropagation in iterative form since convolutional 
layers are unable to feed-forward multiple input samples. <br>
Gradient descent steps and necessary weights and biases updates 
are delegated into layers.

---

## Conclusions

Because convolutional layers are slow, network trainings and evaluations were 
possible only on small subsets of MNIST dataset. <br>
After performing several filters adjustments and hyper-parameters tuning I was 
able to obtain results similar to regular fully-connected network. <br>
For fully-connected layers network I achieved:

- 91.5 % accuracy on validation set.
- 80.6 % accuracy on test set.

For a convolutional network I achieved:

- 90.5 % accuracy on validation set.
- 80.3 % accuracy on test set.

It is worth mentioning that:

- Training data set contained 500 examples.
- Validation data set contained 200 examples.
- Test data set contained 300 examples. 

Convolutional network uses fewer parameters than a regular network. <br>
Regular network has (in our evaluation process):<br>
`784 * 100 + 100 * 10 + 100 + 10 = 79510` <br>
Convolutional network uses: <br>
`795 * 10 + 2 * 144 + 3 * 169 + 10 + 5 = 8760` <br>
We see significant difference in the amount of parameters. <br>
Convolutional network uses only 5 filters.
Despite this it is able to perform just fine in comparison to fully-connected network.
I guess that if the convolution operation was not that slow, 
convolutional network would outperform fully-connected network. <br>
Final note: never use self-implemented neural networks, especially designed
with sophisticated ideas and algorithms.
