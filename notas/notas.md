# Deep Feedforward Networks
The goal of a feedforward network is to approximate some funcion $f^*$. A feedforward network defines a mapping $\bm{y}=f(\bm{x};\bm{\theta})$ and learns the value of the parameters $\bm{\theta}$ that result in the best function approximation.

These models are called **feedforward** because information flows through the function being evaluated from $\bm{x}$, through the intermediate computations used to define $f$, and finally to the ouput $\bm{y}$. There are no **feedback** connections in which outputs of the model are fed back into itself.

The model is associated with a directed acyclic graph describing how the functions are composed together. For example, we might have three funcitons $f^{(1)}$, $f^{(2)}$, and $f^{(3)}$ connected in a chain to form $f(\bm{x})=f^{(3)}(f^{(2)}(f^{(1)}(\bm{x})))$. These chain structures are the most commonly used structures of neural networks. In this case, $f^{(1)}$ is called the **first layer** of the network, $f^{(2)}$ is called the **second layer**, and so on. The overall length of the chain gives the **depth** of the model. It is from this terminology that the name "deep learning" arises. The final layer of a feedforward network is called the **output layer**. During neural network training, we drive $f(\bm{x})$ to match $f^{*}(\bm{x})$. The training data provides us with noisy, approximate examples of $f^{*}(\bm{x})$ evaluated at different training points. Each example $\bm{x}$ is accompanied by a label $y\approx f^{*}(\bm{x})$. The training examples specify directly what the output layer must do at each point $\bm{x}$; it must produce a value that is close to $y$. The behavior of the other layers is not directly specified by the training data. Instead, the learning algorithm must decide how to use these layers to best implement an approximation of $f^{*}$. Because the training data does not show the desired output for each of these layers, these layers are called **hidden layers**.

Finally, these networks are called *neural* because they are loosely inspired by neuroscience. Each hidden layer of the network is typically vector-valued. The dimensionality of these hidden layers determines the **width** of the model. Each element of the vector may be interpreted as playing a role analogous to a neuron. Rather than thinking of the layer as representing a single vector-to-vector funciont, we can also think of the layer as consisting of many **units** that act in parallel, each representing a vector-to-scalar function.

To extend linear models to represent nonlinear functions of $\bm{x}$, we can apply the linear model not to $\bm{x}$ itself but to a transformed input $\phi (\bm{x})$, where $\phi$ is a nonlinear transformation. The question is the how to choose the mapping $\phi$. The strategy of deep learning is to learn $\phi$. In this approach, we have a model $y=f(\bm{x};\bm{\theta},\bm{w})=\phi (\bm{x};\bm{\theta})^\top\bm{w}$. We now have parameters $\bm{\theta}$ that we use to learn $\phi$ from a broad class of functions, adn parameters $\bm{w}$ that map from $\phi(\bm{x})$ to the desired output. This approach gives up on the convexity of the training problem, but the benefits outweigh the harms. In this approach, we parametrize the representation as $\phi(\bm{x};\bm{\theta})$ and use the optimization algorithm to find the $\bm{\theta}$ that corresponds to a good representation. Feedforward networkds have introduced the concept of a hidden layer, and this requires us to choose the **activation functions** ($g$), that will be used to compute the hidden layer values.

# Gradient-Based Learning
Learning in deep neural networks requires computing the gradients of complicated functions. The **back-propagation** algorithm can be used to efficiently compute these gradients.

# Architecture Design
Most neural networks are organized into groups of units called layers. Most neural network architectures arrange these layers in a chain structure, with each layer being a function of the layer that preceded it. In this structure, the first layer is given by
$$
\bm{h}^{(1)} = g^{(1)} \left( \bm{W}^{(1)\top} \bm{x} + \bm{b}^{(1)} \right),
$$
the second layer is given by
$$
\bm{h}^{(2)} = g^{(2)} \left( \bm{W}^{(2)\top} \bm{h}^{(1)} + \bm{b}^{(2)} \right),
$$
and so on.

Deeper networks often are able to use far fewer units per layer and far fewer parameters and often generalize to the test set, but are also often harder to optimize. The ideal network architecture for a task must be found via experimentation guided by monitoring the validation set error.

# Universal Approximation
The **universal approximation theorem** states that a feedforward network with a linear output layer and at least one hidden layer with any "squashing" activation function (such as the logistic sigmoid activation function) can approximate any Borel measurable function from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units. For this purpose, the definition of **Borel measurable** function, is any continuous function on a closed and bounded subset of $\mathbb{R}^n$.

# Back-Propagation
When we use a feedforward neural network to accept an input $\bm{x}$ and produce an output $\hat{\bm{y}}$, information flows forward through the network. The inputs $\bm{x}$ provide the initial information that then propagates up to the hidden units at each layer and finally produces $\hat{\bm{y}}$. This is called **forward propagation**. During training, forward propagation can continue onward until it pruduces a scalar cost $J({\bm{\theta}})$. The **back-propagation** algorithm, allows the information from the cost to then flow backwards through the network, in order to compute the gradient.

Computing an analytical expression for the gradient is straightforward, but numerically evaluating such an expression can be computationally expensive. The back-propagation algorithm does so using a simple and inexpensive procedure.

The term back-propagation is often misunderstood as meaning the whole learning algorithm for multi-layer neural networks. Actually, back-propagation refers only to the method for computing the gradient, while another algorithm, such as stochastic gradient descent, is used to perform learning using this gradient.

## Computational Graphs
To describe the back-propagation algorithm more precisely, it is helpful to have a more precise **computational graph** language. Here, we use each node in the graph to indicate a variable. The variable may be a scalar, vector, matrix, tensor, or even a variable of another type.

To formalize our graphs, we also need to introduce the idea of an **operation**. An operation is a simple function of one or more variables. Our graph language is accompanied by a set of allowable operations. Functions more complicated than the operations in this set may be describe by composing operations together.

If a variable $y$ is computed by applying an operation to a variable $x$, then we draw a directed edge from $x$ to $y$. We sometimes annotate the output node with the name of the operation applied.

## Chain Rule of Calculus
Let $x$ be a real number, and let $f$ and $g$ both be functions mapping from a real number to a real number. Suppose that $y=g(x)$ and $z=f(g(x))=f(y)$. Then the chain rule states that
$$
\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}
$$

We can generalize this beyond the scalar case. Suppose that $\bm{x}\in\mathbb{R}^m$, $\bm{y}\in\mathbb{R}^n$, $g$ maps from $\mathbb{R}^m$ to $\mathbb{R}^n$, and $f$ maps from $\mathbb{R}^n$ to $\mathbb{R}$. If $\bm{y}=g(\bm{x})$ and $z=f(\bm{y})$, then
$$
\frac{\partial z}{\partial x_i}=\sum_j\frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i}.
$$
In vector notation, this may be equivalently written as
$$
\nabla_{\bm{x}}z=\left( \frac{\partial\bm{y}}{\partial\bm{x}} \right)^\top\nabla_{\bm{y}}z,
$$
where $\frac{\partial\bm{y}}{\partial\bm{x}}$ is the $n\times m$ Jacobian matrix of $g$.

From this we see that the gradient of a variable $\bm{x}$ can be obtained by multiplying a Jacobian matrix $\frac{\partial \bm{y}}{\partial x\bm{x}}$ by a gradient $\nabla_{\bm{y}}z$. The back-propagation algorithm consists of performing such a Jacobian-gradient product for each operation in the graph.

## Recursively Applying the Chain Rule to Obtain Backprop
Using the chain rule, it is straightforward to write down an algebraic expressino for the gradient of a scalar with respecto to any node in the computational graph that produced that scalar. However, actually evaluating that expression in a computer introduces some extra considerations.

Specifically, many subexpressions may be repeated several times within the overall expression for the gradient. Any procedure that computes the gradient will need to choose whether to store these subexpressions or to recompute them several times.

We first begin by a version of the back-propagation algorithm that specifies the actual gradient computation directly, in the order it will actually be done and according to the recursive application of chain rule.

First consider a computational graph describing how to compute a single scalar $u^{(n)}$ (say the loss on a training example). This scalar is the quantity whose gradient we want to obtain, with respect to the $n_i$ input nodes $u^{(1)}$ to $u^{(n_i)}$. In other words we wish to compute $\frac{\partial u^{(n)}}{\partial u^{(i)}}$ for all $i\in\{1,2,\dots,n_i\}$. In the application of back-propagation to computing gradient descent over parameters, $u^{(n)}$ will be the cost associated with an example or a minibatch, while $u^{(1)}$ to $u^{(n_i)}$ correspond to the parameters of the model.

We will assume that the nodes of the graph have been ordered in such a way that we can compute their output one after the other, starting at $u^{(n_i+1)}$ and going up to $u^{(n)}$. Each node is associated with an operation $f^{(i)}$ and is computed by evaluating the function
$$
u^{(i)}=f(\mathbb{A}^{(i)})
$$
where $\mathbb{A}^{(i)}$ is the set of all nodes that are parents of $u^{(i)}$.

Here is a procedure that performs the computations mapping $n_i$ inputs $u^{(1)}$ to $u^{(n_i)}$ to an output $u^{(n)}$. This defines a computational graph where each node computes numerical value $u^{(i)}$ by applying a function $f^{(i)}$ to the set of arguments $\mathbb{A}^{(i)}$ that comprises the values of previous nodes $u^{(j)}$, $j<i$ with $j\in Pa(u^{(i)})$. The input to the computational graph is the vector $\bm{x}$, and is set into the first $n_i$ nodes $u^{(1)}$ to $u^{(n_i)}$. The output of the computational graph is read off the last (output) node $u^{(n)}$:

**for** $i=1,\dots,n_i$ **do**  
&nbsp;&nbsp;&nbsp;&nbsp;$u^{(k)}\leftarrow x_i$  
**end for**  
**for** $i=n_i+1,\dots,n$ **do**  
&nbsp;&nbsp;&nbsp;&nbsp;$\mathbb{A}^{(i)}\leftarrow\{u^{(j)}|j\in Pa(u^{(i)}) \}$  
&nbsp;&nbsp;&nbsp;&nbsp;$u^{(k)}\leftarrow f^{(i)}(\mathbb{A}^{(i)})$  
**end for**  
**return** $u^{(n)}$

