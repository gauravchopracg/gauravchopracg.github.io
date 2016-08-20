---
layout: post
title: "cs231n assignment 1"
---

Recently I was following an [online course](http://vision.stanford.edu/teaching/cs231n/syllabus.html) on Convolutional Neural Networks (CNN) provided by Stanford. I find it a very nice hands-on material: slides and notes are easy to understand. Purely reading formulations can be confusing sometimes, but practicing experiments helps better understanding what the formulations and the symbols in them are expressing.

The first assignment is about basic assignments. It also includes some practice on vectorization, which may make Python code faster. I don't do the nested version in some assignments, and some of code are half-vectorized.

## Prerequisites
* [Broadcast in numpy](http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting)

## 1. KNN
KNN is the easiest one; this part is still worth doing, because it helps understand vectorization and cross validation.

### Train
In KNN, the process of training is simply remembering `X_train` and `y_train`:

* `X_train`: Shape as (#features, #train). Each column corresponds to a training sample.
* `y_train`: Shape as (#train,). Labels.

### Distances
Essence of KNN lies in computing distances. Input is simply `X`. Output is `dists`.

* `X`: Shape as (#features, #test). Each column corresponds to a test sample.
* `XT`: Shape as (#features, #train). Each column corresponds to a training sample.
* `dists`: Shape as (#test, #train). `dists[i, j]` means the distance of $i^{th}$ test sample to $j^{th}$ training sample. 

For brevity, denote 

* $D$ as \#features
* $X^{(i)}$ as $i^{th}$ column of $X$, i.e. $i^{th}$ sample

Since
$$\eqalign {
dists_{i,j} &= \|X^{(i)} - XT^{(j)}\| \\
&= \sqrt{\sum_{k=0}^{D-1}(X^{(i)}_k - XT^{(j)}_{k})^2} \\
&= \sqrt{\sum_{k=0}^{D-1}{(X^{(i)}_k)^2} + \sum_{k=0}^{D-1}{(XT^{(j)}_k)^2} - 2\sum_{k=0}^{D-1}{X^{(i)}_{k} \cdot XT^{(j)}_{k}}} \\
&= \sqrt{\|X^{(i)}\|^2 + \|XT^{(j)}\|^2 - 2(X^{(i)})^{T} \cdot XT^{(j)} }
}
$$

, we can vectorize it like this:

```python
  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    # vectorize
    ip = X.dot(self.X_train.T) # inner product
    XT2 = np.sum(self.X_train ** 2, axis=1)
    X2 = np.sum(X ** 2, axis=1)
    dists = np.sqrt(-2*ip + XT2 + X2.reshape(-1, 1))
```

### Predict

```python
  from collections import Counter
  def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    
    for i in xrange(num_test):
      # pick nearest neighbors  
      closest_y = self.y_train[np.argsort(dists[i])[:k]]
      
      # count which class appears most
      y_pred[i] = Counter(closest_y).most_common(1)[0][0]
    
    return y_pred
```

### Cross validation

Cross validation is a process to determine hyper-parameters; in this case, `k` is a hyper-parameter. We have `X_train` and `X_test`; now we subdivide `X_train` into `cv_X_train` and `cv_X_test`, where `cv_` means cross validation.

```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.array(np.array_split(X_train, num_folds))
y_train_folds = np.array(np.array_split(y_train, num_folds))

k_to_accuracies = {}
dim = X_train.shape[1]
from cs231n.classifiers import KNearestNeighbor
for k in k_choices:
    k_to_accuracies[k] = [0] * num_folds
    for test_idx in range(num_folds):
        train_ids = [idx for idx in range(num_folds) if idx != test_idx]
        cv_X_train = X_train_folds[train_ids].reshape(-1, dim)
        cv_X_test = X_train_folds[test_idx]
        cv_y_train = y_train_folds[train_ids].reshape(-1)
        cv_y_test = y_train_folds[test_idx]
        
        classifier = KNearestNeighbor()
        classifier.train(cv_X_train, cv_y_train)
        dists = classifier.compute_distances_no_loops(cv_X_test)
        cv_y_test_pred = classifier.predict_labels(dists, k)
        
        num_correct = np.sum(cv_y_test_pred == cv_y_test)
        num_test = len(cv_y_test)
        accuracy = float(num_correct) / num_test
        k_to_accuracies[k][test_idx] = accuracy
        
for k in sorted(k_to_accuracies): # Print out the computed accuracies
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)
```

## 2. SVM

Here I omit the pre-processing part -- subtract the mean image and add an extra dimension '1' to each sample, which has been explained by lecture notes.

Denote

* $x^{i}$ as $i^{th}$ column of $X$, i.e. $i^{th}$ sample
* $w_{j}$ as $j^{th}$ row of $W$
* $N$ as # of training samples
* $C$ as # of classes

Then

* Score function `scores` or `wx`: 
$$
f(x^{(i)}, W) = W x^{(i)}
$$

* Loss function: 
$$
L_{i} = \sum_{j \ne y_{i}}{\max(0, f(x^{(i)}, W)_{j} - f(x^{(i)}, W) + \Delta)}
$$

* Gradient of loss `ddW`: 
$$
\nabla_{w_{j}} L_{i} = \cases {
{\bf 1}(w_{j}x^{(i)} - w_{y_{i}}x^{(i)} + \Delta > 0) \cdot x^{(i)} & \text{, if } j \ne y_{i} \\
-\left( \sum_{j \ne y_{i}}{\bf 1}(w_{j}x^{(i)} - w_{y_{i}}x^{(i)} + \Delta > 0) \right) \cdot x^{(i)} & \text{, otherwise}
}
$$

* Total loss `loss`:
$$
L = \frac{1}{N}\sum_{i}{L_{i}} + \frac{1}{2}\lambda\sum_{k,l}{w_{k,l}^{2}}
$$

* Gradient of total loss `dW`:
$$
\nabla_{W}L = \frac{1}{N}\sum_{i}{\nabla_{W} L_{i}} + \lambda W
$$


_`variable`_s that represent these terms are used in the following snippet.

### Train

Training is to obtain best $W$, minimizing total loss $L$. Here we use gradient descent to find best $W$, and we need to calculate total loss and gradient of it.

#### Loss by nested loop

Outer loop iterates `i` of $L_{i}$ over $N$, corresponding to a specific sample. Inner loop iterates `j` of $W_{j}$ over $C$, corresponding to a specific class.

```python
def svm_loss_naive(W, X, y, reg):
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1

  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    ddW = np.zeros(W.shape)
    ddWyi = np.zeros(W[0].shape)
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        loss += margin
        ddW[j] = X[:, i] ## be careful, it's a reference
        ddWyi += ddW[j]
    ddW[y[i]] = -ddWyi
    dW += ddW
  
  # divided by num_train
  loss /= num_train
  dW /= num_train
  
  # add regularization term
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  return loss, dW
```

#### Loss by half vectorization

Inner loop can be vectorized. Rather than count `margin` loop by loop under `j`, we can count them in a once. It depends on `scores` and `correct_class_score`. `scores` already satisfies, so we'll try to obtain `correct_class_score` for each `j`.

Sorry about naming, but here I name `wx` as `scores`, `judge` as `margin`s. Here's the snippet:

```python
def svm_loss_vectorized(W, X, y, reg):
  dW = np.zeros(W.shape) # initialize the gradient as zero
  wx = W.dot(X)
  delta = 1

  ### loss
  # wxy chooses scores of right labels
  # its shape is (#samples,)
  wxy = [ wx[y[i], i] for i in xrange(wx.shape[1]) ]
  
  # judge expression
  # remember to exclude on y[i]'s
  judge = wx - wxy + delta
  # make judge 0 on y[i]
  for i in xrange(wx.shape[1]):
    judge[y[i], i] = 0
  
  # mass is a matrix holding all useful temp results
  # shape of judge is (#class, #train)
  mass = np.maximum(0, judge)
  
  loss = np.sum(mass) / X.shape[1]
  loss += 0.5 * reg * np.sum(W * W)
```

Now explain on the `dW` part. In each iteration under `i`, `dW` adds an `ddW`. Each row of `ddW` is a weight timing `x[:, i]`. For most rows who's not `y[i]`, weight is 1, i.e. `ddW[i] = x[:, i]`. For the specific row who is `y[i]`, `ddW[i]`'s weight is the negative sum of all other rows. This logic corresponds to the definition of $$\nabla_{w_{j}} L_{i}$$:

$$
\nabla_{w_{j}} L_{i} = \cases {
{\bf 1}(w_{j}x^{(i)} - w_{y_{i}}x^{(i)} + \Delta > 0) \cdot x^{(i)} & \text{, if } j \ne y_{i} \\
-\left( \sum_{j \ne y_{i}}{\bf 1}(w_{j}x^{(i)} - w_{y_{i}}x^{(i)} + \Delta > 0) \right) \cdot x^{(i)} & \text{, otherwise}
}
$$

```python
def svm_loss_vectorized(W, X, y, reg):
  # continue on last snippet
  # weight to be producted by X
  # its shape is (#classes, #samples)
  weight = np.array((judge > 0).astype(int))
  
  # weights on y[i] needs special care
  weight_yi = -np.sum(weight, axis=0)
  for i in xrange(wx.shape[1]):
    weight[y[i], i] = weight_yi[i]
  
  # half vectorized
  for i in xrange(X.shape[1]):
    ddW = X[:, i] * weight[:, i].reshape(-1, 1)
    dW += ddW
    
  dW /= X.shape[1]
  dW += reg * W
  
  return loss, dW
```

#### Comparison of the two-loop version and the half vectorized version

Following the ipython notebook and testing on training dataset with shape (3073, 49000) and 10 labels, training of the two-loop version takes around `7s` and that of half vectorized version takes around `4s`. Actually the complexity of the algorithm doesn't change, i.e. $O \left( N \cdot D \cdot C \right)$; That speed goes up is because Python is slow in for loops, and maybe numpy has done some optimization in matrix multiplication. 

#### Gradient descent

This part is comparatively easy. Having gradient $\nabla_{W}L$, we may just subtract $W$ by $\alpha \nabla_{W}L$ in each iteration, where $\alpha$ is learning rate.

### Predict

The process of prediction is simply choosing the label with highest score. I omit it here.


## 3. Softmax

Keep the symbols defined in SVM, here we have:

* Score function: 
$$ f(x^{(i)}, W) = W x^{(i)} $$

* Output function:
$$ h(x^{(i)}, W) = \frac{e^{f_i}}{\sum_j e^{f_j}} $$

* Loss function: 
$$ L_{i} = -f_{y_{i}} + \log \sum_{j}{e^{f_{j}}} $$

* Gradient function:
$$ \eqalign{
\nabla_{W}L_{i} 
&= -\nabla_{W}f^{(i)}_{y_{i}} + \frac{1}{\sum_{j}e^{f^{(i)}_{j}}} \sum_{j} \left( e^{f^{(i)}_{j}} \cdot \nabla_{W}f^{(i)}_{j} \right) \\
}
$$

, where $$
\nabla_{W_k}L_{f_{j}^{(i)}} = \cases{
x^{(i)} & \text{, if } k = j \\
0 & \text{, otherwise}
}
$$

Or you may refer to [this version](http://deeplearning.stanford.edu/wiki/index.php/Softmax_Regression) of gradient function.

* Total loss:
$$L = \frac{1}{N}\sum_{i}{L_{i}} + \frac{1}{2}\lambda\sum_{k,l}{w_{k,l}^{2}}$$

* Gradient of total loss:
$$\nabla_{W}L = \frac{1}{N}\sum_{i}{\nabla_{W} L_{i}} + \lambda W$$

According to [the lecture note](http://cs231n.github.io/linear-classify/#softmax), here we adjust $$f^{(i)}_{j}$$ to $$f^{(i)}_{j} - \max_{j}{f^{(i)}_{j}}$$. Here's the snippet, very similar to that of SVM:

```python
def softmax_loss_naive(W, X, y, reg):
  dW = np.zeros_like(W)
  
  ### loss
  scores = W.dot(X)
  scores_max = np.max(scores, axis=0)
  scores -= scores_max
  exp_scores = np.exp(scores)
  sums = np.sum(exp_scores, axis=0)
  log_sums = np.log(sums)
  scores_y = np.array([scores[y[i], i] for i in xrange(X.shape[1])])
  loss = np.sum(-scores_y + log_sums)

  loss /= X.shape[1]
  loss += .5 * np.sum(W * W)

  ### dW
  for i in xrange(X.shape[1]):
    # dW += 1./sums[i] * log_sums[i] * X[:, i]
    dW += 1./sums[i] * exp_scores[:, i].reshape(-1, 1) * X[:, i]
    dW[y[i]] -= X[:, i]

  dW /= X.shape[1]
  dW += reg * W
  
  return loss, dW
```

Note the sigma term
$$\sum_{j} \left( e^{f^{(i)}_{j}} \cdot \nabla_{W}f^{(i)}_{j} \right)$$.
In Python it is interpreted as `exp_scores[:, i].reshape(-1, 1) * X[:, i]`, because for each `j`, there's only one none zero row in $\nabla_{W}f^{(i)}_{j}$, whose value is $e^{f^{(i)}}$ dot product $x^{(i)}$. Shape of $e^{f^{(i)}}$ is `(C,)` and shape of $x^{(i)}$ is `(D,)`. Making use of broadcast, we can reshape $e^{f^{(i)}}$ to `(C, 1)` and make a `(C, D)` matrix; that is what we want for `ddW`.

## Say last
Thank everyone that points out my mistakes. If there's any more mistake, please don't hesitate to tell me. The original purpose of this blog is just to note down my immediate thoughts when following cs231n in my spare time. It's totally written in an amateur's view; I would be happy if it helps you.

The official version of SVM and Softmax is really nice, and comes in later assignments. 
