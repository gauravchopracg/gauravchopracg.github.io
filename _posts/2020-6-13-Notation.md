In this tutorial, you will discover notations to build sequence models.

Let's say you want to build a sequence model to automatically tell you what are the people
names in the sentence. Consider an input sentence like this, "Harry Potter and Hermione Granger 
invented a new spell". So, this is a problem called Named-Entity Recognition and this is used
by search engines. Now given this input x, you want a model to predict y that has one outputs
per input word and the target output the design y tells you for each of the input word is that
part of a person's name.

For a given input sentence:

x: Harry Potter and Hermoine Granger invented a new spell.

model predicts y

y: 1  1 0 1 1 0 0 0 0

Now, input is the sequence of nine words, so we have nine sets of features to represent 
these nine words, and index into the positions and sequence. To do that I'm going to use
x superscript 1, 2, 3 upto 9 to index different positions

x: Harry Potter and Hermione Granger invented a new spell.
   $x^{1}$ $x^{2}$ $x^{3}$ ....................... $x^{9}$

so, I'm going to use $x^{t}$ with the index t to index into positions into middle of the sequences and t implies temporal sequences. Similarly for outputs, I'm going to refer outputs as:

y: 1  1  0  1  1  0  0  0  0
   $y^{1}$ $y^{2}$ $y^{3}$ ..... $y^{9}$

Let's also use $T_x$ to denote the length of the input sequence, In this there are nine words
$T_x$ = 9

and we will use $T_y$ to denote the length of the output sequence which is also 9. In this example $T_x$ is equal to $T_y$ 
We will also use $X^(i)$ to denote the ith training example so, to refer to 't'th element in the sequence of training example i will use $X^(i)<t>$

Whereas $T_x$ will denote the length of a sequence then different examples in your training set can have different lengths. So, $T_x^(i)$ would be the input sentence length for training example i

As $y^(i)<t>$ means the 't'th element in the output sequence of training example and $T_y^(i) will be the length of the output sequence in the ith training example

$x^{1}$ $x^{2}$ $x^{3}$


* dists: Shape as (#test, #train). dists[i, j] means the distance of $i^{th}$ test sample to $j^{th}$ training sample.

### Representing Words
To represent a word in the sequence the first thing you do is come up with a vocabulary which is sometimes also called a dictionary and which it does is make a list of words that we will use in the representations.
After that we will use those representations to one-hot encode the vectors. The vocabulary will consist of most common words but if we encounter a word that is not in the vocabulary. We will create a new fake word called Unknown Word (UNK) to represent words not in the vocabulary.
