---
layout: post
title: "Linear Model"
---

In this tutorial, you will discover how to use text classification model on top of features that we have described

We will develop the model based on sentiment classification. To do that let us take a sample dataset of IMDB movie reviews dataset, that is freely available. It contains 25,000 positive and 25,000 negative reviews. We will use this dataset of movie reviews from IMDB website for text classification. In this dataset, we have text and number of stars, and we can actually think of stars as sentiment. If we have at least seven stars, we can label it as positive sentiment. If it has atmost 4 stars we can label it as negative sentiment. It contains at most 30 reviews per movie just to make it less biased for any particular movie. These dataset also provides a 50/50 train test split so that future researchers can use the same split and reproduce their result and enhance the model. For evaluation, we can use accuracy because our dataset is balanced in terms of the size of the classes so we can evaluate accuracy here.

Let's start with first model, by taking features from bag of 1-grams with TF-IDF values. And in the result, we will have a matrix of features, 25,000 rows and 75,000 columns, and that is a pretty huge feature matrix and what is more, it is extremely sparse. If you look at how many 0s are there, then you will see that 99.8% of all values in that matrix are 0s. So that actually applies some restrictions on the models that we can use on top of these features.

Linear Classification Model

The first model that is usable for these features is logistic regression, which works like the following. It tries to predict the probability of a review being a positive one given the features that we gave that model for that particular review. And the features is the vector of TF-IDF values. And what we will actually try to do is find the weight for every feature of that bag of words representation. Then, we can multiply each value, each TF-IDF value by that weight, sum all of those things and pass it through a sigmoid activation function. This the basic modeling process in logistic regression. Since logistic regression is actually a linear classification model and it can easily handle sparse data, it is fast to train and what's more, the weights that we get after the training can be interpreted.

Let's us take a look at sigmoid graph for deeper understanding. If we have a linear combination that is close to 0, that means that sigmoid will output 0.5. So the probability of review being positive is 0.5. So we really don't know whether it's positive or negative. But if that linear combination in the argument of our sigmoid fuction starts to become more and more positive, so it goes further away from zero. Then you see that the probability of a review being positive actually grows really fast. And that means that if we get the weight of our features that are positive, then those weights will likely correspond to the words that a positive. And if you take negative weights, they will correspond to the words that are negative like disgusting or awful.


If we train logistic regression over bag of 1-grams with TF-IDF values, we will observe that accuracy on test set is 88.5% and is a huge jump from a random classifier which outputs 50% accuracy. If we look at learnt features we will see that at top positive weights correspond to words such as great, excellent, perfect, best, wonderful. So it's really cool becuase the model captured that sentiment, the sentiment of those words, and it knows nothing about English language, it knows only the examples that we provided it with. And if take top negative ways, then we will see words like worst, awful, bad, waste, boring, and so forth. 

Let's us introduce 2-grams to our model

Further, we will try to make our model a little better, by throwing away some n-grams that are not frequent that are seen less than 5 times, as those are likely either typos or very rare words. We will also threshold for minimum frequency. Now, we will see that we have 25000 rows, 156821 columns for training. Now we will train logistic regression over bag of 1, 2-grams with TF-IDF values we will observe that it gets the accuracy of 89.9% and if we take a look at learnt weights we will see that our 2-grams are actually used by our model and words with high weights are well worth, better than, the worst. We can further make it better by :

* Playing around with tokenization: special tokens like emoji, :), !!!
* Try to normalize tokens by stemming or lemmatization
* Try different models like SVM, Naive Bayes
* Throw BOW away and use Deep Learning
