---
layout: post
title: "Text-Preprocessing" 
---

In this tutorial, you will discover what is text preprocessing and how to pre-process text before extracting features from it or feeding it into a machine learning model.

*You can run the tutorial yourself at [here](https://colab.research.google.com/drive/1Yj4PQ1AbjlWuO8OSjpm-gDc0tTFuwkVK?usp=sharing).*

*The GitHub links for this tutorial are: [Browse](https://github.com/gauravchopracg/blog), [Zip](https://github.com/gauravchopracg/blog/archive/master.zip).*

### Text Processing

Text preprocessing is the process of cleaning the data, preparing the data to be used for machine learning. The text preprocessing steps consider in this tutorial are:

1. Tokenization
2. Text Normalization: Lower-casing, stemming and lemmatization

### Tokenization

The first step in text preprocessing is Tokenization i.e., process of splitting an input text into meaningful chunks. We can think of text as a sequence of words and further word as a meaningful sequence of characters (We can also think text as a sequence of characters or phrases but right now we will consider words as a part of text for simpler understanding). In that way, we need to first split text into small chunks which we will call tokens, a token is a useful unit for further semantic processing. We can split text by whitespaces, by punctuation or any set of rules specifically to that task. In this tutorial you will discover three ways of doing that:

1. By white spaces
2. By Punctuation
3. By rules of English grammar

### Split by white spaces

To split the text into tokens or meaningful words using white spaces, we can use python library NLTK, it offers different classes of tokenizer which we can use it to split text into meaningful chunks for example to splits the input sequence on white spaces, that could be a space or any other character that is not visible. We can use nltk.tokenize.WhitespaceTokenizer() function and simply pass the text: This is Andrew’s text, isn’t it?

| This | is  | Andrew's | text, | isn't | it? |
| ---- | --- | -------- | ----- | ----- | --- |
| This | is  | Andrew's | text, | isn't | it? |
| ---- | --- | -------- | ----- | ----- | --- |

However, the problem is ‘text,’ and ‘text’ are two different words for tokenizer similarly ‘it’ and ‘it?’ we might want to merge these two tokens because they have essentially the same meaning,.

### Split by punctuation

Similarly as before, we can split the text by punctuation using nltk.tokenize.WordPunctTokenizer() and the result will be:


 | This | is  | Andrew |  '  |  s  | text |  ,  | isn |  '  |  t  |  it |  ?  |
 | ---- | --- | ------ | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
 | This | is  | Andrew |  '  |  s  | text |  ,  | isn |  '  |  t  |  it |  ?  |
 | ---- | --- | ------ | --- | --- | ---- | --- | --- | --- | --- | --- | --- |

the problem is ‘s’, ‘isn’ ‘t’ are not very meaningful and punctuation are different tokens hence, it doesn’t make sense to analyze them

### By set of heuristics

We can also come up with a set of rules or heuristics which can be easily found in TreebankWordTokenizer and it actually uses grammar rules of english language to make it tokenization that actually makes sense for further analysis. In reality this is very close to perfect tokenization that we want for English language

| This |  is | Andrew | 's  | text |  ,  | is  | n't |  it |  ?  |
| ---- | --- | ------ | --- | ---- | --- | --- | --- | --- | --- | 
| This |  is | Andrew | 's  | text |  ,  | is  | n't |  it |  ?  |
| ---- | --- | ------ | --- | ---- | --- | --- | --- | --- | --- | 

### Text Normalization

The next thing we might want to do is token normalization. We may want the same token for different forms of the word. like wolf, wolves -> wolf or talk, talks -> talk. The process of normalizing the words into same form is called Text Normalization. They consist of:

1. Stemming
2. Lemmatization

### Stemming

Stemming is the process of removing and replacing suffixes to get to the root form of the word, which is called the stem. It is usually refers to heuristics that chop off suffixes. To apply stemming, we can use NLTK library function nltk.stem.PorterStemmer(), it is the oldest stemmer for English language. It has five heuristic phases of word reductions applied sequentially.

### Lemmatization

Lemmatization is usually refers to doing things properly with the use of a vocabulary and morphological analysis. It returns the base or dictionary form of a word, which is known as the lemma. To apply lemmatization, we can use NLTK library function nltk.stem.WordNetLemmatizer(), it uses WordNet Database to lookup lemmas.

it works like this:

feet -> foot, cats -> cat, wolves -> wolf, talked -> talked

The problem lies in that it does not reduce all forms and we need to try both stemming or lemmatization to decide which is best for our task

### Challenges

There is couple of problems that we need to deal with while normalization of words. Let’s us take capital letters consider Us and us written in different forms and if they are both different forms then we can safely reduce it to the word, us but if us and US (country name in capital form) are both exist in text and if text is written in capital letter that makes it difficult to normalize for that we can use heuristics:

* Lowercase the beginning of each sentence
* Lowercase word in titles
* leave mid-sentence words as they are

Further, if we want we can also use machine learning to retrieve true casing. We need to also normalize acronyms eta, e.t.a, E.T.A -> E.T.A or we can write a bunch of regular expression by hand to take care of that but it requires planning all the possible forms that can be occurring in the text.
