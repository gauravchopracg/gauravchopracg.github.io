In this tutorial, you will discover how to extract feature from text.

*You can run the tutorial yourself at [here](https://colab.research.google.com/drive/1v0NoKE413L9mp4PoUC7PdvA1u0hzezOg?usp=sharing).*

*The GitHub links for this tutorial are: [Browse](https://github.com/gauravchopracg/blog), [Zip](https://github.com/gauravchopracg/blog/archive/master.zip).*

### Bag of Words (BOW)

In Bag of Words, we count occurrences of a particular token in our text like excellent or disappointed, and we want to detect those words, and make decisions based on absence or presence of that particular word, and how it might work. Let us take an example:

Consider we have three reviews like a good movie, not a good movie, did not like and we will take all the possible words or tokens that we have in our documents and for each such token, let's introduce a new feature or column that will correspond to that particular word. Our matrix will look like this:

|       text       | good | movie | not |  a  |  did | like |
|------------------| ---- | ----- | --- | --- | ---- | ---- |
|     good movie   |   1  |   1   |  0  |  0  |   0  |  0   |
|------------------| ---- | ----- | --- | --- | ---- | ---- |
| not a good movie |   1  |   1   |  1  |  1  |   0  |  0   |
|------------------| ---- | ----- | --- | --- | ---- | ---- |
|   did not like   |   0  |   0   |  1  |  0  |   1  |  1   |

In this matrix, our example good movie review token have the word good, which is present in our text, so we will put one in the column that corresponds to that word then comes word movie, and we put one in the second column just to show that word is actually seen in our text. After that, we don't have any other words so the rest are zeroes. This is called text vectorization, because we actually replace the text with a huge vector of numbers, and each dimension of that vector corresponds to a certain token in our database.

#### Problem in text vectorization:
* We loose word order, because we can actually shuffle over words and the matrix will not change that is why it is called bag of words, because it's a bag in which words aren't ordered, and so they can come up in any order
* Counters are not normalized

### N-grams

Let us try to solve these problem, first by preserving some ordering. We can do this by taking token pairs, triplets, or different combinations. This approach is called as extracting n-grams. One gram stands for tokens, two gram stands for token pairs and so forth. Let's us take again same example and try to extract n-grams, our good movie review now translates into vector which has one in a column corresponding to that token pair good movie for movies which are good and so forth. This is one way to preserve some local word order, but it also have some problems:

* This representation can have too many features, because let's say you have 100,00 words in your database, and if you try to take the pairs of those words, then you can actually come up with a huge number that can exponentially grow with the number of consecutive words that you want to analyze.
* We can easily deal with this problem by removing some n-grams from features based on their occurence frequency in document of our corpus. We have three types of n-grams:

  1. High-frequency n-grams
  2. Medium-frequency n-grams
  3. Low-frequency n-grams

For high-frequency n-grams, if we take a text and observe some of them we will realise that for English language that would be articles, and preposition, and stuff like that. Since, they're just there for grammatical structure and they don't have much meaning. These are called stop-words, they won't help us to discriminate text, and we can pretty easily remove them. 

Similary, for low-frequency n-grams, if we take a look at them we will see they are typos because people type with mistakes, or rare n-grams that's usually not seen in any other text. 

Hence, both of them are bad for the model, because they will make it likely to overfit. 

Therefore, medium frequency n-grams are one really good for training of machine learning model but there're a lot of them and it would be useful we can filter them out and use them for discriminating to capture a specific issue for our model training.

## Term Frequency

It is the frequency of term t in a document d and the term is an n-gram, token, anything like that. There are various type of term frequency:

1. Binary : it compose of zero or one based on the fact whether that token is absent or present in our text.
2. Raw count: it just take a raw count of how many times we've seen that term in our document, and let's denote that by f. Then you can take a term frequency, so you can actually look at all the counts of all the terms that you have seen in your document and you can normalize those counters to have a sum of 1. 
3. Logarithmic normalization
