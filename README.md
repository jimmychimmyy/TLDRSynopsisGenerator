# “TLDR” Plot Synopsis Generator
The goal of this project was to learn a model that could generate semantically accurate plot synopses from movie plot summaries. Four seq2seq models were trained and their performance was evaluating by comparing the similarity between words in the target sentences and generated sentences. While none of the four models have achieved immediate accurate results, the seq2seq models show great promise for generating semantically accurate plot synopses.

The code used in this project is based on the source code from the Pytorch Sequence to Sequence Translation Tutorial. The code was extended to use a bidirectional encoder.

### Getting Started
```
pip install requirements.txt
python3 main.py
```

### Diagram
![Diagram](https://github.com/jimmychimmyy/tldr_synopsis_generator/blob/master/diagram.png)

### How It Works
I will be describing how one instance passes through the model. Each Wi is a word from an input plot summary where there are n total words. Starting with W1, a word is passed into an embedding layer which creates a 50-dimensional word embedding. This word embedding and a randomly initialized hidden state become the input for a GRU cell which is colored blue in the diagram. This first GRU cell outputs a hidden state which along with the next word embedding (which is W2 in this case) become the input for the next GRU cell. This process repeats n times for which the final output is the final hidden state. (For the bidirectional encoder, the input plot synopsis is also fed backwards in the same manner producing another final hidden state. Then both hidden states are concatenated before being passed into the decoder).

The decoder takes the input of the final hidden state and Vi the next word in the target plot synopsis and outputs a vector. (Note: the decoder also takes an attention input which is not depicted in the diagram). This output vector is m-dimensional where m is the total number of unique words in the corpus. This vector is then passed through a softmax function which returns another m-dimensional vector in which each element is the probability of a word being the next target word (we use index2word to determine what the word Vi is). This process is repeated until the predicted word Vi  is <EOS> which marks the end of the sentence.

The final step is the backpropagation step where we use stochastic gradient descent to optimize the weights in the encoder and decoder (and attention mechanism).

### Example
![example1](https://github.com/jimmychimmyy/tldr_synopsis_generator/blob/master/example1.png)

![example2](https://github.com/jimmychimmyy/tldr_synopsis_generator/blob/master/example2.png)

### Discussion

Initially, I treated this problem similar to a language translation problem: a model would take in some input sentence in language X and output a corresponding sentence in language Y. In the case of plot synopsis generation however, there were a few notable differences. The first difference that the length of the input to output varies much more than language translation. In language translation, the difference in length between two corresponding sentences is at most a few words. For my models the input length was between 280 and 5000 characters while the output length was between 25 and 280 characters.

An obvious issue is the size of the dataset used to train the model. After cleaning and preprocessing the data the dataset may have been too small to achieve good results. A good next step for this project would be to gather plot summaries and plot synopsis directly from IMDB where there are over 1 million examples.

Aside from the size of the dataset, another possible issue may have been how different each plot summary and plot synopsis were in terms of writing style. This may have been prevented the models from learning a good mapping from summaries to synopsis. There are various genres of movies within the CMU dataset ranging from Hollywood to Bollywood films and everything in between. Because of how diverse these films are, the language in each plot summary and target plot synopsis is very different from one another.
Another issue is the evaluation metric chosen for these models. Summing word embeddings and finding the cosine distance between two word embeddings leaves a lot to be desired. For one thing, this method is indifferent to the ordering of words. Two sentences, “I love movies” and “Movies love I”, would receive identical scores even if the second sentence is ungrammatical and makes no real sense. A possible solution to this would be to compare n-grams of two sentences (This was not done due to the time constraint).

Finally, I did not tune hyperparameters for any of the models. I chose to train each model using the same hyperparameters to find out which model would have the best performance. Doing a simple grid search in the future may help the model learn the mapping better.

All things considered, the seq2seq model shows great promise for mapping plot summaries to plot synopses.

### References

* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
* https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
