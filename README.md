# “TLDR” Plot Synopsis Generator

### Abstract
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

### Examples
![example1](https://github.com/jimmychimmyy/tldr_synopsis_generator/blob/master/example1.png)

![example2](https://github.com/jimmychimmyy/tldr_synopsis_generator/blob/master/example2.png)
