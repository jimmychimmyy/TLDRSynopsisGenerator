#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import os
import re
import time
import math
import random
from pathlib import Path

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_LEN = 5000
MAX_LEN_SYNOPSIS = 280
MIN_LEN_SYNOPSIS = 25

SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on", device, "...")

learning_rate = 1e-3

CONTINUE_TRAINING = False

MODEL_NAME = "./model.ckpt"

# TODO: may keep - and replace them with space


# ## Check if model is saved, if so load it and continue training

# In[12]:


def checkIfModelExists(model):
    model = Path(model)
    if model.is_file():
        return True
    return False

CONTINUE_TRAINING = checkIfModelExists(MODEL_NAME)


# ## Load datasets

# In[13]:


def loadData():
    dataset = './data/dataset.csv'
    train = './data/train.csv'
    test = './data/test.csv'
    cv = './data/cv.csv'
    df_all = pd.read_csv(dataset, sep='\t', encoding='utf-8')
    df_train = pd.read_csv(train, sep='\t', encoding='utf-8')
    df_test = pd.read_csv(test, sep='\t', encoding='utf-8')
    df_cv = pd.read_csv(cv, sep='\t', encoding='utf-8')
    return df_train, df_test, df_cv, df_all

df_train, df_test, df_cv, df_all = loadData()
#df_train.head()


# In[14]:


print("number of total instances:", df_all.shape[0])
print("number of training instances:", df_train.shape[0])
print("number of testing instances:", df_test.shape[0])
print("number of cv instances:", df_cv.shape[0])


# ## Prepare Training Data: create word embeddings
#
# TODO: try using pretrained word2vec

# In[15]:


# create word embeddings
class Corpus:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # start by counting sos and eos

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[34]:


def prepareData(df_all, df_train, df_test, df_cv):
    print("preparing data")
    corpus = Corpus()
    pairs_train = []
    pairs_test = []
    pairs_cv = []
    for index, row in df_all.iterrows():
        corpus.addSentence(row['MoviePlot'])
        corpus.addSentence(row['Synopsis'])
    for index, row in df_train.iterrows():
        pairs_train.append([row['MoviePlot'], row['Synopsis'], row['MovieTitle']])
    for index, row in df_test.iterrows():
        pairs_test.append([row['MoviePlot'], row['Synopsis'], row['MovieTitle']])
    for index, row in df_cv.iterrows():
        pairs_cv.append([row['MoviePlot'], row['Synopsis'], row['MovieTitle']])
    print("Words in corpus:", corpus.n_words)
    print("Number of training pairs:", len(pairs_train))
    print("Number of testing pairs:", len(pairs_test))
    print("Number of cv pairs:", len(pairs_cv))
    return corpus, pairs_train, pairs_test, pairs_cv

corpus, pairs_train, pairs_test, pairs_cv = prepareData(df_all, df_train, df_test, df_cv)
#print(random.choice(pairs))


# In[35]:


def indexesFromSentence(corpus, sentence):
    return [corpus.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(corpus, sentence):
    indexes = indexesFromSentence(corpus, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# should only have one shared corpus
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(corpus, pair[0])
    target_tensor = tensorFromSentence(corpus, pair[1])
    return (input_tensor, target_tensor)


# In[41]:


# get count of vocab, needed for pretrained word embeddings
target_vocab = set()
target_vocab.add(SOS_token)
target_vocab.add(EOS_token)
df_all['MoviePlot'].str.split().apply(target_vocab.update)
df_all['Synopsis'].str.split().apply(target_vocab.update)
print("size of target vocab:", len(target_vocab))


# ## Using Pretrained Word Embeddings (GloVe)

# In[37]:


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

vocab = loadGloveModel('glove.6B/glove.6B.50d.txt')


# In[38]:


# for each word, check if in glove, else init random vector
matrix_len = len(target_vocab)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

emb_dim = 50 #?

for i, word in enumerate(target_vocab):
    try:
        weights_matrix[i] = vocab[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))


num_embeddings, embedding_dim = weights_matrix.shape
print("Number of embeddings:", num_embeddings)
print("Embeddings dimension:", embedding_dim)


# In[39]:


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight = nn.Parameter(torch.Tensor(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


# ## Define Encoder-Decoder GRU Network
#
# TODO: try both GRU and LSTM to see which performs better
#
# TODO: change learning rate

# In[40]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, use_pretrained=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_pretrained = use_pretrained
        #self.hidden_size = hidden_size / self.num_directions

        if self.use_pretrained:
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)
            self.gru = nn.GRU(embedding_dim, hidden_size, bidirectional=self.bidirectional)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size,  bidirectional=self.bidirectional)


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)

        #packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, len(input))
        #packed_outputs, hidden = self.gru(packed_emb, hidden)
        #packed_outputs, hidden = self.gru(embedded)
        #output, output_lens = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        output = embedded
        output, hidden = self.gru(output, hidden)

        #if self.bidirectional:
        #    hidden = self.catHidden(hidden)

        return output, hidden

    def initHidden(self):
        if self.bidirectional:
            return torch.zeros(2, 1, self.hidden_size, device=device)
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def catHidden(self, hidden):

        def cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # hidden contains a tuple (hidden state, cell state)
            #hidden = tuple([torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for h in hidden])
            hidden = tuple([cat(h) for h in hidden])
        else:
            # GRU hidden
            #hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            hidden = cat(hidden)
        return hidden


# In[68]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[69]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, encoder, dropout_p=0.1, max_len=MAX_LEN):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size * encoder.num_directions
        self.output_size = output_size
        self.num_layers = encoder.num_layers
        self.dropout_p = dropout_p
        self.max_length = max_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        #hidden = hidden[-self.num_layers:]
        hi = torch.cat((embedded[0], hidden[0]), 1)
        #print(hi.shape)
        #print(embedded[0].shape)
        #print(hidden[0].shape)
        #print(hidden[1].shape)
        #print(hidden.shape)

        attn_weights = F.softmax(self.attn((hi)), dim=1)
        #print("attn_weights", attn_weights.shape)
        #print("encoder outputs", encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        #print("attn_applied", attn_applied.shape)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        #print("output", output.shape)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ## Training

# In[71]:


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LEN):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    #print(encoder_hidden.shape)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)
    #print("encoder_outputs", encoder_outputs.shape)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        #encoder_output = (encoder_output[:, :, encoder.hidden_size:] + encoder_output[:, :, :encoder.hidden_size])
        #print("encoder_output", encoder_output.shape)
        #print(encoder_outputs.shape)
        encoder_outputs[ei] = encoder_output[0, 0]

    #print("encoder_outputs", encoder_outputs.shape)
    decoder_input = torch.tensor([[SOS_token]], device=device)

    #decoder_hidden = encoder_hidden[-decoder.num_layers:]
    #decoder_hidden = encoder_hidden.resize(1, 1, 512)
    decoder_hidden = encoder_hidden.resize(1, 1, decoder.hidden_size)
    #print("decoder_hidden", decoder_hidden.shape)
    #decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# In[72]:


def asMinutes(s):
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[73]:


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, iteration=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # reset every print_every
    plot_loss_total = 0 # reset every plot_every

    # using stochastic gradient descent
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs_train)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    print_loss_avg = 0

    for i in range(iteration, n_iters+1):
        training_pair = training_pairs[i - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Time Elapsed: %s | Iteration: %d/%d | Percent Complete: %d%% | Loss: %.4f'
                  % (timeSince(start, i / n_iters),
                                         i, n_iters, i / n_iters * 100, print_loss_avg))
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'loss': print_loss_avg,
                'iteration': i,
            }, 'model.ckpt')
            #print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         #i, i / n_iters * 100, print_loss_avg))

        # am not doing plot loss currently

    # final save
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
        'loss': print_loss_avg,
        'iteration': n_iters+1,
    }, 'model.ckpt')



# ## Evaluation

# In[74]:


def evaluate(encoder, decoder, sentence, max_length=MAX_LEN):
    with torch.no_grad():
        input_tensor = tensorFromSentence(corpus, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(corpus.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# In[42]:


def evaluateRandomly(encoder, decoder, n=10, test_or_cv="test"):
    for i in range(n):
        if test_or_cv == "test":
            pair = random.choice(pairs_test)
        else:
            pair = random.choice(pairs_cv)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# In[43]:

def evaluateSimilarity(target, gen, corpus, weights_matrix):
    target_idx = indexesFromSentence(corpus, target)
    gen_idx = indexesFromSentence(corpus, gen)

    target_emb = np.zeros(50)
    gen_emb = np.zeros(50)

    for i in target_idx:
        target_emb += weights_matrix[i]

    for i in gen_idx:
        gen_emb += weights_matrix[i]

    target_emb = list(map(sum, zip(target_emb)))
    gen_emb = list(map(sum, zip(gen_emb)))

    result = 1 - spatial.distance.cosine(target_emb, gen_emb)
    return result

def meanSimilarity(pairs, corpus, weights_matrix):
    result = 0
    for pair in pairs:
        result += evaluateSimilarity(pair[0], pair[1], corpus, weights_matrix)
    return result / len(pairs)

def evaluateToyStory(encoder, decoder, summary=False):
    plot_summary = "woody is a pull string cowboy doll and leader of a group of toys that belong to a boy named andy davis, which act lifeless when humans are present. with his family moving homes one week before his birthday, the toys stage a reconnaissance mission to discover andys new presents. andy receives a space ranger buzz lightyear action figure, whose impressive features see him replacing woody as andys favorite toy. woody is resentful, especially as buzz also gets attention from the other toys. however buzz believes himself to be a real space ranger on a mission to return to his home planet, as woody fails to convince him he is a toy. andy prepares for a family outing at the space themed pizza planet restaurant with buzz. woody attempts to be picked by misplacing buzz. he intends to trap buzz in a gap behind andys desk, but the plan goes disastrously wrong when he accidentally knocks buzz out the window, resulting in him being accused of murdering buzz out of jealousy. with buzz missing, andy takes woody to pizza planet, but buzz climbs into the car and confronts woody when they stop at a gas station. the two fight and fall out of the car, which drives off and leaves them behind. woody spots a truck bound for pizza planet and plans to rendezvous with andy there, convincing buzz to come with him by telling him it will take him to his home planet. once at pizza planet, buzz makes his way into a claw game machine shaped like a spaceship, thinking it to be the ship woody promised him. inside, he finds squeaky aliens who revere the claw arm as their master. when woody clambers into the machine to rescue buzz, the aliens force the two towards the claw and they are captured by andys neighbour sid phillips, who finds amusement in destroying toys. at sids house, the two attempt to escape before andys moving day, encountering sids nightmarish toy creations and his vicious dog, scud. buzz sees a commercial for buzz lightyear action figures and realizes that he really is a toy. attempting to fly to test this, buzz falls and loses one of his arms, going into depression and unable to cooperate with woody. woody waves buzzs arm from a window to seek help from the toys in andys room, but they are horrified thinking woody attacked him, while woody realizes sids toys are friendly when they reconnect buzzs arm. sid prepares to destroy buzz by strapping him to a rocket, but is delayed that evening by a thunderstorm. woody convinces buzz that life is worth living because of the joy he can bring to andy, which helps buzz regain his spirit. cooperating with sids toys, woody rescues buzz and scares sid away by coming to life in front of him, warning him to never torture toys again. woody and buzz then wave goodbye to the mutant toys and return home through a fence, but miss andys car as it drives away to his new house. down the road, they climb onto the moving truck containing andys other toys, but scud chases them and buzz tackles the dog to save woody. woody attempts to rescue buzz with andys rc car but the other toys, who think woody now got rid of rc, toss woody off onto the road. spotting woody driving rc back with buzz alive, the other toys realize their mistake and try to help. when rcs batteries become depleted, woody ignites the rocket on buzzs back and manages to throw rc into the moving truck before they soar into the air. buzz opens his wings to cut himself free before the rocket explodes, gliding with woody to land safely into a box in andys car. andy looks into it and is elated to have found his two missing toys. on christmas day at their new house, buzz and woody stage another reconnaissance mission to prepare for the new toy arrivals, one of which is a mrs. potato head, much to the delight of mr. potato head. as woody jokingly asks what might be worse than buzz, the two share a worried smile as they discover andys new gift is a puppy."
    summary_summary = "however buzz believes himself to be a real space ranger on a mission to return to his home planet, as woody fails to convince him he is a toy. woody spots a truck bound for pizza planet and plans to rendezvous with andy there, convincing buzz to come with him by telling him it will take him to his home planet. when woody clambers into the machine to rescue buzz, the aliens force the two towards the claw and they are captured by andys neighbour sid phillips, who finds amusement in destroying toys. woody and buzz then wave goodbye to the mutant toys and return home through a fence, but miss andys car as it drives away to his new house. spotting woody driving rc back with buzz alive, the other toys realize their mistake and try to help. on christmas day at their new house, buzz and woody stage another reconnaissance mission to prepare for the new toy arrivals, one of which is a mrs."
    target = "a cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boys room"
    if summary:
        output_words, attentions = evaluate(encoder, decoder, summary_summary)
        print('>', summary_summary)
    else:
        output_words, attentions = evaluate(encoder, decoder, plot_summary)
        print('>', plot_summary)
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')


# ## Train and Evaluate
#
# TODO try different hidden_size

# In[ ]:


## USING ATTENTION DECODER
# Note:: If you run this notebook you can train, interrupt the kernel, evaluate, and continue training later.
# Comment out the lines where the encoder and decoder are initialized and run trainIters again.

hidden_size = 256
#hidden_size = 512

if CONTINUE_TRAINING:
    print("resuming training")
    checkpoint = torch.load("./model.ckpt")

    # what happens if i remove .to(device)?
    encoder = EncoderRNN(corpus.n_words, hidden_size, bidirectional=True, num_layers=2).to(device)
    #decoder = DecoderRNN(hidden_size, corpus.n_words).to(device)
    decoder = AttnDecoderRNN(hidden_size, corpus.n_words, encoder, dropout_p=0.1).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    i = checkpoint['iteration']

    # do i need this if i keep .to(device) in earlier lines?
    if device == "cuda":
        encoder.to(device)
        decoder.to(device)

    trainIters(encoder, decoder, df_train.shape[0]*2, print_every=100, iteration=i)
else:
    print("starting training")
    encoder = EncoderRNN(corpus.n_words, hidden_size, bidirectional=True, num_layers=2).to(device)
    #decoder = DecoderRNN(hidden_size, corpus.n_words).to(device)
    decoder = AttnDecoderRNN(hidden_size, corpus.n_words, encoder, dropout_p=0.1).to(device)
    trainIters(encoder, decoder, df_train.shape[0]*2, print_every=100)

evaluateRandomly(encoder, decoder)
evaluateToyStory(encoder, decoder)

print("mean similarity train", meanSimilarity(pairs_train, corpus, weights_matrix))
print("mean similarity test", meanSimilarity(pairs_test, corpus, weights_matrix))
print("mean similarity cv", meanSimilarity(pairs_cv, corpus, weights_matrix))

# ## USING ATTENTION DECODER
# # Note:: If you run this notebook you can train, interrupt the kernel, evaluate, and continue training later.
# # Comment out the lines where the encoder and decoder are initialized and run trainIters again.
#
# hidden_size = 256
# #hidden_size = 512
#
# if CONTINUE_TRAINING:
#     print("resuming training")
#     checkpoint = torch.load("./attn_model_glove.ckpt")
#
#     # what happens if i remove .to(device)?
#     encoder = EncoderRNN(corpus.n_words, hidden_size).to(device) # i changed all corpus.n_words to len(target_vocab)
#     decoder = decoder = AttnDecoderRNN(hidden_size, corpus.n_words, dropout_p=0.1).to(device)
#     encoder.load_state_dict(checkpoint['encoder_state_dict'])
#     decoder.load_state_dict(checkpoint['decoder_state_dict'])
#     i = checkpoint['iteration']
#
#     # do i need this if i keep .to(device) in earlier lines?
#     if device == "cuda":
#         encoder.to(device)
#         decoder.to(device)
#
#     # dont't think these train functions are doing anything
#     #encoder.train()
#     #decoder.train()
#     trainIters(encoder, decoder, df_train.shape[0], print_every=100, iteration=i) # i set to 75000 iterations
# else:
#     print("starting training")
#     encoder = EncoderRNN(corpus.n_words, hidden_size).to(device)
#     decoder = AttnDecoderRNN(hidden_size, corpus.n_words, dropout_p=0.1).to(device)
#     trainIters(encoder, decoder, df_train.shape[0], print_every=100)
#
# evaluateRandomly(encoder, decoder)

# ## USING BASIC DECODER
# #### Note:: If you run this notebook you can train, interrupt the kernel, evaluate, and continue training later.
# #### Comment out the lines where the encoder and decoder are initialized and run trainIters again.
#
# hidden_size = 256
# #hidden_size = 512
#
# if CONTINUE_TRAINING:
#     print("resuming training")
#     checkpoint = torch.load("./model.ckpt")
#
#     # what happens if i remove .to(device)?
#     encoder = EncoderRNN(corpus.n_words, hidden_size).to(device)
#     decoder = DecoderRNN(hidden_size, corpus.n_words).to(device)
#     encoder.load_state_dict(checkpoint['encoder_state_dict'])
#     decoder.load_state_dict(checkpoint['decoder_state_dict'])
#     i = checkpoint['iteration']
#
#     # do i need this if i keep .to(device) in earlier lines?
#     if device == "cuda":
#         encoder.to(device)
#         decoder.to(device)
#
#     # dont't think these train functions are doing anything
#     #encoder.train()
#     #decoder.train()
#     trainIters(encoder, decoder, df_train.shape[0], print_every=100, iteration=i)
# else:
#     print("starting training")
#     encoder = EncoderRNN(corpus.n_words, hidden_size).to(device)
#     decoder = DecoderRNN(hidden_size, corpus.n_words).to(device)
#     trainIters(encoder, decoder, df_train.shape[0], print_every=100)
#
# evaluateRandomly(encoder, decoder)

# torch.save({
#     'encoder_state_dict': encoder.state_dict(),
#     'decoder_state_dict': decoder.state_dict()
# }, 'model.ckpt')

# ## Visualize Attention

# In[ ]:
