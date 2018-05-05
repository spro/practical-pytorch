# coding: utf-8

import unicodedata
import string
import re
import random
import time
import datetime
import math
import socket
hostname = socket.gethostname()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import numpy as np

import io
import torchvision
from PIL import Image
import visdom
vis = visdom.Visdom()

USE_CUDA = True

SOS_token = 0
EOS_token = 1

# Configure models
attn_model = 'dot'
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 100
batch_size = 50

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000
epoch = 0
plot_every = 20
print_every = 100
evaluate_every = 1000

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# ----------------------------------------------------------------------------------------

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count=3):
        keep = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep.append(k)

        print('total', len(self.word2index))
        print('keep', len(keep))
        print('keep %', len(keep) / len(self.word2index))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS

        for word in keep:
            self.index_word(word)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
#     filename = '../data/%s-%s.txt' % (lang1, lang2)
    filename = '../%s-%s.txt' % (lang1, lang2)
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MIN_LENGTH = 5
MAX_LENGTH = 20

good_prefixes = (
    "i ", "he  ", "she ", "you ", "they ", "we "
)

def filter_pair(p):
    return len(p[0].split(' ')) <= MAX_LENGTH and len(p[1].split(' ')) <= MAX_LENGTH and         len(p[0].split(' ')) >= MIN_LENGTH and len(p[1].split(' ')) >= MIN_LENGTH # and \
#         p[1].startswith(good_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
input_lang.trim()
output_lang.trim()

keep_pairs = []

for pair in pairs:
    keep_input = True
    keep_output = True

    for word in pair[0].split(' '):
        if word not in input_lang.word2index:
            keep_input = False
            break

    for word in pair[1].split(' '):
        if word not in output_lang.word2index:
            keep_output = False
            break

    if keep_input and keep_output:
        keep_pairs.append(pair)

print(len(pairs))
print(len(keep_pairs))
print(len(keep_pairs) / len(pairs))
pairs = keep_pairs

# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size=3):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch x seq) tensors, transpose into (seq x batch)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths

random_batch()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) # unpack (back to padded)
        return output, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
#         print('[attn] seq len', seq_len)
#         print('[attn] encoder_outputs', encoder_outputs.size()) # S x B x N
#         print('[attn] hidden', hidden.size()) # S=1 x B x N

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size, seq_len)) # B x S
#         print('[attn] attn_energies', attn_energies.size())

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
#         print('[attn] attn_energies', attn_energies.size())
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

rnn_output = Variable(torch.zeros(1, 2, 10))
encoder_outputs = Variable(torch.zeros(3, 2, 10))
attn = Attn('concat', 10)
attn(rnn_output, encoder_outputs)

attn_weights = torch.zeros(2, 1, 3)
print('attn_weights', attn_weights.size())
encoder_outputs = torch.zeros(3, 2, 10)
print('encoder_outputs', encoder_outputs.size())
#    B x N x M
#  , B x M x P
# -> B x N x P
context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
context = context.transpose(0, 1)
print('context', context.size())

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
#         print('[decoder] input_seq', input_seq.size()) # batch_size x 1
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, hidden_size) # S=1 x B x N
#         print('[decoder] word_embedded', embedded.size())

        # Get current hidden state from input word and last hidden state
#         print('[decoder] last_hidden', last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
#         print('[decoder] rnn_output', rnn_output.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
#         print('[decoder] attn_weights', attn_weights.size())
#         print('[decoder] encoder_outputs', encoder_outputs.size())
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
#         print('[decoder] context', context.size())

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
#         print('[decoder] rnn_output', rnn_output.size())
#         print('[decoder] context', context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6)
#         output = F.log_softmax(self.out(concat_output))
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


# ## Testing the models
# 
# To make sure the encoder and decoder are working (and working together) we'll do a quick test.
# 
# First by creating and padding a batch of sequences:

# In[394]:

# Input as batch of sequences of word indexes
batch_size = 2
input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)
print('input_batches', input_batches.size())
print('target_batches', target_batches.size())


# Create models with a small size (in case you need to manually inspect):

# In[395]:

# Create models
hidden_size = 8
n_layers = 2
encoder_test = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder_test = LuongAttnDecoderRNN('general', hidden_size, output_lang.n_words, n_layers)

if USE_CUDA:
    encoder_test.cuda()
    decoder_test.cuda()


# Then running the entire batch of input sequences through the encoder to get per-batch encoder outputs:

# In[396]:

# Test encoder
encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)
print('encoder_outputs', encoder_outputs.size()) # max_len x B x hidden_size
print('encoder_hidden', encoder_hidden.size()) # n_layers x B x hidden_size


# Then starting with a SOS token, run word tokens through the decoder to get each next word token. Instead of doing this with the whole sequence, it is done one at a time, to support using it's own predictions to make the next prediction. This will be one time step at a time, but batched per time step. In order to get this to work for short padded sequences, the batch size is going to get smaller each time.

# In[397]:

decoder_attns = torch.zeros(batch_size, MAX_LENGTH, MAX_LENGTH)
decoder_hidden = encoder_hidden
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))
criterion = nn.NLLLoss()

max_length = max(target_lengths)
all_decoder_outputs = Variable(torch.zeros(max_length, batch_size, decoder_test.output_size))

if USE_CUDA:
    decoder_context = decoder_context.cuda()
    all_decoder_outputs = all_decoder_outputs.cuda()

loss = 0

# import masked_cross_entropy
import importlib
importlib.reload(masked_cross_entropy)

# Run through decoder one time step at a time
for t in range(max_length - 1):
    decoder_input = target_batches[t]
    target_batch = target_batches[t + 1]

    decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs
    )
    print('decoder output = %s, decoder_hidden = %s, decoder_attn = %s'% (
        decoder_output.size(), decoder_hidden.size(), decoder_attn.size()
    ))
    all_decoder_outputs[t] = decoder_output

# print('all decoder outputs', all_decoder_outputs.size())
# print('target batches', target_batches.size())
# print('all_decoder_outputs', all_decoder_outputs.transpose(0, 1).contiguous().view(-1, decoder_test.output_size))
print('target lengths', target_lengths)
loss = masked_cross_entropy.compute_loss(
    all_decoder_outputs.transpose(0, 1).contiguous(),
    target_batches.transpose(0, 1).contiguous(),
    target_lengths
)
# loss = criterion(all_decoder_outputs, target_batches)
# print('loss', loss.size())
print('loss', loss.data[0])


# # Training
# 
# ## Defining a training iteration
# 
# To train we first run the input sentence through the encoder word by word, and keep track of every output and the latest hidden state. Next the decoder is given the last hidden state of the decoder as its first hidden state, and the `<SOS>` token as its first input. From there we iterate to predict a next token from the decoder.
# 
# ### Teacher Forcing vs. Scheduled Sampling
# 
# "Teacher Forcing", or maximum likelihood sampling, means using the real target outputs as each next input when training. The alternative is using the decoder's own guess as the next input. Using teacher forcing may cause the network to converge faster, but [when the trained network is exploited, it may exhibit instability](http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf).
# 
# You can observe outputs of teacher-forced networks that read with coherent grammar but wander far from the correct translation - you could think of it as having learned how to listen to the teacher's instructions, without learning how to venture out on its own.
# 
# The solution to the teacher-forcing "problem" is known as [Scheduled Sampling](https://arxiv.org/abs/1506.03099), which simply alternates between using the target values and predicted values when training. We will randomly choose to use teacher forcing with an if statement while training - sometimes we'll feed use real target as the input (ignoring the decoder's output), sometimes we'll use the decoder's output.

# In[398]:

[SOS_token] * 5


# In[399]:

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] * batch_size])).transpose(0, 1)
#     print('decoder_input', decoder_input.size())
    decoder_context = encoder_outputs[-1]
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder

    max_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    # TODO: Get targets working
    if True:
        # Run through decoder one time step at a time
        for t in range(max_length):
#             target_batch = target_batches[t]

            # Trim down batches of other inputs
#             decoder_hidden = decoder_hidden[:, :len(target_batch)]
#             encoder_outputs = encoder_outputs[:, :len(target_batch)]

            decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs
            )
#             print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())

#             loss += criterion(decoder_output, target_batch)
            all_decoder_outputs[t] = decoder_output

            decoder_input = target_batches[t]
            # TODO decoder_input = target_variable[di] # Next target is next input

    # Teacher forcing: Use the ground-truth target as the next input
    elif use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    # Without teacher forcing: use network's own prediction as the next input
    else:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Loss calculation and backpropagation
#     print('all_decoder_outputs', all_decoder_outputs.size())
#     print('target_batches', target_batches.size())
    loss = masked_cross_entropy.compute_loss(
        all_decoder_outputs.transpose(0, 1).contiguous(), # seq x batch -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # seq x batch -> batch x seq
        target_lengths
    )

    loss.backward()

    # Clip gradient norm
#     ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
#     dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc

# ## Running training

# Plus helper functions to print time elapsed and estimated time remaining, given the current time and progress.

# In[404]:

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def evaluate(input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [indexes_from_sentence(input_lang, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        # THIS MIGHT BE THE LAST PART OF BATCHING (or is it already going?)
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly():
    pair = random.choice(pairs)

    output_words, attentions = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    show_attention(pair[0], output_words, attentions)

    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')

def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    plt.show()
    plt.close()

def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    vis.text(text, win=win, opts={'title': win})


# Begin!
ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1

    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)

    # Run the train function
    loss, ec, dc = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec
    dca += dc

    if epoch == 1:
        evaluate_randomly()
        continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        evaluate_randomly()

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        # TODO: Running average helper
        ecs.append(eca / plot_every)
        dcs.append(dca / plot_every)
        ecs_win = 'encoder grad (%s)' % hostname
        dcs_win = 'decoder grad (%s)' % hostname
        vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
        vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
        eca = 0
        dca = 0

