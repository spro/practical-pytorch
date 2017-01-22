import torch
from data import *
from model import *
import random

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_langs)

def classFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    class_i = top_i[0][0]
    return all_langs[class_i], class_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    lang = randomChoice(all_langs)
    name = randomChoice(lang_names[lang])
    return lang, name

criterion = nn.NLLLoss()

def train(target_lang, input_name):
    input = Variable(nameToTensor(input_name))
    target = Variable(torch.LongTensor([all_langs.index(target_lang)]))
    hidden = Variable(torch.zeros(1, n_hidden))

    rnn.zero_grad()

    for i in range(len(input_name)):
        output, hidden = rnn(input[i], hidden)

    loss = criterion(output, target)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-0.005, p.grad.data)

    return output, loss.data[0]

n_epochs = 100000
print_every = 5000
plot_every = 1000

rnn = RNN(n_letters, n_hidden, n_langs)

# Keep track of losses for plotting
current_loss = 0
all_losses = []

for epoch in range(1, n_epochs):
    target_lang, input_name = randomTrainingPair()
    output, loss = train(target_lang, input_name)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = classFromOutput(output)
        correct = '✓' if guess == target_lang else '✗ (%s)' % target_lang
        print('%d (%.2f) %s / %s %s' % (epoch, current_loss, input_name, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

