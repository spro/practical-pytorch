from model import *
from data import *
import sys

rnn = torch.load('char-rnn-classification.pt')

# Just return an output given a name
def evaluate(input_name):
    input = Variable(nameToTensor(input_name))
    hidden = rnn.emptyHidden()

    for i in range(len(input_name)):
        output, hidden = rnn(input[i], hidden)

    return output

def predict(input_name):
    output = evaluate(input_name)

    # Get top 3 languages
    top3v, top3i = output.data.topk(3, 1, True)

    for i in range(3):
        value = top3v[0][i]
        lang_index = top3i[0][i]
        print('(%.2f) %s' % (value, all_langs[lang_index]))

predict(sys.argv[1])
