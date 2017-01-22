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

def predict(input_name, n_predictions=3):
    output = evaluate(input_name)

    # Get top N languages
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        lang_index = topi[0][i]
        print('(%.2f) %s' % (value, all_langs[lang_index]))
        predictions.append([value, all_langs[lang_index]])

    return predictions

if __name__ == '__main__':
    predict(sys.argv[1])
