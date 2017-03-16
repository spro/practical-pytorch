# Practical PyTorch: Generating Names with a Conditional Character-Level RNN
# https://github.com/spro/practical-pytorch

import sys

if len(sys.argv) < 2:
    print("Usage: generate.py [language]")
    sys.exit()

else:
    language = sys.argv[1]

import torch
import torch.nn as nn
from torch.autograd import Variable

from data import *
from model import *

rnn = torch.load('conditional-char-rnn.pt')

# Generating from the Network

max_length = 20

def generate_one(category, start_char='A', temperature=0.5):
    category_input = make_category_input(category)
    chars_input = make_chars_input(start_char)
    hidden = rnn.init_hidden()

    output_str = start_char
    
    for i in range(max_length):
        output, hidden = rnn(category_input, chars_input[0], hidden)
        
        # Sample as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Stop at EOS, or add to output_str
        if top_i == EOS:
            break
        else:    
            char = all_letters[top_i]
            output_str += char
            chars_input = make_chars_input(char)

    return output_str

def generate(category, start_chars='ABC'):
    for start_char in start_chars:
        print(generate_one(category, start_char))

generate(language)
