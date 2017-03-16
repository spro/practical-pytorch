# Practical PyTorch: Generating Shakespeare with a Character-Level RNN
# https://github.com/spro/practical-pytorch

import torch
from torch.autograd import Variable
import unidecode
import string
import random

# Prepare data

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('../data/shakespeare.txt').read())
file_len = len(file)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        i = all_characters.index(string[c])
        tensor[c] = i
    return Variable(tensor)

def random_training_set(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

