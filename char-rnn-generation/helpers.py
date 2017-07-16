# https://github.com/spro/practical-pytorch

import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable

# Reading and un-unicode-encoding data

#all_characters = string.printable
#n_characters = len(all_characters)
#all_characters = ['\x83', '\x87', '\x8b', '\x8f', '\x93', '\x97', '\x9b', '\x9f', ' ', '\xa3', '\xa7', '(', '\xab', ',', '\xaf', '\xb7', '\xbb', '\xbf', '\xc3', 'H', 'L', 'P', 'T', 'd', 'h', 'l', '\xef', 'p', 't', '|', '\x80', '\x88', '\x8c', '\x90', '\x94', '\x98', '\x9c', '\xa0', '\xa4', "'", '\xa8', '\xac', '/', '\xb0', '\xb8', '\xbc', '?', 'C', 'S', 'W', '\xe0', 'c', 'g', 'k', 'o', 's', 'w', '\x81', '\x85', '\x89', '\n', '\x8d', '\x95', '\x99', '\x9d', '\xa1', '\xa5', '*', '\xad', '.', '\xb1', '\xb5', '\xb9', ':', '\xbd', 'B', 'F', 'J', 'V', 'b', 'f', 'n', 'r', 'v', 'z', '~', '\x82', '\x86', '\x8a', '\x96', '\x9a', '\x9e', '!', '\xa2', '\xa6', ')', '\xaa', '-', '\xae', '\xb2', '\xb6', '\xbe', 'A', '\xc2', 'E', 'I', 'M', 'U', 'Y', 'a', '\xe2', 'e', 'i', 'm', 'q', 'u', 'y']
#n_characters = len(all_characters)

def read_file(filename):
    #global all_characters
    #global n_characters
    s = open(filename).read()
    all_characters = [i for i in set(s)]
    n_characters = len(all_characters)
    return s, len(s), all_characters, n_characters

# Turning a string into a tensor

def char_tensor(s, all_characters):
    tensor = torch.zeros(len(s)).long()
    for c in range(len(s)):
        tensor[c] = all_characters.index(s[c])
    return Variable(tensor)

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
