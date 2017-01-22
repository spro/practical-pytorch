import torch
import glob
import unicodedata
import string

filenames = glob.glob('../data/names/*.txt')

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def stripAccents(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Read a file and split into lines
def readNames(filename):
    names = open(filename).read().strip().split('\n')
    return [stripAccents(name) for name in names]

# Build the lang_names dictionary, a list of names per language
lang_names = {}
all_langs = []
for filename in filenames:
    lang = filename.split('/')[-1].split('.')[0]
    all_langs.append(lang)
    names = readNames(filename)
    lang_names[lang] = names

n_langs = len(all_langs)

all_letters = string.ascii_letters
n_letters = len(all_letters)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a name into a <name_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def nameToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

