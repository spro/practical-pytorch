# https://github.com/spro/practical-pytorch
# -*- coding: utf-8 -*-

import torch

from helpers import *
from model import *

def generate(decoder, all_characters, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str, all_characters)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)

    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char, all_characters)

    return predicted

if __name__ == '__main__':
    # Parse command line arguments
    import argparse, pickle
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('-f', '--charset-file', type=str, default='charset.pickle')
    args = argparser.parse_args()
    print args
    with open(args.charset_file) as fd:
        all_characters = pickle.load(fd)
    decoder = torch.load(args.filename)
    del args.filename
    del args.charset_file
    print all_characters
    #print(generate(decoder=decoder, all_characters=all_characters, **vars(args)))
    print generate(decoder, all_characters=all_characters, prime_str='अध्याय', predict_len=500)
