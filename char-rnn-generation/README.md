# Practical PyTorch: Generating Shakespeare with a Character-Level RNN

## Dataset

Download [this Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (from [Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn)) and save as `shakespeare.txt`

## Jupyter Notebook

The [Jupyter Notebook version of the tutorial](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb) describes the model and steps in detail.

## Python scripts

Run `train.py` with a filename to train and save the network:

```
> python train.py shakespeare.txt

Training for 2000 epochs...
(10 minutes later)
Saved as shakespeare.pt
```

After training the model will be saved as `[filename].pt` &mdash; now run `generate.py` with that filename to generate some new text:

```
> python generate.py shakespeare.pt --prime_str "Where"

Where, you, and if to our with his drid's
Weasteria nobrand this by then.

AUTENES:
It his zersit at he
```

### Training options

```
Usage: train.py [filename] [options]

Options:
--n_epochs         Number of epochs to train
--print_every      Log learning rate at this interval
--hidden_size      Hidden size of GRU
--n_layers         Number of GRU layers
--learning_rate    Learning rate
--chunk_len        Length of chunks to train on at a time
```

### Generation options
```
Usage: generate.py [filename] [options]

Options:
-p, --prime_str      String to prime generation with
-l, --predict_len    Length of prediction
-t, --temperature    Temperature (higher is more chaotic)
```

