The [Jupyter Notebook version of the tutorial](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb) describes the model and steps in more detail.

Run `train.py` to train and save the network:

```
Usage: generate.py [options]

Options:
--n_epochs         Number of epochs to train
--print_every      Log learning rate at this interval
--hidden_size      Hidden size of GRU
--n_layers         Number of GRU layers
--learning_rate    Learning rate
--chunk_len        Length of chunks to train on at a time
```

Run `generate.py` with a name to view predictions:

```
Usage: generate.py [options]

Options:
-p, --prime_str      String to prime generation with
-l, --predict_len    Length of prediction
-t, --temperature    Temperature (higher is more chaotic)
```

```
> python generate.py --predict_len 100
AY:
Surt him, ach thend all for and for she had
Shy all word his be and herth the girse, for 'tis thi
```
