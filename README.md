![Practical Pytorch](https://i.imgur.com/eBRPvWB.png)

Learn PyTorch with project-based tutorials. These tutorials demonstrate modern techniques with readable code and use regular data from the internet.

## Tutorials

#### Series 1: RNNs for NLP

Applying recurrent neural networks to natural language tasks, from classification to generation.

* [Classifying Names with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb)
* [Generating Shakespeare with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb)
* [Generating Names with a Conditional Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/conditional-char-rnn/conditional-char-rnn.ipynb)
* [Translation with a Sequence to Sequence Network and Attention](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
* [Exploring Word Vectors with GloVe](https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb)
* *WIP* Sentiment Analysis with a Word-Level RNN and GloVe Embeddings

#### Series 2: RNNs for timeseries data

* *WIP* Predicting discrete events with an RNN

## Get Started

The quickest way to run these on a fresh Linux or Mac machine is to install [Anaconda](https://www.continuum.io/anaconda-overview):
```
curl -LO https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
bash Anaconda3-4.3.0-Linux-x86_64.sh
```

Then install PyTorch:

```
conda install pytorch -c soumith
```

Then clone this repo and start Jupyter Notebook:

```
git clone http://github.com/spro/practical-pytorch
cd practical-pytorch
jupyter notebook
```

## Recommended Reading

### PyTorch basics

* http://pytorch.org/ For installation instructions
* [Offical PyTorch tutorials](http://pytorch.org/tutorials/) for more tutorials (some of these tutorials are included there)
* [Deep Learning with PyTorch: A 60-minute Blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) to get started with PyTorch in general
* [Introduction to PyTorch for former Torchies](https://github.com/pytorch/tutorials/blob/master/Introduction%20to%20PyTorch%20for%20former%20Torchies.ipynb) if you are a former Lua Torch user
* [jcjohnson's PyTorch examples](https://github.com/jcjohnson/pytorch-examples) for a more in depth overview (including custom modules and autograd functions)

### Recurrent Neural Networks

* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows a bunch of real life examples
* [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) for an overview on word embeddings and RNNs for NLP
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is about LSTMs work specifically, but also informative about RNNs in general

### Machine translation

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078)
* [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215)

### Attention models

* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

### Other RNN uses

* [A Neural Conversational Model](http://arxiv.org/abs/1506.05869)

### Other PyTorch tutorials

* [Deep Learning For NLP In PyTorch](https://github.com/rguthrie3/DeepLearningForNLPInPytorch)

## Feedback

If you have ideas or find mistakes [please leave a note](https://github.com/spro/practical-pytorch/issues/new).
