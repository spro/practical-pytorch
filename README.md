# Practical PyTorch

Learn PyTorch with project-based tutorials. So far they are focused on applying recurrent neural networks to natural language tasks.

These tutorials aim to:

* Acheive specific goals with minimal parts
* Demonstrate modern techniques with common data
* Use low level but low complexity models
* Reach for readablity over efficiency

## Tutorials

* [Classifying Names with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb)
* [Generating Names with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb)
* [Translation with a Sequence to Sequence Network and Attention](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
* *WIP* Intent Parsing and Slot Filling with Pointer Networks

# Recommended Reading

I assume you have at least installed PyTorch, know Python, and understand Tensors:

* http://pytorch.org/ For installation instructions
* [Deep Learning with PyTorch: A 60-minute Blitz](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb) to get started with PyTorch in general
* [jcjohnson's PyTorch examples](https://github.com/jcjohnson/pytorch-examples) for a wide and deep overview
* [Introduction to PyTorch for former Torchies](https://github.com/pytorch/tutorials/blob/master/Introduction%20to%20PyTorch%20for%20former%20Torchies.ipynb) if you are former Lua Torch user

It would also be useful to know about Recurrent Neural Networks and how they work:

* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows a bunch of real life examples
* [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) for an overview on word embeddings and RNNs for NLP
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is about LSTMs specifically but also informative about RNNs in general

And the papers that introduced many of these topics:

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078)
* [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/abs/1409.3215)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [A Neural Conversational Model](http://arxiv.org/abs/1506.05869)

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
