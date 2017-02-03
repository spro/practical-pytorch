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
* *WIP* Translation with Sequence to Sequence Networks
* *TODO* Intent parsing with Sequence to Sequence Networks and Attention

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
