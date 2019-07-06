# Fake news: approaching automated lie detection using pre-trained word embeddings
In this repository, the following research question is explored: what is the performance of combinations of pre-trained embedding techniques with machine learning algorithms when classifying fake news?
This research will be focussed on applying transfer learning on [earlier research by Wang (2017)](https://arxiv.org/abs/1705.00648). Results of Wang will be used as a benchmark for performance. 

## Table of contents
1. [Requirements](#requirements)
2. [Research questions](#rq)
3. [Results](#results)

<a name="requirements"/>

## Requirements
To run the code in the `code` folder, the following packages must be installed:
- `flair`
- `allennlp`
- `tensorflow`
- `tensorflow_hub`
- `pytorch`
- `pytorch_pretrained_bert`
- `spacy`
- `hypopt`
- `gensim`
- `fasttext`

You can install these packages by running `pip install -r /code/requirements.txt`. 

<a name="rq"/>

## Research questions
#### Which way of pooling vectors to a fixed length works best for classifying fake news?
#### At what padding sequence length do neural networks hold the highest accuracy when classifying fake news?
#### How well do neural network classification architectures classify fake news compared to non-neural classification algorithms?

<a name="results">

## Results
![Experiment results](https://imgur.com/9E87eEb.png)

With a combination of BERT embeddings and a logistic regression, an accuracy of 52.96% on 3 labels can be achieved, which is an increase of almost 4% compared to [previous research in which only traditional linguistic methods were used](https://esc.fnwi.uva.nl/thesis/centraal/files/f1840275767.pdf). 
On the original 6 labels, this combination achieves an accuracy of 27.51%, which is 0.51% better than the [original research by Wang (2017)](https://arxiv.org/abs/1705.00648). 