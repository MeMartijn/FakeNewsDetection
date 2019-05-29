# Bachelor thesis: exploring algorithmic detection of fake news
In my thesis, I'm going to explore the following question: what is the performance of combinations of pre-trained embedding techniques with machine learning algorithms when classifying fake news?
This research will be focussed on applying transfer learning on [earlier research by Wang (2017)](https://arxiv.org/abs/1705.00648). Results of Wang will be used as a benchmark for performance. 

## Table of contents
1. [Requirements](#requirements)
2. [Research questions](#rq)

<a name="requirements"/>

## Requirements
To run the code in the `code` folder, the following packages must be installed (using pip):
- `flair`
- `allennlp`
- `tensorflow`
- `tensorflow_hub`
- `pytorch`
- `pytorch_pretrained_bert`

<a name="rq"/>

## Research questions
#### Which way of pooling vectors to a fixed length works best for classifying fake news?
#### At what padding sequence length do neural networks hold the highest accuracy when classifying fake news?
#### How well do neural network classification architectures classify fake news compared to non-neural classification algorithms?