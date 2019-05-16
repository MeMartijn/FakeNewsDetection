# Bachelor thesis: exploring algorithmic detection of fake news
In my thesis, I'm going to explore the following question: what is the performance of combinations of pre-trained embedding techniques with machine learning algorithms?
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
### Q1: How well do Transformer architectures capture relevant information from text compared to non-neural text embeddings?
Hypothesis: Transformer architectures perform better because the vectors created are much more accurate due to context being explicitly taken into consideration upon generation.

### Q2: How well do neural network architecture classify fake news compared to non-neural classification algorithms?
Hypothesis: because neural networks (especially Bi-LSTMs) keep track of sequences in the vector sequences, they outperform non-neural classification algorithms.