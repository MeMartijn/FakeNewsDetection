# Bachelor thesis: exploring algorithmic detection of fake news
In my thesis, I'm going to explore the following question: how well can pre-trained language embedding techniques classify fake news?
This research will be focussed on applying transfer learning on [earlier research by Wang (2017)](https://arxiv.org/abs/1705.00648). Results of Wang will be used as a benchmark for performance. 

## Table of contents
1. [Research questions](#rq)
2. [Results](#results)

<a name="rq"/>

## Research questions
### Q1: How can fake news be defined?
Goals:
* Find a definition of fake news
* Find related algorithmic detection of fake news

### Q2: Which pre-trained models are available for embedding raw text?
Goals:
* ~~Gather a list of possible word/sentence embeddings for later use~~
* Create embeddings for each of the listed techniques

### Q3: What is the performance of combinations of pre-trained embedding techniques with machine learning algorithms?
Goals:
* ~~Define which classifiers can be used and make sense~~
* Compare classifier/embedding combinations

<a name="results"/>

## (Interim) results
|                     | Bag of Words                       | InferSent | ELMo | BERT | GPT-2 | Transformer-XL | MT-DNN |
|---------------------|------------------------------------|-----------|------|------|-------|----------------|--------|
| SVM                 | Test: 0.249&nbsp;Validation: 0.251 |           |      |      |       |                |        |
| Logistic regression |                                    |           |      |      |       |                |        |
| Bi-LSTM             |                                    |           |      |      |       |                |        |
| CNN                 |                                    |           |      |      |       |                |        |
