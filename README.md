# Bachelor thesis: exploring algorithmic detection of fake news
In my thesis, I'm going to explore the following question: how well can pre-trained language embedding techniques classify fake news?
This research will be focussed on applying new techniques on [earlier research by Wang (2017)](https://arxiv.org/abs/1705.00648). Results of Wang will be used as a benchmark for performance. 

## Q1: How can fake news be defined and characterized?
Goals:
* Find a definition of fake news
* Find related algorithmic detection of fake news

## Q2: Which pre-trained models are available for embedding raw text?
Goals:
* Gather a list of possible word/sentence embeddings for later use
* Create embeddings for each of the listed techniques

## Q3: What is the performance of combinations of pre-trained embedding techniques with machine learning algorithms?
Goals:
* Define which classifiers can be used and make sense
* Compare classifier/embedding combinations

## Q4: To what extent can performance of fake news classifiers be improved with increased amounts of raw data?
Goals:
* Scrape additional statements from [PolitiFact](https://www.politifact.com/)
* Concatenate the original dataset with the newly aggregated statements
* Retrain the classifier/embedding combinations from Q3 with the new dataset